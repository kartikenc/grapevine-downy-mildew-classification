#!/usr/bin/env python3
"""
16_edge_benchmark.py
====================
Edge Inference Benchmark on Jetson Xavier NX.
Measures latency, throughput, power, and model size for
selected models from the downy mildew severity study.

Metrics:
  - PyTorch GPU/CPU inference latency (ms)
  - ONNX Runtime CPU inference latency (ms)
  - ONNX model size (MB)
  - PyTorch model parameters & FLOPs
  - Throughput (images/sec)
  - Power draw via tegrastats (W)

Author: Kartik E. Cholachgudda
Date: June 2026
"""

import os
import sys
import time
import json
import subprocess
import threading
import re
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm
import onnx

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("WARNING: onnxruntime not available")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ──────────────────────── Configuration ────────────────────────
IMAGE_SIZE     = 384
NUM_CLASSES    = 5
BATCH_SIZE     = 1      # Single-image inference for latency
WARMUP_RUNS    = 20     # GPU warmup iterations
BENCH_RUNS     = 100    # Benchmark iterations
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR      = Path(os.path.expanduser("~/benchmark/models"))
RESULTS_DIR    = Path(os.path.expanduser("~/benchmark/results"))
ONNX_DIR       = Path(os.path.expanduser("~/benchmark/onnx"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────── Model Definitions ────────────────────────
def build_model(name):
    """Build model architecture matching training code exactly."""
    if name == 'vit_s_16':
        m = timm.create_model('vit_small_patch16_224', pretrained=False,
                              num_classes=NUM_CLASSES, img_size=IMAGE_SIZE)
    elif name == 'maxvit_t':
        m = timm.create_model('maxvit_tiny_tf_384', pretrained=False,
                              num_classes=NUM_CLASSES)
    elif name == 'convnext_tiny':
        m = tv_models.convnext_tiny(weights=None)
        in_feat = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
    elif name == 'mobilenetv2':
        m = tv_models.mobilenet_v2(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
    elif name == 'efficientnet_b0':
        m = tv_models.efficientnet_b0(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
    elif name == 'mobilenetv4_s':
        m = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k',
                              pretrained=False, num_classes=NUM_CLASSES)
    elif name == 'fastvit_t8':
        m = timm.create_model('fastvit_t8.apple_in1k', pretrained=False,
                              num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m

# Models to benchmark (edge-relevant subset)
MODELS = {
    'vit_s_16':       'model_vit_s_16.pt',
    'maxvit_t':       'model_maxvit_t.pt',
    'convnext_tiny':  'model_convnext_tiny.pt',
    'mobilenetv2':    'model_mobilenetv2.pt',
    'efficientnet_b0':'model_efficientnet_b0.pt',
    'mobilenetv4_s':  'model_mobilenetv4_s.pt',
    'fastvit_t8':     'model_fastvit_t8.pt',
}

# ──────────────────────── Power Monitor ────────────────────────
class PowerMonitor:
    """Read GPU/SoC power via tegrastats on Jetson."""
    def __init__(self):
        self.readings = []
        self._proc = None
        self._thread = None
        self._stop = False

    def start(self):
        self._stop = False
        self.readings = []
        try:
            self._proc = subprocess.Popen(
                ['tegrastats', '--interval', '200'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        except FileNotFoundError:
            print("  tegrastats not found, skipping power measurement")

    def _read_loop(self):
        while not self._stop and self._proc and self._proc.poll() is None:
            line = self._proc.stdout.readline()
            if line:
                # Parse VDD_GPU_SOC or VDD_IN power from tegrastats
                # Format: "VDD_GPU_SOC 1234mW/1500mW" or "VDD_IN 3456mW/5000mW"
                for pattern in [r'VDD_GPU_SOC\s+(\d+)mW', r'VDD_IN\s+(\d+)mW',
                                r'VDD_CPU_GPU_CV\s+(\d+)mW', r'POM_5V_IN\s+(\d+)mW']:
                    match = re.search(pattern, line)
                    if match:
                        self.readings.append(int(match.group(1)))
                        break

    def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=2)
        return self.readings

    def mean_power_w(self):
        if self.readings:
            return np.mean(self.readings) / 1000.0  # mW -> W
        return None

# ──────────────────────── Benchmark Functions ────────────────────────
def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def estimate_flops(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE)):
    """Rough FLOPs estimate using torch profiler."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        inp = torch.randn(*input_size).to(DEVICE)
        model.eval().to(DEVICE)
        with FlopCounterMode(display=False) as flop_counter:
            model(inp)
        return flop_counter.get_total_flops()
    except:
        return None

def benchmark_pytorch(model, device, runs=BENCH_RUNS, warmup=WARMUP_RUNS):
    """Benchmark PyTorch inference latency."""
    model.eval().to(device)
    dummy = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    return latencies

def export_onnx(model, name):
    """Export model to ONNX format."""
    onnx_path = ONNX_DIR / f"{name}.onnx"
    if onnx_path.exists():
        print(f"    ONNX already exists: {onnx_path.name}")
        return onnx_path

    model.eval().to('cpu')
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    try:
        torch.onnx.export(
            model, dummy, str(onnx_path),
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        # Verify
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"    ONNX exported: {onnx_path.stat().st_size / 1e6:.1f} MB")
        return onnx_path
    except Exception as e:
        print(f"    ONNX export FAILED: {e}")
        return None

def benchmark_onnx(onnx_path, runs=BENCH_RUNS, warmup=WARMUP_RUNS):
    """Benchmark ONNX Runtime inference latency."""
    if not HAS_ORT or onnx_path is None:
        return None

    providers = ort.get_available_providers()
    # Prefer CUDA > TensorRT > CPU
    if 'CUDAExecutionProvider' in providers:
        use_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        use_providers = ['CPUExecutionProvider']

    session = ort.InferenceSession(str(onnx_path), providers=use_providers)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies, use_providers[0]

# ──────────────────────── Main ────────────────────────
def main():
    print("=" * 70)
    print("  Edge Inference Benchmark — Jetson Xavier NX")
    print(f"  Device: {DEVICE}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Benchmark Runs: {BENCH_RUNS}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {}

    for model_name, weight_file in MODELS.items():
        weight_path = MODEL_DIR / weight_file
        if not weight_path.exists():
            print(f"\n  SKIP {model_name}: weight file not found")
            continue

        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")

        # Build & load model
        try:
            model = build_model(model_name)
            state = torch.load(str(weight_path), map_location='cpu')
            model.load_state_dict(state)
            print(f"  Loaded weights: {weight_file}")
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            continue

        # Model stats
        total_params, _ = count_parameters(model)
        model_size_mb = weight_path.stat().st_size / 1e6
        print(f"  Parameters: {total_params:,}")
        print(f"  Model file: {model_size_mb:.1f} MB")

        # FLOPs
        flops = estimate_flops(model)
        if flops:
            print(f"  FLOPs: {flops / 1e9:.2f} G")

        entry = {
            'model': model_name,
            'params': total_params,
            'model_size_mb': round(model_size_mb, 1),
            'flops_g': round(flops / 1e9, 2) if flops else None,
        }

        # PyTorch GPU benchmark
        if torch.cuda.is_available():
            print(f"\n  [PyTorch GPU] Benchmarking ({BENCH_RUNS} runs)...")
            power_mon = PowerMonitor()
            power_mon.start()
            gpu_latencies = benchmark_pytorch(model, DEVICE)
            power_readings = power_mon.stop()

            entry['pytorch_gpu_latency_ms'] = {
                'mean': round(np.mean(gpu_latencies), 2),
                'std': round(np.std(gpu_latencies), 2),
                'median': round(np.median(gpu_latencies), 2),
                'p95': round(np.percentile(gpu_latencies, 95), 2),
                'p99': round(np.percentile(gpu_latencies, 99), 2),
                'min': round(np.min(gpu_latencies), 2),
                'max': round(np.max(gpu_latencies), 2),
            }
            entry['pytorch_gpu_throughput'] = round(1000.0 / np.mean(gpu_latencies), 1)
            entry['power_w'] = round(power_mon.mean_power_w(), 2) if power_mon.mean_power_w() else None

            print(f"    Mean: {np.mean(gpu_latencies):.2f} ms")
            print(f"    Median: {np.median(gpu_latencies):.2f} ms")
            print(f"    P95: {np.percentile(gpu_latencies, 95):.2f} ms")
            print(f"    Throughput: {entry['pytorch_gpu_throughput']:.1f} img/s")
            if entry['power_w']:
                print(f"    Power: {entry['power_w']:.1f} W")

        # PyTorch CPU benchmark (fewer runs since it's slower)
        print(f"\n  [PyTorch CPU] Benchmarking (50 runs)...")
        model_cpu = model.to('cpu')
        cpu_latencies = benchmark_pytorch(model_cpu, torch.device('cpu'),
                                          runs=50, warmup=5)
        entry['pytorch_cpu_latency_ms'] = {
            'mean': round(np.mean(cpu_latencies), 2),
            'median': round(np.median(cpu_latencies), 2),
            'p95': round(np.percentile(cpu_latencies, 95), 2),
        }
        entry['pytorch_cpu_throughput'] = round(1000.0 / np.mean(cpu_latencies), 1)
        print(f"    Mean: {np.mean(cpu_latencies):.2f} ms")
        print(f"    Throughput: {entry['pytorch_cpu_throughput']:.1f} img/s")

        # ONNX export & benchmark
        print(f"\n  [ONNX] Exporting...")
        onnx_path = export_onnx(model_cpu, model_name)
        if onnx_path and HAS_ORT:
            entry['onnx_size_mb'] = round(onnx_path.stat().st_size / 1e6, 1)
            print(f"  [ONNX Runtime] Benchmarking (50 runs)...")
            ort_result = benchmark_onnx(onnx_path, runs=50, warmup=10)
            if ort_result:
                ort_latencies, provider = ort_result
                entry['ort_provider'] = provider
                entry['ort_latency_ms'] = {
                    'mean': round(np.mean(ort_latencies), 2),
                    'median': round(np.median(ort_latencies), 2),
                    'p95': round(np.percentile(ort_latencies, 95), 2),
                }
                entry['ort_throughput'] = round(1000.0 / np.mean(ort_latencies), 1)
                print(f"    Provider: {provider}")
                print(f"    Mean: {np.mean(ort_latencies):.2f} ms")
                print(f"    Throughput: {entry['ort_throughput']:.1f} img/s")

        results[model_name] = entry

        # Cleanup
        del model, model_cpu
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ──────────────── Summary Table ────────────────
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    header = f"{'Model':<18} {'Params':>10} {'Size(MB)':>9} {'GPU(ms)':>9} {'CPU(ms)':>9} {'ORT(ms)':>9} {'GPU fps':>8} {'Power(W)':>9}"
    print(header)
    print("-" * 100)
    for name, r in sorted(results.items(), key=lambda x: x[1].get('pytorch_gpu_latency_ms', {}).get('mean', 9999)):
        gpu_ms = r.get('pytorch_gpu_latency_ms', {}).get('mean', '-')
        cpu_ms = r.get('pytorch_cpu_latency_ms', {}).get('mean', '-')
        ort_ms = r.get('ort_latency_ms', {}).get('mean', '-')
        gpu_fps = r.get('pytorch_gpu_throughput', '-')
        power = r.get('power_w', '-')
        print(f"  {name:<16} {r['params']:>10,} {r['model_size_mb']:>9.1f} {gpu_ms:>9} {cpu_ms:>9} {ort_ms:>9} {gpu_fps:>8} {power:>9}")

    # Save results
    out_path = RESULTS_DIR / 'edge_benchmark_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'device': 'Jetson Xavier NX',
            'cuda': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'benchmark_runs': BENCH_RUNS,
            'timestamp': datetime.now().isoformat(),
            'models': results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("=" * 70)
    print("  Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
