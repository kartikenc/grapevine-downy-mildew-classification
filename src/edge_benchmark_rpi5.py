#!/usr/bin/env python3
"""
Edge Benchmark — Raspberry Pi 5 (CPU only)
Measures PyTorch CPU and ONNX Runtime CPU inference on RPi 5.
"""
import os, sys, time, json, gc
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

IMAGE_SIZE = 384
NUM_CLASSES = 5
BATCH_SIZE = 1
WARMUP_RUNS = 5
BENCH_RUNS = 30   # Fewer runs since CPU is slow

MODEL_DIR = Path(os.path.expanduser("~/benchmark/models"))
RESULTS_DIR = Path(os.path.expanduser("~/benchmark/results"))
ONNX_DIR = Path(os.path.expanduser("~/benchmark/onnx"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR.mkdir(parents=True, exist_ok=True)

def build_model(name):
    if name == 'vit_s_16':
        return timm.create_model('vit_small_patch16_224', pretrained=False,
                                 num_classes=NUM_CLASSES, img_size=IMAGE_SIZE)
    elif name == 'maxvit_t':
        return timm.create_model('maxvit_tiny_tf_384', pretrained=False,
                                 num_classes=NUM_CLASSES)
    elif name == 'convnext_tiny':
        m = tv_models.convnext_tiny(weights=None)
        in_feat = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
        return m
    elif name == 'mobilenetv2':
        m = tv_models.mobilenet_v2(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
        return m
    elif name == 'efficientnet_b0':
        m = tv_models.efficientnet_b0(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
        return m
    elif name == 'fastvit_t8':
        return timm.create_model('fastvit_t8.apple_in1k', pretrained=False,
                                 num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")

MODELS = {
    'mobilenetv2':    'model_mobilenetv2.pt',
    'efficientnet_b0':'model_efficientnet_b0.pt',
    'fastvit_t8':     'model_fastvit_t8.pt',
    'convnext_tiny':  'model_convnext_tiny.pt',
    'vit_s_16':       'model_vit_s_16.pt',
    'maxvit_t':       'model_maxvit_t.pt',
}

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_pytorch_cpu(model, runs=BENCH_RUNS, warmup=WARMUP_RUNS):
    model.eval().cpu()
    dummy = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies

def export_onnx(model, name):
    onnx_path = ONNX_DIR / f"{name}.onnx"
    if onnx_path.exists():
        print(f"    ONNX exists: {onnx_path.name}")
        return onnx_path
    model.eval().cpu()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    try:
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"    ONNX exported: {onnx_path.stat().st_size / 1e6:.1f} MB")
        return onnx_path
    except Exception as e:
        print(f"    ONNX FAILED: {e}")
        return None

def benchmark_ort(onnx_path, runs=BENCH_RUNS, warmup=WARMUP_RUNS):
    if not HAS_ORT or not onnx_path:
        return None
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies

def main():
    print("=" * 70)
    print("  Edge Benchmark -- Raspberry Pi 5 (CPU only)")
    print(f"  Image: {IMAGE_SIZE}x{IMAGE_SIZE} | Batch: {BATCH_SIZE} | Runs: {BENCH_RUNS}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {}
    for name, wf in MODELS.items():
        wp = MODEL_DIR / wf
        if not wp.exists():
            print(f"\n  SKIP {name}: not found")
            continue
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        try:
            model = build_model(name)
            state = torch.load(str(wp), map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            print(f"  Loaded: {wf}")
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            # Try without weights_only
            try:
                state = torch.load(str(wp), map_location='cpu', weights_only=False)
                model.load_state_dict(state)
                print(f"  Loaded (legacy): {wf}")
            except Exception as e2:
                print(f"  LOAD ERROR (retry): {e2}")
                continue

        params = count_params(model)
        size_mb = wp.stat().st_size / 1e6
        print(f"  Params: {params:,} | Size: {size_mb:.1f} MB")

        entry = {'model': name, 'params': params, 'model_size_mb': round(size_mb, 1)}

        # PyTorch CPU
        print(f"  [PyTorch CPU] {BENCH_RUNS} runs...")
        cpu_lat = benchmark_pytorch_cpu(model)
        entry['pytorch_cpu_ms'] = {
            'mean': round(np.mean(cpu_lat), 1),
            'median': round(np.median(cpu_lat), 1),
            'p95': round(np.percentile(cpu_lat, 95), 1),
            'min': round(np.min(cpu_lat), 1),
            'max': round(np.max(cpu_lat), 1),
        }
        entry['pytorch_cpu_fps'] = round(1000.0 / np.mean(cpu_lat), 2)
        print(f"    Mean: {np.mean(cpu_lat):.1f} ms | FPS: {entry['pytorch_cpu_fps']:.2f}")

        # ONNX export + benchmark
        print(f"  [ONNX] Exporting (opset 17)...")
        onnx_path = export_onnx(model, name)
        if onnx_path:
            entry['onnx_size_mb'] = round(onnx_path.stat().st_size / 1e6, 1)
            print(f"  [ORT CPU] {BENCH_RUNS} runs...")
            ort_lat = benchmark_ort(onnx_path)
            if ort_lat:
                entry['ort_cpu_ms'] = {
                    'mean': round(np.mean(ort_lat), 1),
                    'median': round(np.median(ort_lat), 1),
                    'p95': round(np.percentile(ort_lat, 95), 1),
                }
                entry['ort_cpu_fps'] = round(1000.0 / np.mean(ort_lat), 2)
                speedup = np.mean(cpu_lat) / np.mean(ort_lat)
                print(f"    Mean: {np.mean(ort_lat):.1f} ms | FPS: {entry['ort_cpu_fps']:.2f} | Speedup: {speedup:.1f}x")

        results[name] = entry
        del model
        gc.collect()

    # Summary
    print("\n" + "=" * 90)
    print("  SUMMARY — Raspberry Pi 5")
    print("=" * 90)
    hdr = f"{'Model':<18} {'Params':>10} {'Size':>7} {'CPU(ms)':>9} {'ORT(ms)':>9} {'CPU FPS':>8} {'ORT FPS':>8}"
    print(hdr)
    print("-" * 90)
    for n, r in sorted(results.items(), key=lambda x: x[1].get('pytorch_cpu_ms', {}).get('mean', 9999)):
        cpu = r.get('pytorch_cpu_ms', {}).get('mean', '-')
        ort_v = r.get('ort_cpu_ms', {}).get('mean', '-')
        cfps = r.get('pytorch_cpu_fps', '-')
        ofps = r.get('ort_cpu_fps', '-')
        print(f"  {n:<16} {r['params']:>10,} {r['model_size_mb']:>6.1f}M {cpu:>9} {ort_v:>9} {cfps:>8} {ofps:>8}")

    out_path = RESULTS_DIR / 'edge_benchmark_rpi5.json'
    with open(out_path, 'w') as f:
        json.dump({
            'device': 'Raspberry Pi 5 8GB',
            'cpu': 'Cortex-A76 4-core',
            'ram_gb': 8,
            'pytorch_version': torch.__version__,
            'ort_version': ort.__version__ if HAS_ORT else None,
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'benchmark_runs': BENCH_RUNS,
            'timestamp': datetime.now().isoformat(),
            'models': results,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
