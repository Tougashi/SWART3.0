"""
SWART - Benchmark beberapa model ONNX (latensi, throughput, suhu)
================================================================
Dijalankan di Raspberry Pi 4 untuk membandingkan model lama vs baru.
Bisa juga dijalankan di PC untuk tes awal.

Pemakaian:
    python bench_models.py weights/best2_main.onnx weights/yolo11n.onnx --imgsz 320 --threads 4
    python bench_models.py weights/best2_main.onnx weights/yolo11n.onnx weights/yolo26n.onnx --imgsz 320
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from swart_core import OnnxDetector


def get_cpu_temp():
    """Baca suhu CPU (hanya Linux/Raspberry Pi)."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def bench_one(model_path, imgsz, threads, warmup, runs, images):
    """Benchmark satu model. Return dict hasil."""
    det = OnnxDetector(model_path, imgsz, conf=0.25, iou=0.45, threads=threads, verbose=False)

    # Warmup
    dummy = np.full((480, 640, 3), 114, np.uint8)
    for _ in range(warmup):
        det.detect(dummy)

    # Benchmark dengan dummy image
    times_dummy = []
    for _ in range(runs):
        t0 = time.perf_counter()
        det.detect(dummy)
        times_dummy.append((time.perf_counter() - t0) * 1000)

    # Benchmark dengan real images (jika ada)
    times_real = []
    dets_count = []
    for img in images:
        t0 = time.perf_counter()
        dets = det.detect(img)
        times_real.append((time.perf_counter() - t0) * 1000)
        dets_count.append(len(dets))

    temp = get_cpu_temp()
    if temp and temp >= 80:
        print(f"  [WARN] Suhu CPU {temp:.1f}°C >= 80°C! Pasang pendingin untuk hasil yang akurat.")

    result = {
        'model': Path(model_path).name,
        'imgsz': f"{det.w}x{det.h}",
        'threads': det.sess.get_session_options().intra_op_num_threads,
        'n_outputs': det.n_outputs,
        'dummy_ms_mean': np.mean(times_dummy),
        'dummy_ms_median': np.median(times_dummy),
        'dummy_ms_p95': np.percentile(times_dummy, 95),
        'dummy_ms_min': np.min(times_dummy),
        'dummy_fps': 1000.0 / np.mean(times_dummy),
        'temp_c': temp,
    }
    if times_real:
        result.update({
            'real_ms_mean': np.mean(times_real),
            'real_ms_median': np.median(times_real),
            'real_ms_p95': np.percentile(times_real, 95),
            'real_fps': 1000.0 / np.mean(times_real),
            'avg_dets': np.mean(dets_count),
        })
    return result


def load_test_images(path, max_imgs=50):
    """Load gambar test untuk benchmark realistis."""
    if path is None:
        return []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = sorted(p for p in Path(path).rglob('*') if p.suffix.lower() in exts)[:max_imgs]
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
    return imgs


def main():
    ap = argparse.ArgumentParser(description='Benchmark model ONNX untuk SWART')
    ap.add_argument('models', nargs='+', help='path ke file .onnx')
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--threads', type=int, default=None, help='thread ONNX (default: CPU count - 1)')
    ap.add_argument('--warmup', type=int, default=10, help='jumlah warmup inference')
    ap.add_argument('--runs', type=int, default=100, help='jumlah inference untuk timing')
    ap.add_argument('--images', default=None, help='folder gambar test (opsional)')
    a = ap.parse_args()

    print(f"[INFO] Platform: {platform.platform()}")
    print(f"[INFO] CPU: {platform.processor() or 'unknown'}")
    print(f"[INFO] onnxruntime {ort.__version__}")
    print(f"[INFO] Warmup: {a.warmup} | Runs: {a.runs}")

    images = load_test_images(a.images)
    if images:
        print(f"[INFO] {len(images)} gambar test dimuat dari {a.images}")
    print()

    results = []
    for m in a.models:
        if not Path(m).exists():
            print(f"[SKIP] {m} tidak ditemukan")
            continue
        print(f"Benchmarking: {Path(m).name} ...")
        r = bench_one(m, a.imgsz, a.threads, a.warmup, a.runs, images)
        results.append(r)
        print(f"  Inference (dummy): {r['dummy_ms_mean']:.1f} ms mean | "
              f"{r['dummy_ms_median']:.1f} ms median | "
              f"{r['dummy_fps']:.1f} FPS")
        if 'real_ms_mean' in r:
            print(f"  Inference (real):  {r['real_ms_mean']:.1f} ms mean | "
                  f"{r['real_fps']:.1f} FPS | avg {r['avg_dets']:.1f} dets")
        if r['temp_c']:
            print(f"  Suhu CPU: {r['temp_c']:.1f}°C")
        print()

    if len(results) >= 2:
        base = results[0]
        print("=" * 70)
        print(f"{'Model':<25} {'Mean ms':>10} {'Median ms':>10} {'FPS':>8} {'Speedup':>8}")
        print("-" * 70)
        for r in results:
            speedup = base['dummy_ms_mean'] / r['dummy_ms_mean']
            print(f"{r['model']:<25} {r['dummy_ms_mean']:>10.1f} {r['dummy_ms_median']:>10.1f} "
                  f"{r['dummy_fps']:>8.1f} {speedup:>7.2f}x")
        print("=" * 70)
        print(f"(Speedup relatif terhadap {base['model']})")


if __name__ == '__main__':
    main()

