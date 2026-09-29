"""
Buang cabang AUXILIARY dari ONNX YOLOv9 (DualDDetect) -> hanya sisakan cabang utama.

Kenapa perlu:
  * Model dilatih dengan train_dual.py / yolov9-s.yaml -> head punya 2 cabang.
    Export ONNX menghasilkan 2 output: `output0` (auxiliary) dan `1598` (utama).
  * Cabang utama = yang dievaluasi val_dual.py (preds[1]) dan dipakai detect.py (pred[0][1]).
    Skrip lama memakai outputs[0] = cabang auxiliary (kurang akurat) DAN tetap menghitung keduanya.
  * Meminta hanya output utama lewat sess.run([...]) TIDAK menghemat komputasi (sudah diuji),
    jadi graph harus benar-benar dipotong.

Pemakaian:
    python prune_onnx.py weights/best2.onnx                # -> weights/best2_main.onnx
    python prune_onnx.py weights/best.onnx weights/best2.onnx --bench
    python prune_onnx.py model.onnx --out model_lite.onnx

Butuh: pip install onnx onnxruntime numpy
"""
import argparse
import os
import time

import numpy as np
import onnx
import onnx.utils
import onnxruntime as ort


def session(path, threads=1):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


def bench(sess, shape, n=30):
    x = np.random.rand(*shape).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def prune(src, dst, do_bench, threads):
    model = onnx.load(src)
    ins = model.graph.input[0]
    in_name = ins.name
    outs = [o.name for o in model.graph.output]
    dims = [d.dim_value or d.dim_param for d in ins.type.tensor_type.shape.dim]
    print(f"\n== {src}\n   input {in_name} {dims} | outputs {outs}")

    if len(outs) == 1:
        print("   Sudah 1 output -> tidak perlu dipangkas.")
        return
    keep = outs[-1]  # output terakhir = cabang utama (DualDDetect mengembalikan [aux, main])
    onnx.utils.extract_model(src, dst, [in_name], [keep])
    print(f"   -> {dst} | output dipertahankan: {keep} | "
          f"{os.path.getsize(src) / 1e6:.1f} MB -> {os.path.getsize(dst) / 1e6:.1f} MB")

    # Verifikasi: output cabang utama harus identik dengan model asli
    shape = [d if isinstance(d, int) else (1 if i == 0 else 320) for i, d in enumerate(dims)]
    x = np.random.rand(*shape).astype(np.float32)
    a = session(src, threads).run([keep], {in_name: x})[0]
    b = session(dst, threads).run(None, {in_name: x})[0]
    diff = float(np.abs(a - b).max())
    print(f"   verifikasi: max|selisih| = {diff:.2e} ({'OK' if diff < 1e-4 else 'PERIKSA!'})")

    if do_bench:
        t_full = bench(session(src, threads), shape)
        t_main = bench(session(dst, threads), shape)
        print(f"   benchmark ({threads} thread): asli {t_full:.1f} ms -> pangkas {t_main:.1f} ms "
              f"(hemat {100 * (1 - t_main / t_full):.0f}%)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--out', help='nama output (hanya jika 1 model)')
    ap.add_argument('--bench', action='store_true', help='bandingkan kecepatan sebelum/sesudah')
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    for m in a.models:
        dst = a.out if (a.out and len(a.models) == 1) else os.path.splitext(m)[0] + '_main.onnx'
        prune(m, dst, a.bench, a.threads)
