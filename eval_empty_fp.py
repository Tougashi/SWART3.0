"""
SWART - Ukur FALSE POSITIVE pada gambar bak kosong untuk model ONNX apa pun
===========================================================================
Dipakai untuk membandingkan model LAMA vs BARU pada set bak kosong yang SAMA (sesi yang tidak dipakai training).
Satu kali inference per gambar pada confidence terendah, lalu dihitung ulang untuk beberapa ambang.

Pemakaian:
    python eval_empty_fp.py --weights weights/best2_main.onnx --images dataset_empty_test/ --imgsz 320
    python eval_empty_fp.py --weights runs/yolo11n_320/weights/best.onnx --images dataset_empty_test/
    python eval_empty_fp.py --weights new.onnx --images dataset_empty_test/ --calib calib/empty_ref.npz
        (--calib = terapkan juga filter bak kosong dari calibrate.py, untuk melihat sisa false positive di sistem utuh)
Model dengan >1 output (cabang auxiliary YOLOv9) pangkas dulu dengan prune_onnx.py.
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from swart_core import EmptyReference, OnnxDetector

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SWEEP = (0.15, 0.25, 0.35, 0.45, 0.55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--images', required=True, help='folder berisi gambar bak kosong (rekursif)')
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--calib', default=None)
    ap.add_argument('--min-change', type=float, default=0.12)
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.images).rglob('*') if p.suffix.lower() in EXT)
    if not paths:
        raise SystemExit('[ERROR] Tidak ada gambar ditemukan')
    det = OnnxDetector(a.weights, a.imgsz, SWEEP[0], 0.45, a.threads, verbose=False)
    ref = EmptyReference.load(a.calib) if a.calib else None
    print(f'[INFO] {len(paths)} gambar | model {Path(a.weights).name} ({det.w}x{det.h}) | '
          f'filter kalibrasi: {"ya" if ref else "tidak"}')

    per_img = []                                         # list of [(conf, nama)]
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        hits = det.detect(img, conf=SWEEP[0])
        if ref is not None:
            _, fg = ref.foreground(img)
            hits = [d for d in hits if ref.center_in_roi(d[:4], img.shape)
                    and ref.box_change(fg, d[:4], img.shape) >= a.min_change]
        per_img.append([(d[4], det.names.get(d[5], str(d[5]))) for d in hits])
        if (i + 1) % 500 == 0:
            print(f'  ... {i + 1}/{len(paths)}')

    n = len(per_img)
    print(f'\n{"conf":>6} | {"frame dgn false positive":>26} | kelas terbanyak')
    for thr in SWEEP:
        bad = [[x for x in hits if x[0] >= thr] for hits in per_img]
        k = sum(1 for b in bad if b)
        top = Counter(name for b in bad for _, name in b).most_common(3)
        print(f'{thr:6.2f} | {k:6d}/{n} = {100 * k / n:6.2f}%        | {top}')
    confs = [c for hits in per_img for c, _ in hits]
    if confs:
        print(f'\nconfidence false positive: median {np.median(confs):.2f}, p90 {np.percentile(confs, 90):.2f}, maks {max(confs):.2f}')
        print('Jika p90 mendekati confidence kertas asli (~0.4-0.6), menaikkan --conf TIDAK menyelesaikan masalah; perlu data/hardware.')


if __name__ == '__main__':
    main()
