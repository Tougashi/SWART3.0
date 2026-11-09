"""
SWART - Cek KEBOCORAN data: apakah gambar valid/test hampir identik dengan gambar train?
========================================================================================
Memakai perceptual hash (dHash 64-bit). Gambar hasil augmentasi (flip/rotasi kecil/brightness) dan frame berurutan
dari objek yang sama sering lolos sebagai "valid" padahal bocoran dari train -> mAP terlalu optimistis.
(flip/rotasi 90 derajat juga dicek: tiap gambar valid dibandingkan dengan 8 orientasinya.)

Pemakaian:
    python check_leakage.py --train dataset/train/images --val dataset/valid/images
    python check_leakage.py --train dataset/train/images --val dataset/test/images --thr 8 --csv bocor.csv
Hasil: persentase gambar valid yang punya "kembaran" di train + contoh pasangan terdekat.
Aturan baca: > ~2% = split bermasalah, pisahkan ulang per kelompok kembaran/sesi.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
POP = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).view('>u8')[0]


def orientations(gray):
    outs = []
    for k in range(4):
        r = np.rot90(gray, k)
        outs += [r, r[:, ::-1]]
    return outs


def list_images(d):
    return sorted(p for p in Path(d).rglob('*') if p.suffix.lower() in EXT)


def hashes(paths, all_orient):
    out, ok = [], []
    for i, p in enumerate(paths):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = np.ascontiguousarray(g)
        out.append([dhash(np.ascontiguousarray(x)) for x in (orientations(g) if all_orient else [g])])
        ok.append(p)
        if (i + 1) % 2000 == 0:
            print(f'  ... {i + 1}/{len(paths)}')
    return np.array(out, dtype=np.uint64), ok


def hamming_min(q, base):
    """q: (K,) hash orientasi satu gambar; base: (N,) -> (jarak_min, indeks)."""
    best, idx = 65, -1
    for h in q:
        x = np.bitwise_xor(base, h).view(np.uint8).reshape(-1, 8)
        d = POP[x].sum(1)
        j = int(d.argmin())
        if d[j] < best:
            best, idx = int(d[j]), j
    return best, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--thr', type=int, default=6, help='jarak Hamming maks (dari 64 bit) yang dianggap kembaran')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--max-val', type=int, default=0, help='batasi jumlah gambar valid (0 = semua)')
    a = ap.parse_args()

    tr, va = list_images(a.train), list_images(a.val)
    if a.max_val:
        va = va[::max(1, len(va) // a.max_val)]
    print(f'[INFO] train {len(tr)} gambar | val {len(va)} gambar')
    th, tr = hashes(tr, False)
    vh, va = hashes(va, True)
    base = th[:, 0]

    rows, hit = [], 0
    for i, q in enumerate(vh):
        d, j = hamming_min(q, base)
        if d <= a.thr:
            hit += 1
        rows.append((str(va[i]), str(tr[j]), d))
    rows.sort(key=lambda r: r[2])
    pct = 100 * hit / max(len(va), 1)
    print(f'\n[HASIL] {hit}/{len(va)} gambar valid ({pct:.1f}%) punya kembaran di train (jarak <= {a.thr})')
    print('        ' + ('BOCOR: split perlu diulang per kelompok/sesi' if pct > 2 else 'tampak aman (<= 2%)'))
    print('\nPasangan paling mirip:')
    for v, t, d in rows[:10]:
        print(f'  d={d:2d}  {Path(v).name}  <->  {Path(t).name}')
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['val_image', 'nearest_train_image', 'hamming'])
            w.writerows(rows)
        print(f'[INFO] Detail disimpan: {a.csv}')


if __name__ == '__main__':
    main()
