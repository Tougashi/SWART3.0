"""
SWART - Zero Leakage Dataset Resplitter (Multiprocessing + RTX 2050 GPU)
========================================================================
Guarantees train <-> val/test leakage <= 2% (targeting 0.0%) under check_leakage.py (thr=10).
1. Proper grouping by Roboflow hash + rotation suffix.
2. 8-orientation dHash computation using 8 multiprocessing workers (~2.2 mins).
3. RTX 2050 GPU-accelerated all-orientation pairwise distance (~12 secs):
   min_{k in 0..7} dist(H_i[0], H_j[k]) <= 10 -> union groups.
4. Stratified 80/10/10 split by independent cluster.
5. Synthetic empty bin backgrounds added to train (2400) and test_empty (300).
"""
import csv
import multiprocessing
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

RAW_DIR = Path("dataset_raw")
CLEAN_DIR = Path("dataset_clean")
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10
CLASS_NAMES = ['electronics', 'glass', 'metal', 'organic', 'paper', 'plastic']
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def dhash(gray, size=9):
    small = cv2.resize(gray, (size, size - 1), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).view('>u8')[0]


def orientations(gray):
    outs = []
    for k in range(4):
        r = np.rot90(gray, k)
        outs += [r, r[:, ::-1]]
    return outs


def proper_group_name(fname):
    stem = Path(fname).stem
    for sep in ['_jpg.rf.', '_png.rf.', '.rf.']:
        if sep in stem:
            stem = stem.split(sep)[0]
            break
    stem = re.sub(r'_(0|90|180|270)$', '', stem)
    return stem


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def hash_single_group(item):
    grp, rep_path = item
    gray = cv2.imread(str(rep_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return grp, None
    gray = np.ascontiguousarray(gray)
    h8 = [int(dhash(np.ascontiguousarray(x))) for x in orientations(gray)]
    return grp, h8


def popcount64_gpu(x):
    """Vectorized bitcount for int64 tensor on GPU."""
    m1 = 0x5555555555555555
    m2 = 0x3333333333333333
    m4 = 0x0f0f0f0f0f0f0f0f
    x = x - ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return x & 0x7f


def generate_synthetic_backgrounds():
    bgs = []
    rng = np.random.RandomState(SEED)
    sz = 640
    for i in range(2700):
        base_val = rng.randint(220, 255)
        img = np.full((sz, sz, 3), base_val, dtype=np.float32)

        if rng.rand() > 0.3:
            gx = np.linspace(rng.uniform(-20, 0), rng.uniform(0, 20), sz).reshape(1, sz, 1)
            gy = np.linspace(rng.uniform(-20, 0), rng.uniform(0, 20), sz).reshape(sz, 1, 1)
            img += gx + gy

        if rng.rand() > 0.3:
            y, x = np.ogrid[:sz, :sz]
            cx, cy = sz // 2 + rng.randint(-30, 30), sz // 2 + rng.randint(-30, 30)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            max_d = (sz // 2) ** 2
            shadow = rng.uniform(20, 50) * (dist_sq / max_d)
            shadow = np.clip(shadow, 0, 60)[:, :, None]
            img -= shadow

        if rng.rand() > 0.4:
            c_sz = rng.randint(80, 200)
            corn = np.linspace(rng.uniform(15, 45), 0, c_sz).reshape(c_sz, 1, 1)
            if rng.rand() > 0.5:
                img[:c_sz, :c_sz] -= (corn * corn.transpose(1, 0, 2)) / 30
            else:
                img[-c_sz:, -c_sz:] -= (corn * corn.transpose(1, 0, 2)) / 30

        noise = rng.normal(0, rng.uniform(1.5, 4.0), (sz, sz, 3))
        img += noise
        tint = rng.uniform(-4, 4, (1, 1, 3))
        img += tint

        img = np.clip(img, 0, 255).astype(np.uint8)
        bgs.append(img)
    return bgs


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 65, flush=True)
    print("STEP 1: GATHERING RAW DATASET IMAGES & EXTRACTING LABELS", flush=True)
    print("=" * 65, flush=True)

    all_images = {}
    name_groups = defaultdict(list)

    for split in ['train', 'valid']:
        img_dir = RAW_DIR / split / 'images'
        lab_dir = RAW_DIR / split / 'labels'
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXT)

        # Pre-index labels for speed
        lab_map = {p.stem: p for p in lab_dir.iterdir() if p.suffix == '.txt'}

        for p in imgs:
            key = str(p.relative_to(RAW_DIR))
            lab_path = lab_map.get(p.stem)

            classes = []
            if lab_path and lab_path.stat().st_size > 0:
                with open(lab_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            classes.append(int(parts[0]))

            pg = proper_group_name(p.name)
            name_groups[pg].append(key)
            all_images[key] = {
                'path': p,
                'lab_path': lab_path,
                'classes': classes,
                'proper_group': pg
            }

    print(f"Total raw images: {len(all_images)}", flush=True)
    print(f"Unique proper groups (base items): {len(name_groups)}", flush=True)

    uf = UnionFind()
    for pg, keys in name_groups.items():
        for k in keys[1:]:
            uf.union(keys[0], k)

    print("\n" + "=" * 65, flush=True)
    print("STEP 2: COMPUTING 8-ORIENTATION dHASH (MULTIPROCESSING 8 WORKERS)", flush=True)
    print("=" * 65, flush=True)

    group_list = list(name_groups.keys())
    items_to_hash = [(g, str(all_images[name_groups[g][0]]['path'])) for g in group_list]

    t0 = time.time()
    h_all_dict = {}
    with multiprocessing.Pool(processes=8) as pool:
        for i, (grp, h8) in enumerate(pool.imap(hash_single_group, items_to_hash, chunksize=50)):
            if h8 is not None:
                h_all_dict[grp] = h8
            if (i + 1) % 5000 == 0:
                print(f"  ... hashed {i + 1}/{len(items_to_hash)} groups ({time.time() - t0:.1f}s)", flush=True)

    valid_groups = [g for g in group_list if g in h_all_dict]
    N = len(valid_groups)
    print(f"Computed hashes for {N} groups in {time.time() - t0:.1f}s", flush=True)

    print("\n" + "=" * 65, flush=True)
    print(f"STEP 3: GPU-ACCELERATED ALL-ORIENTATION PAIRWISE DISTANCE ({DEVICE.upper()})", flush=True)
    print("=" * 65, flush=True)

    raw_h0 = np.array([h_all_dict[g][0] for g in valid_groups], dtype=np.int64)
    raw_hall = np.array([h_all_dict[g] for g in valid_groups], dtype=np.int64)
    group_reps = [name_groups[g][0] for g in valid_groups]

    t_h0 = torch.tensor(raw_h0, dtype=torch.int64, device=DEVICE)
    t_hall = torch.tensor(raw_hall, dtype=torch.int64, device=DEVICE)

    t0 = time.time()
    C = 250
    merged_count = 0

    with torch.no_grad():
        for i in range(0, N, C):
            c = t_h0[i:i + C, None, None]  # (C, 1, 1)
            xor = c ^ t_hall[None, :, :]   # (C, N, 8)
            d = popcount64_gpu(xor).min(dim=-1).values  # (C, N)

            matches = torch.nonzero((d <= 10) & (d >= 0)).cpu().numpy()
            for r, c_idx in matches:
                idx1 = i + int(r)
                idx2 = int(c_idx)
                if idx1 < idx2:
                    uf.union(group_reps[idx1], group_reps[idx2])
                    merged_count += 1

            if (i // C + 1) % 10 == 0 or (i + C) >= N:
                progress = min(100.0, 100.0 * (i + C) / N)
                print(f"  ... {progress:5.1f}% | Merged pairs: {merged_count:6d} ({time.time() - t0:.1f}s)", flush=True)

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    print(f"GPU distance comparison done in {time.time() - t0:.1f}s | Total merged pairs: {merged_count}", flush=True)

    clusters = defaultdict(list)
    for key in all_images:
        root = uf.find(key)
        clusters[root].append(key)

    print(f"\nTotal independent clusters: {len(clusters)}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("STEP 4: STRATIFIED SPLIT 80/10/10 PER CLUSTER", flush=True)
    print("=" * 65, flush=True)

    cluster_info = {}
    for root, keys in clusters.items():
        all_cls = []
        for k in keys:
            all_cls.extend(all_images[k]['classes'])
        dom = Counter(all_cls).most_common(1)[0][0] if all_cls else -1
        cluster_info[root] = {'keys': keys, 'dom_cls': dom}

    cls_clusters = defaultdict(list)
    bg_clusters = []
    for root, info in cluster_info.items():
        if info['dom_cls'] == -1:
            bg_clusters.append(root)
        else:
            cls_clusters[info['dom_cls']].append(root)

    train_keys, val_keys, test_keys = [], [], []

    for cls_id in sorted(cls_clusters.keys()):
        roots = cls_clusters[cls_id]
        random.shuffle(roots)
        n = len(roots)
        n_val = max(1, int(round(n * VAL_RATIO)))
        n_test = max(1, int(round(n * TEST_RATIO)))

        val_roots = roots[:n_val]
        test_roots = roots[n_val:n_val + n_test]
        train_roots = roots[n_val + n_test:]

        for r in val_roots:
            val_keys.extend(cluster_info[r]['keys'])
        for r in test_roots:
            test_keys.extend(cluster_info[r]['keys'])
        for r in train_roots:
            train_keys.extend(cluster_info[r]['keys'])

        cname = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
        print(f"  Class {cls_id:2d} ({cname:11s}): {len(roots):4d} clusters -> "
              f"Train={len(train_roots):4d}, Val={len(val_roots):4d}, Test={len(test_roots):4d}", flush=True)

    for r in bg_clusters:
        train_keys.extend(cluster_info[r]['keys'])

    print(f"\nFinal assignments: Train={len(train_keys)}, Val={len(val_keys)}, Test={len(test_keys)}", flush=True)

    train_roots_set = set(uf.find(k) for k in train_keys)
    val_roots_set = set(uf.find(k) for k in val_keys)
    test_roots_set = set(uf.find(k) for k in test_keys)
    print(f"Root overlap Train <-> Val: {len(train_roots_set.intersection(val_roots_set))}", flush=True)
    print(f"Root overlap Train <-> Test: {len(train_roots_set.intersection(test_roots_set))}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("STEP 5: SYNTHESIZING EMPTY BIN BACKGROUNDS & WRITING DATASET", flush=True)
    print("=" * 65, flush=True)

    if CLEAN_DIR.exists():
        print(f"Removing old {CLEAN_DIR}...", flush=True)
        shutil.rmtree(CLEAN_DIR)

    for sn in ['train', 'valid', 'test']:
        (CLEAN_DIR / sn / 'images').mkdir(parents=True, exist_ok=True)
        (CLEAN_DIR / sn / 'labels').mkdir(parents=True, exist_ok=True)
    (CLEAN_DIR / 'test_empty' / 'images').mkdir(parents=True, exist_ok=True)
    (CLEAN_DIR / 'test_empty' / 'labels').mkdir(parents=True, exist_ok=True)

    def copy_split(split_name, keys):
        img_dst = CLEAN_DIR / split_name / 'images'
        lab_dst = CLEAN_DIR / split_name / 'labels'
        for k in keys:
            info = all_images[k]
            src_img = info['path']
            shutil.copy2(src_img, img_dst / src_img.name)
            if info['lab_path'] and info['lab_path'].exists():
                shutil.copy2(info['lab_path'], lab_dst / (src_img.stem + '.txt'))
            else:
                (lab_dst / (src_img.stem + '.txt')).touch()

    print(f"Writing train ({len(train_keys)} images)...", flush=True)
    copy_split('train', train_keys)

    print(f"Writing valid ({len(val_keys)} images)...", flush=True)
    copy_split('valid', val_keys)

    print(f"Writing test ({len(test_keys)} images)...", flush=True)
    copy_split('test', test_keys)

    print("\nGenerating 2700 synthetic empty bin images...", flush=True)
    bgs = generate_synthetic_backgrounds()
    train_img_dst = CLEAN_DIR / 'train' / 'images'
    train_lab_dst = CLEAN_DIR / 'train' / 'labels'
    for idx, bg in enumerate(bgs[:2400]):
        fn = f"synthetic_empty_bin_{idx:04d}.jpg"
        cv2.imwrite(str(train_img_dst / fn), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (train_lab_dst / f"synthetic_empty_bin_{idx:04d}.txt").touch()

    empty_img_dst = CLEAN_DIR / 'test_empty' / 'images'
    empty_lab_dst = CLEAN_DIR / 'test_empty' / 'labels'
    for idx, bg in enumerate(bgs[2400:]):
        fn = f"synthetic_empty_test_{idx:04d}.jpg"
        cv2.imwrite(str(empty_img_dst / fn), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (empty_lab_dst / f"synthetic_empty_test_{idx:04d}.txt").touch()

    print("Background synthesis complete.", flush=True)

    manifest_path = Path("split_manifest.csv")
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'group_root', 'split'])
        for k in train_keys:
            writer.writerow([Path(k).name, uf.find(k), 'train'])
        for k in val_keys:
            writer.writerow([Path(k).name, uf.find(k), 'valid'])
        for k in test_keys:
            writer.writerow([Path(k).name, uf.find(k), 'test'])

    print(f"\nFinal dataset counts in {CLEAN_DIR}:", flush=True)
    for s in ['train', 'valid', 'test', 'test_empty']:
        n_img = len(list((CLEAN_DIR / s / 'images').glob('*.*')))
        n_lab = len(list((CLEAN_DIR / s / 'labels').glob('*.txt')))
        print(f"  {s:12s}: {n_img:5d} images, {n_lab:5d} labels", flush=True)


if __name__ == '__main__':
    main()
