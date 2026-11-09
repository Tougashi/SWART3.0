"""
SWART - Final Zero-Leak Resplit (Sequential hash + GPU distance)
================================================================
Guarantees leakage <= 2% under check_leakage.py --thr 10.
- Sequential image hashing (no multiprocessing to avoid Windows freeze)
- GPU pairwise Hamming distance (RTX 2050)
- Stratified 80/10/10 split by Union-Find cluster
- 2400 synthetic empty backgrounds in train, 300 in test_empty
"""
import csv
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


def popcount64_gpu(x):
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


def generate_synthetic_backgrounds(count=2700):
    bgs = []
    rng = np.random.RandomState(SEED)
    sz = 640
    for _ in range(count):
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
            img -= np.clip(shadow, 0, 60)[:, :, None]
        if rng.rand() > 0.4:
            c_sz = rng.randint(80, 200)
            corn = np.linspace(rng.uniform(15, 45), 0, c_sz).reshape(c_sz, 1, 1)
            if rng.rand() > 0.5:
                img[:c_sz, :c_sz] -= (corn * corn.transpose(1, 0, 2)) / 30
            else:
                img[-c_sz:, -c_sz:] -= (corn * corn.transpose(1, 0, 2)) / 30
        img += rng.normal(0, rng.uniform(1.5, 4.0), (sz, sz, 3))
        img += rng.uniform(-4, 4, (1, 1, 3))
        bgs.append(np.clip(img, 0, 255).astype(np.uint8))
    return bgs


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    t_total = time.time()

    # ===== STEP 1: Gather images =====
    print("=" * 60)
    print("STEP 1: Gathering images and labels")
    print("=" * 60)

    all_images = {}
    name_groups = defaultdict(list)

    for split in ['train', 'valid']:
        img_dir = RAW_DIR / split / 'images'
        lab_dir = RAW_DIR / split / 'labels'
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXT)
        lab_set = {p.stem for p in lab_dir.iterdir() if p.suffix == '.txt'}

        for p in imgs:
            key = str(p.relative_to(RAW_DIR))
            lab_path = lab_dir / (p.stem + '.txt') if p.stem in lab_set else None

            classes = []
            if lab_path and lab_path.exists():
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
            }

    print(f"  Total images: {len(all_images)}")
    print(f"  Proper groups: {len(name_groups)}")

    # ===== STEP 2: Union-Find on name groups =====
    uf = UnionFind()
    for pg, keys in name_groups.items():
        for k in keys[1:]:
            uf.union(keys[0], k)

    # ===== STEP 3: Hash one representative per group (sequential) =====
    print("\n" + "=" * 60)
    print("STEP 2: Hashing group representatives")
    print("=" * 60)

    group_list = list(name_groups.keys())
    cache_path = Path("group_hashes_cache.npz")

    if cache_path.exists():
        print(f"  Loading cached hashes from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        valid_groups = list(data['groups'])
        raw_hall = data['hall'].view(np.int64)
        raw_h0 = raw_hall[:, 0]
        h_all_dict = {g: raw_hall[i] for i, g in enumerate(valid_groups)}
        print(f"  Loaded {len(valid_groups)} group hashes from cache!")
    else:
        h_all_dict = {}
        t0 = time.time()

        for i, g in enumerate(group_list):
            rep_key = name_groups[g][0]
            img_path = all_images[rep_key]['path']
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray = np.ascontiguousarray(gray)
            h8 = [int(dhash(np.ascontiguousarray(x))) for x in orientations(gray)]
            h_all_dict[g] = h8

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(group_list) - i - 1) / rate
                print(f"  {i+1}/{len(group_list)} ({rate:.0f} img/s, ETA {eta:.0f}s)")

        valid_groups = [g for g in group_list if g in h_all_dict]
        raw_hall_u64 = np.array([h_all_dict[g] for g in valid_groups], dtype=np.uint64)
        np.savez_compressed(cache_path, groups=np.array(valid_groups), hall=raw_hall_u64)
        print(f"  Saved hash cache to {cache_path}")
        raw_hall = raw_hall_u64.view(np.int64)
        raw_h0 = raw_hall[:, 0]
        print(f"  Hashed {len(valid_groups)} groups in {time.time() - t0:.1f}s")

    N = len(valid_groups)

    # ===== STEP 4: GPU pairwise distance =====
    print("\n" + "=" * 60)
    print(f"STEP 3: GPU pairwise Hamming distance ({DEVICE})")
    print("=" * 60)

    group_reps = [name_groups[g][0] for g in valid_groups]

    t_h0 = torch.tensor(raw_h0, dtype=torch.int64, device=DEVICE)
    t_hall = torch.tensor(raw_hall, dtype=torch.int64, device=DEVICE)

    t0 = time.time()
    C = 250
    merged_count = 0

    with torch.no_grad():
        for i in range(0, N, C):
            c = t_h0[i:i + C, None, None]
            xor = c ^ t_hall[None, :, :]
            d = popcount64_gpu(xor).min(dim=-1).values

            matches = torch.nonzero((d <= 10) & (d >= 0)).cpu().numpy()
            for r, c_idx in matches:
                idx1 = i + int(r)
                idx2 = int(c_idx)
                if idx1 < idx2:
                    uf.union(group_reps[idx1], group_reps[idx2])
                    merged_count += 1

            if (i // C + 1) % 20 == 0 or (i + C) >= N:
                progress = min(100.0, 100.0 * (i + C) / N)
                print(f"  {progress:5.1f}% | Merged: {merged_count} ({time.time() - t0:.1f}s)")

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    print(f"  Done in {time.time() - t0:.1f}s | Merged pairs: {merged_count}")

    # Free GPU memory for training
    del t_h0, t_hall
    torch.cuda.empty_cache()

    # ===== STEP 5: Build clusters and split =====
    print("\n" + "=" * 60)
    print("STEP 4: Stratified split 80/10/10")
    print("=" * 60)

    clusters = defaultdict(list)
    for key in all_images:
        clusters[uf.find(key)].append(key)
    print(f"  Independent clusters: {len(clusters)}")

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

        cname = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f'cls{cls_id}'
        print(f"  {cname:11s}: {len(roots):4d} clusters -> T={len(train_roots)} V={len(val_roots)} Te={len(test_roots)}")

    for r in bg_clusters:
        train_keys.extend(cluster_info[r]['keys'])

    # Verify no cluster overlap
    tr_roots = set(uf.find(k) for k in train_keys)
    va_roots = set(uf.find(k) for k in val_keys)
    te_roots = set(uf.find(k) for k in test_keys)
    print(f"\n  Assignments: Train={len(train_keys)} Val={len(val_keys)} Test={len(test_keys)}")
    print(f"  Cluster overlap T<->V: {len(tr_roots & va_roots)} | T<->Te: {len(tr_roots & te_roots)}")

    # ===== STEP 6: Write files =====
    print("\n" + "=" * 60)
    print("STEP 5: Writing dataset_clean/")
    print("=" * 60)

    if CLEAN_DIR.exists():
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
            src = info['path']
            shutil.copy2(src, img_dst / src.name)
            if info['lab_path'] and info['lab_path'].exists():
                shutil.copy2(info['lab_path'], lab_dst / (src.stem + '.txt'))
            else:
                (lab_dst / (src.stem + '.txt')).touch()

    t0 = time.time()
    print(f"  Copying train ({len(train_keys)})...")
    copy_split('train', train_keys)
    print(f"  Copying valid ({len(val_keys)})...")
    copy_split('valid', val_keys)
    print(f"  Copying test ({len(test_keys)})...")
    copy_split('test', test_keys)
    print(f"  File copy done in {time.time() - t0:.1f}s")

    # Synthetic backgrounds
    print("  Generating synthetic backgrounds...")
    bgs = generate_synthetic_backgrounds(2700)
    train_img = CLEAN_DIR / 'train' / 'images'
    train_lab = CLEAN_DIR / 'train' / 'labels'
    for idx, bg in enumerate(bgs[:2400]):
        fn = f"synthetic_empty_bin_{idx:04d}.jpg"
        cv2.imwrite(str(train_img / fn), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (train_lab / f"synthetic_empty_bin_{idx:04d}.txt").touch()

    empty_img = CLEAN_DIR / 'test_empty' / 'images'
    empty_lab = CLEAN_DIR / 'test_empty' / 'labels'
    for idx, bg in enumerate(bgs[2400:]):
        fn = f"synthetic_empty_test_{idx:04d}.jpg"
        cv2.imwrite(str(empty_img / fn), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (empty_lab / f"synthetic_empty_test_{idx:04d}.txt").touch()

    # Save manifest
    with open("split_manifest.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'group_root', 'split'])
        for k in train_keys:
            w.writerow([Path(k).name, uf.find(k), 'train'])
        for k in val_keys:
            w.writerow([Path(k).name, uf.find(k), 'valid'])
        for k in test_keys:
            w.writerow([Path(k).name, uf.find(k), 'test'])

    # Final report
    print("\n" + "=" * 60)
    print("FINAL DATASET COUNTS")
    print("=" * 60)
    for s in ['train', 'valid', 'test', 'test_empty']:
        n = len(list((CLEAN_DIR / s / 'images').glob('*.*')))
        print(f"  {s:12s}: {n:5d} images")

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
    print("DONE! Ready for training.")


if __name__ == '__main__':
    main()

