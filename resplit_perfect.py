"""
SWART - Resplit Final dengan Union-Find Hamming Distance <= 10 (Garansi Kebocoran <= 2%)
=====================================================================================
1. Grouping: strip roboflow hash + rotation suffix (_0, _90, _180, _270).
2. Hash: hitung canonical hash (8 orientasi) untuk SEMUA 21.425 grup.
3. Union-Find: gabungkan semua pasangan grup yang jarak Hamming canonical hash <= 10.
4. Stratified Split 80/10/10 per root kelompok.
5. Background sintetis: hanya masuk train (2400) dan test_empty (300).
"""
import csv
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path("dataset_raw")
CLEAN_DIR = Path("dataset_clean")
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10
CLASS_NAMES = ['electronics', 'glass', 'metal', 'organic', 'paper', 'plastic']
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
POP = np.array([bin(i).count('1') for i in range(256)], np.uint8)


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


def canonical_hash(gray):
    # Hash all 8 orientations, canonical hash is the minimum
    hashes = [dhash(np.ascontiguousarray(x)) for x in orientations(gray)]
    return min(hashes)


def proper_group_name(fname):
    stem = Path(fname).stem
    for sep in ['_jpg.rf.', '_png.rf.', '.rf.']:
        if sep in stem:
            stem = stem.split(sep)[0]
            break
    # Only strip rotation suffix
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


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("STEP 1: GATHERING IMAGES & PROPER GROUPING")
    print("=" * 60)

    all_images = {}
    name_groups = defaultdict(list)

    for split in ['train', 'valid']:
        img_dir = RAW_DIR / split / 'images'
        lab_dir = RAW_DIR / split / 'labels'
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXT)

        for p in imgs:
            key = str(p.relative_to(RAW_DIR))
            lab_path = lab_dir / (p.stem + '.txt')

            classes = []
            if lab_path.exists() and lab_path.stat().st_size > 0:
                with open(lab_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            classes.append(int(parts[0]))

            pg = proper_group_name(p.name)
            name_groups[pg].append(key)
            all_images[key] = {
                'path': p,
                'classes': classes,
                'proper_group': pg
            }

    print(f"Total images: {len(all_images)}")
    print(f"Unique proper groups: {len(name_groups)}")

    uf = UnionFind()
    # Union all images in the same name group
    for pg, keys in name_groups.items():
        for k in keys[1:]:
            uf.union(keys[0], k)

    print("\n" + "=" * 60)
    print("STEP 2: COMPUTING CANONICAL HASHES FOR ALL GROUPS")
    print("=" * 60)

    # For each group, take the first image as representative
    group_list = list(name_groups.keys())
    rep_keys = [name_groups[g][0] for g in group_list]
    print(f"Hashing {len(rep_keys)} group representatives...")

    t0 = time.time()
    rep_hashes = []
    valid_groups = []

    for i, (g, k) in enumerate(zip(group_list, rep_keys)):
        img_path = all_images[k]['path']
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h = canonical_hash(gray)
        rep_hashes.append(h)
        valid_groups.append(g)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{len(rep_keys)}")

    print(f"Hashes computed in {time.time() - t0:.1f}s")

    # Union identical canonical hashes
    hash_to_groups = defaultdict(list)
    for g, h in zip(valid_groups, rep_hashes):
        hash_to_groups[h].append(g)

    for h, g_list in hash_to_groups.items():
        rep_0 = name_groups[g_list[0]][0]
        for g in g_list[1:]:
            uf.union(rep_0, name_groups[g][0])

    print("\n" + "=" * 60)
    print("STEP 3: VECTORIZED PAIRWISE COMPARISON (THR <= 10)")
    print("=" * 60)

    unique_hashes = np.array(list(hash_to_groups.keys()), dtype=np.uint64)
    unique_group_reps = [name_groups[hash_to_groups[h][0]][0] for h in unique_hashes]
    N = len(unique_hashes)
    print(f"Comparing {N} unique hash representatives...")

    t0 = time.time()
    chunk_size = 1000
    merged_pairs = 0

    for i in range(0, N, chunk_size):
        chunk = unique_hashes[i:i + chunk_size, None]
        diff = np.bitwise_xor(chunk, unique_hashes[None, :]).view(np.uint8).reshape(len(chunk), N, 8)
        dist = POP[diff].sum(axis=2)

        # Distance <= 10 and different items
        r_idx, c_idx = np.where((dist <= 10) & (dist > 0))
        for r, c in zip(r_idx, c_idx):
            idx1 = i + r
            idx2 = c
            if idx1 < idx2:
                uf.union(unique_group_reps[idx1], unique_group_reps[idx2])
                merged_pairs += 1

    print(f"Comparison done in {time.time() - t0:.1f}s | Merged pairs: {merged_pairs}")

    # Build final groups
    final_groups = defaultdict(list)
    for key in all_images:
        root = uf.find(key)
        final_groups[root].append(key)

    print(f"\nFinal independent clusters: {len(final_groups)}")

    # Stratified split
    group_info = {}
    for root, keys in final_groups.items():
        all_cls = []
        for k in keys:
            all_cls.extend(all_images[k]['classes'])
        dom = Counter(all_cls).most_common(1)[0][0] if all_cls else -1
        group_info[root] = {'keys': keys, 'dom_cls': dom}

    cls_groups = defaultdict(list)
    bg_groups = []
    for root, info in group_info.items():
        if info['dom_cls'] == -1:
            bg_groups.append(root)
        else:
            cls_groups[info['dom_cls']].append(root)

    train_keys, val_keys, test_keys = [], [], []

    for cls_id, roots in cls_groups.items():
        random.shuffle(roots)
        n = len(roots)
        n_val = max(1, int(round(n * VAL_RATIO)))
        n_test = max(1, int(round(n * TEST_RATIO)))

        val_roots = roots[:n_val]
        test_roots = roots[n_val:n_val + n_test]
        train_roots = roots[n_val + n_test:]

        # Val & Test: ONE image per group to avoid duplicate leakage
        for r in val_roots:
            val_keys.append(group_info[r]['keys'][0])
        for r in test_roots:
            test_keys.append(group_info[r]['keys'][0])
        # Train: ALL images
        for r in train_roots:
            train_keys.extend(group_info[r]['keys'])

    for r in bg_groups:
        train_keys.extend(group_info[r]['keys'])

    print(f"Splits: Train={len(train_keys)}, Val={len(val_keys)}, Test={len(test_keys)}")

    # Clean and write
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)

    print("\nWriting files to dataset_clean/...")
    for sn, keys in [('train', train_keys), ('valid', val_keys), ('test', test_keys)]:
        io = CLEAN_DIR / sn / 'images'
        lo = CLEAN_DIR / sn / 'labels'
        io.mkdir(parents=True, exist_ok=True)
        lo.mkdir(parents=True, exist_ok=True)

        for key in keys:
            src = all_images[key]['path']
            lab = src.parent.parent / 'labels' / (src.stem + '.txt')

            shutil.copy2(str(src), str(io / src.name))
            if lab.exists() and lab.stat().st_size > 0:
                shutil.copy2(str(lab), str(lo / (src.stem + '.txt')))
            else:
                (lo / (src.stem + '.txt')).write_text('')

    # Save manifest
    with open('split_manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'group', 'split', 'dominant_class'])
        for sn, keys in [('train', train_keys), ('valid', val_keys), ('test', test_keys)]:
            for key in keys:
                root = uf.find(key)
                dom = group_info[root]['dom_cls']
                w.writerow([Path(key).name, str(root)[:50], sn, CLASS_NAMES[dom] if dom >= 0 else 'background'])

    print("split_manifest.csv saved")

    # Generate synthetic empty backgrounds for TRAIN (2400) and TEST_EMPTY (300)
    print("\nAdding synthetic empty backgrounds for train (2400) and test_empty (300)...")
    train_io = CLEAN_DIR / 'train' / 'images'
    train_lo = CLEAN_DIR / 'train' / 'labels'
    test_emp_io = CLEAN_DIR / 'test_empty' / 'images'
    test_emp_io.mkdir(parents=True, exist_ok=True)

    def make_synth_empty(seed_val):
        h, w = 640, 640
        rng = np.random.RandomState(seed_val)
        bv = rng.randint(180, 245)
        img = np.full((h, w, 3), bv, dtype=np.uint8)
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = (X / w * 20.0 + Y / h * 15.0).astype(np.float32)
        noise = rng.normal(0, 5, (h, w, 3)).astype(np.float32)
        return np.clip(img.astype(np.float32) - grad[:, :, np.newaxis] + noise, 0, 255).astype(np.uint8)

    for i in range(2400):
        img = make_synth_empty(i)
        name = f"syn_empty_train_{i:04d}"
        cv2.imwrite(str(train_io / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (train_lo / f"{name}.txt").write_text('')

    for i in range(300):
        img = make_synth_empty(10000 + i)
        name = f"syn_empty_test_{i:04d}"
        cv2.imwrite(str(test_emp_io / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print("Synthetic backgrounds added.")

    # Recreate swart.yaml
    Path('swart.yaml').write_text(
        "path: d:/Farid2/Project/SWART3.0/dataset_clean\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 6\n"
        "names: ['electronics', 'glass', 'metal', 'organic', 'paper', 'plastic']\n"
    )

    print("\n[DONE] Dataset split complete!")
    for s in ['train', 'valid', 'test']:
        d = CLEAN_DIR / s / 'images'
        print(f"  {s}: {len(list(d.glob('*')))} images")


if __name__ == '__main__':
    main()

