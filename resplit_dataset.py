"""
SWART - Audit dataset, resplit berbasis kelompok, dan generate background sintetis.
Gabungan Langkah 3 + 4 + generate empty backgrounds.
"""
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ============================================================
RAW_DIR = Path("dataset_raw")
CLEAN_DIR = Path("dataset_clean")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10
SEED = 42
EMPTY_TRAIN = 2400
EMPTY_VAL = 300
EMPTY_TEST = 300
CLASS_NAMES = ['electronics', 'glass', 'metal', 'organic', 'paper', 'plastic']
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def dhash(gray, size=9):
    small = cv2.resize(gray, (size, size - 1), interpolation=cv2.INTER_AREA)
    return int(np.packbits((small[:, 1:] > small[:, :-1]).flatten()).tobytes().hex(), 16)

def roboflow_group(fname):
    base = Path(fname).stem
    for sep in ['_jpg.rf.', '_png.rf.', '.rf.']:
        if sep in base:
            return base.split(sep)[0]
    return base

class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        if x not in self.parent: self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.parent[ra] = rb

# ============================================================
# STEP 1: AUDIT
# ============================================================
def audit_dataset():
    print("=" * 60)
    print("STEP 1: AUDIT DATASET")
    print("=" * 60)
    all_images = {}
    instance_counts = Counter()
    no_label = []
    bad_coords = 0
    for split in ['train', 'valid']:
        img_dir = RAW_DIR / split / 'images'
        lab_dir = RAW_DIR / split / 'labels'
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXT)
        print(f"\n  {split}: {len(imgs)} images")
        for img_path in imgs:
            lab_path = lab_dir / (img_path.stem + '.txt')
            key = str(img_path.relative_to(RAW_DIR))
            if not lab_path.exists() or lab_path.stat().st_size == 0:
                no_label.append(key)
                all_images[key] = {'split': split, 'classes': [], 'path': img_path}
                continue
            classes = []
            with open(lab_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        classes.append(cls)
                        instance_counts[cls] += 1
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            bad_coords += 1
            all_images[key] = {'split': split, 'classes': classes, 'path': img_path}
    print(f"\n  Total images: {len(all_images)}")
    print(f"  Background (no label): {len(no_label)}")
    print(f"  Bad coordinates: {bad_coords}")
    print(f"\n  Instance distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {i}: {name:15s} -> {instance_counts.get(i, 0):6d}")
    groups = defaultdict(list)
    for key in all_images:
        groups[roboflow_group(Path(key).name)].append(key)
    multi = {g: imgs for g, imgs in groups.items() if len(imgs) > 1}
    print(f"\n  Roboflow groups: {len(groups)} total, {len(multi)} with >1 copy")
    if multi:
        sizes = [len(v) for v in multi.values()]
        print(f"    Avg copies: {np.mean(sizes):.1f}, Max: {max(sizes)}")
    return all_images, groups

# ============================================================
# STEP 2: GROUP-BASED RESPLIT
# ============================================================
def resplit_dataset(all_images, rf_groups):
    print("\n" + "=" * 60)
    print("STEP 2: GROUP-BASED RESPLIT")
    print("=" * 60)
    uf = UnionFind()
    # Union by Roboflow filename group
    for g, keys in rf_groups.items():
        for k in keys[1:]:
            uf.union(keys[0], k)
    # Compute dHash
    print("  Computing dHash...")
    hash_map = {}
    keys_list = list(all_images.keys())
    for i, key in enumerate(keys_list):
        img = cv2.imread(str(all_images[key]['path']), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            hash_map[key] = dhash(img)
        if (i + 1) % 5000 == 0:
            print(f"    ... {i+1}/{len(keys_list)}")
    # Union exact hash matches
    h2k = defaultdict(list)
    for key, h in hash_map.items():
        h2k[h].append(key)
    for h, keys in h2k.items():
        for k in keys[1:]:
            uf.union(keys[0], k)
    # Final groups
    final_groups = defaultdict(list)
    for key in all_images:
        final_groups[uf.find(key)].append(key)
    print(f"  Final groups: {len(final_groups)}")
    # Dominant class per group
    group_info = {}
    for root, keys in final_groups.items():
        all_cls = []
        for k in keys:
            all_cls.extend(all_images[k]['classes'])
        dom = Counter(all_cls).most_common(1)[0][0] if all_cls else -1
        group_info[root] = {'keys': keys, 'dom_cls': dom}
    # Stratified split
    random.seed(SEED)
    cls_groups = defaultdict(list)
    bg_groups = []
    for root, info in group_info.items():
        if info['dom_cls'] == -1:
            bg_groups.append(root)
        else:
            cls_groups[info['dom_cls']].append(root)
    train_keys, val_keys, test_keys = [], [], []
    for cls, roots in cls_groups.items():
        random.shuffle(roots)
        n = len(roots)
        n_val = max(1, int(n * VAL_RATIO))
        n_test = max(1, int(n * TEST_RATIO))
        for r in roots[:n_val]:
            val_keys.append(group_info[r]['keys'][0])
        for r in roots[n_val:n_val + n_test]:
            test_keys.append(group_info[r]['keys'][0])
        for r in roots[n_val + n_test:]:
            train_keys.extend(group_info[r]['keys'])
    random.shuffle(bg_groups)
    n_bg = len(bg_groups)
    n_bv = max(0, int(n_bg * VAL_RATIO))
    n_bt = max(0, int(n_bg * TEST_RATIO))
    for r in bg_groups[:n_bv]:
        val_keys.append(group_info[r]['keys'][0])
    for r in bg_groups[n_bv:n_bv + n_bt]:
        test_keys.append(group_info[r]['keys'][0])
    for r in bg_groups[n_bv + n_bt:]:
        train_keys.extend(group_info[r]['keys'])
    print(f"  Train: {len(train_keys)} | Val: {len(val_keys)} | Test: {len(test_keys)}")
    # Copy files
    print("  Copying files...")
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    for sn, keys in [('train', train_keys), ('valid', val_keys), ('test', test_keys)]:
        io = CLEAN_DIR / sn / 'images'
        lo = CLEAN_DIR / sn / 'labels'
        io.mkdir(parents=True, exist_ok=True)
        lo.mkdir(parents=True, exist_ok=True)
        for key in keys:
            src = all_images[key]['path']
            sl = src.parent.parent / 'labels' / (src.stem + '.txt')
            shutil.copy2(str(src), str(io / src.name))
            if sl.exists():
                shutil.copy2(str(sl), str(lo / (src.stem + '.txt')))
            else:
                (lo / (src.stem + '.txt')).write_text('')
    # Manifest
    with open('split_manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'group', 'split', 'dominant_class'])
        for sn, keys in [('train', train_keys), ('valid', val_keys), ('test', test_keys)]:
            for key in keys:
                root = uf.find(key)
                dom = group_info.get(root, {}).get('dom_cls', -1)
                w.writerow([Path(key).name, str(root)[:60], sn, CLASS_NAMES[dom] if dom >= 0 else 'background'])
    print("  split_manifest.csv saved")
    # Per-split class distribution
    print("\n  Class distribution per split:")
    for sn, keys in [('train', train_keys), ('valid', val_keys), ('test', test_keys)]:
        cc = Counter()
        for k in keys: 
            for c in all_images[k]['classes']: cc[c] += 1
        print(f"    {sn}: " + " | ".join(f"{CLASS_NAMES[i]}={cc.get(i,0)}" for i in range(6)))
    return train_keys, val_keys, test_keys, uf, group_info

# ============================================================
# STEP 3: GENERATE SYNTHETIC EMPTY BACKGROUNDS
# ============================================================
def generate_empty_backgrounds():
    print("\n" + "=" * 60)
    print("STEP 3: GENERATE SYNTHETIC EMPTY BACKGROUNDS")
    print("=" * 60)
    random.seed(SEED); np.random.seed(SEED)
    def make_empty(idx):
        h, w = 640, 640
        bv = random.randint(180, 250)
        img = np.full((h, w, 3), bv, dtype=np.uint8)
        t = random.choice(['neutral', 'warm', 'cool', 'yellow'])
        if t == 'warm': img[:,:,2] = np.clip(img[:,:,2].astype(int) + random.randint(5,20), 0, 255).astype(np.uint8)
        elif t == 'cool': img[:,:,0] = np.clip(img[:,:,0].astype(int) + random.randint(5,15), 0, 255).astype(np.uint8)
        elif t == 'yellow':
            img[:,:,1] = np.clip(img[:,:,1].astype(int) + random.randint(3,10), 0, 255).astype(np.uint8)
            img[:,:,2] = np.clip(img[:,:,2].astype(int) + random.randint(3,10), 0, 255).astype(np.uint8)
        noise = np.random.normal(0, random.uniform(2,12), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.6:
            ss = random.uniform(10, 50)
            Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
            st = random.choice(['top','bottom','left','right','corner','center'])
            if st == 'top': mask = ss * (1.0 - Y/h)
            elif st == 'bottom': mask = ss * (Y/h)
            elif st == 'left': mask = ss * (1.0 - X/w)
            elif st == 'right': mask = ss * (X/w)
            elif st == 'corner':
                cx, cy = random.choice([(0,0),(w,0),(0,h),(w,h)])
                mask = ss * (1.0 - np.sqrt((X-cx)**2 + (Y-cy)**2) / np.sqrt(w**2+h**2))
            else:
                mask = ss * np.sqrt((X-w/2)**2 + (Y-h/2)**2) / np.sqrt((w/2)**2+(h/2)**2)
            img = np.clip(img.astype(np.float32) - mask[:,:,np.newaxis], 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            for _ in range(random.randint(1,5)):
                x1,y1 = random.randint(50,w-50), random.randint(50,h-50)
                x2,y2 = x1+random.randint(-40,40), y1+random.randint(-40,40)
                c = tuple(int(np.clip(bv+random.randint(-20,-5),0,255)) for _ in range(3))
                cv2.line(img, (x1,y1), (x2,y2), c, random.randint(1,3))
        if random.random() < 0.4:
            img = cv2.GaussianBlur(img, (random.choice([3,5]),random.choice([3,5])), 0)
        img = np.clip(img.astype(np.int16) + random.randint(-30,15), 0, 255).astype(np.uint8)
        return img
    count = 0
    for sn, n in [('train', EMPTY_TRAIN), ('valid', EMPTY_VAL)]:
        io = CLEAN_DIR / sn / 'images'; lo = CLEAN_DIR / sn / 'labels'
        io.mkdir(parents=True, exist_ok=True); lo.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = make_empty(count+i)
            fn = f"empty_syn_{sn}_{i:04d}"
            cv2.imwrite(str(io / f"{fn}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lo / f"{fn}.txt").write_text('')
            count += 1
        print(f"  {sn}: {n} synthetic backgrounds")
    ted = CLEAN_DIR / 'test_empty' / 'images'
    ted.mkdir(parents=True, exist_ok=True)
    for i in range(EMPTY_TEST):
        cv2.imwrite(str(ted / f"empty_syn_test_{i:04d}.jpg"), make_empty(count+i), [cv2.IMWRITE_JPEG_QUALITY, 92])
    count += EMPTY_TEST
    print(f"  test_empty: {EMPTY_TEST} synthetic backgrounds")
    print(f"  Total: {count}")

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    all_images, groups = audit_dataset()
    resplit_dataset(all_images, groups)
    generate_empty_backgrounds()
    # Create swart.yaml
    Path('swart.yaml').write_text(
        "train: dataset_clean/train/images\nval: dataset_clean/valid/images\ntest: dataset_clean/test/images\n\nnc: 6\nnames: ['electronics', 'glass', 'metal', 'organic', 'paper', 'plastic']\n"
    )
    print("\n  swart.yaml created")
    # Final count
    print("\n" + "=" * 60)
    print("FINAL COUNT")
    print("=" * 60)
    for s in ['train', 'valid', 'test']:
        d = CLEAN_DIR / s / 'images'
        if d.exists():
            n = len(list(d.glob('*')))
            ne = sum(1 for f in (CLEAN_DIR/s/'labels').glob('*') if f.stat().st_size == 0) if (CLEAN_DIR/s/'labels').exists() else 0
            print(f"  {s:6s}: {n:6d} images ({ne} empty)")
    te = CLEAN_DIR / 'test_empty' / 'images'
    if te.exists(): print(f"  test_empty: {len(list(te.glob('*')))} images")
    print("\n[DONE] Dataset siap. Jalankan: python train_swart.py")
