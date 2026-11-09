"""
SWART - Fix Leakage + Add Synthetic Backgrounds
================================================
1. Read leak CSV: move val images with Hamming <= 10 to train.
2. Generate & add 2400 synthetic empty backgrounds to train.
3. Generate 300 synthetic empty backgrounds to test_empty.
"""
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

CLEAN_DIR = Path("dataset_clean")
SEED = 42


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
    leak_csv = Path("leak_check_current.csv")
    thr = 10

    # --- STEP 1: Fix leaks ---
    moved = 0
    if leak_csv.exists():
        with open(leak_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = int(row['hamming'])
                if d <= thr:
                    val_img = Path(row['val_image'])
                    if val_img.exists():
                        stem = val_img.stem
                        # Move image to train
                        dst_img = CLEAN_DIR / 'train' / 'images' / val_img.name
                        shutil.move(str(val_img), str(dst_img))
                        # Move label too
                        val_lab = CLEAN_DIR / 'valid' / 'labels' / (stem + '.txt')
                        dst_lab = CLEAN_DIR / 'train' / 'labels' / (stem + '.txt')
                        if val_lab.exists():
                            shutil.move(str(val_lab), str(dst_lab))
                        moved += 1
        print(f"[STEP 1] Moved {moved} leaky val images to train (threshold={thr})")
    else:
        print(f"[STEP 1] No leak CSV found at {leak_csv} - skipping leak fix")

    # --- STEP 2: Add synthetic backgrounds ---
    train_img = CLEAN_DIR / 'train' / 'images'
    train_lab = CLEAN_DIR / 'train' / 'labels'
    empty_img = CLEAN_DIR / 'test_empty' / 'images'
    empty_lab = CLEAN_DIR / 'test_empty' / 'labels'

    # Check if already added
    existing_syn = list(train_img.glob('synthetic_empty_bin_*.jpg'))
    if len(existing_syn) >= 2400:
        print(f"[STEP 2] Synthetic backgrounds already exist ({len(existing_syn)} files). Skipping.")
    else:
        # Remove any partial synthetic files
        for p in existing_syn:
            p.unlink()
        for p in train_lab.glob('synthetic_empty_bin_*.txt'):
            p.unlink()

        print("[STEP 2] Generating 2700 synthetic empty bin backgrounds...")
        bgs = generate_synthetic_backgrounds()

        for idx, bg in enumerate(bgs[:2400]):
            fn = f"synthetic_empty_bin_{idx:04d}.jpg"
            cv2.imwrite(str(train_img / fn), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (train_lab / f"synthetic_empty_bin_{idx:04d}.txt").touch()

        # Also to test_empty if not already there
        existing_test_empty = list(empty_img.glob('synthetic_empty_test_*.jpg'))
        if len(existing_test_empty) < 300:
            empty_img.mkdir(parents=True, exist_ok=True)
            empty_lab.mkdir(parents=True, exist_ok=True)
            for p in existing_test_empty:
                p.unlink()
            for idx, bg in enumerate(bgs[2400:]):
                fn = f"synthetic_empty_test_{idx:04d}.jpg"
                cv2.imwrite(str(empty_img / fn), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (empty_lab / f"synthetic_empty_test_{idx:04d}.txt").touch()

        print(f"[STEP 2] Added 2400 synthetic backgrounds to train, 300 to test_empty")

    # --- Report final counts ---
    for s in ['train', 'valid', 'test', 'test_empty']:
        img_dir = CLEAN_DIR / s / 'images'
        n = len(list(img_dir.glob('*.*'))) if img_dir.exists() else 0
        print(f"  {s:12s}: {n:5d} images")

    val_count = len(list((CLEAN_DIR / 'valid' / 'images').glob('*.*')))
    print(f"\n[INFO] Remaining val images after leak removal: {val_count}")
    if val_count < 100:
        print("[WARN] Very few val images remaining! Consider re-running resplit.")


if __name__ == '__main__':
    main()

