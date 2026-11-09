"""
SWART Dataset Audit Script - Task 1
Counts per-class instances, finds missing labels, checks bounds, detects augmentation groups.
"""
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

CLASS_NAMES = ['electronic', 'glass', 'metal', 'organic', 'paper', 'plastic']
DATASET_ROOT = Path(r"d:\Farid2\Project\SWART3.0\dataset_raw")

def parse_label(label_path):
    """Parse a YOLO label file, return list of (class_id, cx, cy, w, h)."""
    entries = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return entries
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                entries.append((cls, cx, cy, w, h))
    return entries

def extract_group_name(filename):
    """Extract original name before _jpg.rf.<hash>"""
    m = re.match(r'^(.+?)_jpg\.rf\.[a-f0-9]+\.jpg$', filename, re.IGNORECASE)
    if m:
        return m.group(1)
    # Try without .jpg extension pattern
    m = re.match(r'^(.+?)_png\.rf\.[a-f0-9]+\.jpg$', filename, re.IGNORECASE)
    if m:
        return m.group(1)
    return filename  # fallback

def audit_split(split_name, img_dir, lbl_dir):
    print(f"\n{'='*60}")
    print(f"  SPLIT: {split_name}")
    print(f"{'='*60}")
    
    images = sorted(img_dir.glob('*'))
    image_files = [f for f in images if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    
    print(f"Total images: {len(image_files)}")
    
    class_counter = Counter()
    images_per_class = Counter()
    no_label = []
    empty_label = []
    oob_count = 0
    oob_examples = []
    total_instances = 0
    groups = defaultdict(list)
    
    for img_path in image_files:
        stem = img_path.stem
        label_path = lbl_dir / (stem + '.txt')
        
        # Group detection
        group_name = extract_group_name(img_path.name)
        groups[group_name].append(img_path.name)
        
        if not label_path.exists():
            no_label.append(img_path.name)
            continue
        
        entries = parse_label(label_path)
        if len(entries) == 0:
            empty_label.append(img_path.name)
            continue
        
        img_classes = set()
        for cls, cx, cy, w, h in entries:
            class_counter[cls] += 1
            img_classes.add(cls)
            total_instances += 1
            
            # Check bounds
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                oob_count += 1
                if len(oob_examples) < 5:
                    oob_examples.append((img_path.name, cls, cx, cy, w, h))
        
        for cls in img_classes:
            images_per_class[cls] += 1
    
    # Report
    print(f"\nClass distribution (instances):")
    for i, name in enumerate(CLASS_NAMES):
        count = class_counter.get(i, 0)
        img_count = images_per_class.get(i, 0)
        print(f"  {i}: {name:12s} -> {count:6d} instances in {img_count:6d} images")
    
    print(f"\nTotal label instances: {total_instances}")
    print(f"Images without label file: {len(no_label)}")
    print(f"Images with empty label (background): {len(empty_label)}")
    if no_label[:5]:
        print(f"  Examples (no label): {no_label[:3]}")
    if empty_label[:5]:
        print(f"  Examples (empty): {empty_label[:3]}")
    
    print(f"\nOut-of-bounds labels: {oob_count}")
    if oob_examples:
        for ex in oob_examples:
            print(f"  {ex}")
    
    # Group stats
    group_sizes = [len(v) for v in groups.values()]
    multi_groups = [g for g, v in groups.items() if len(v) > 1]
    print(f"\nAugmentation groups:")
    print(f"  Total unique groups: {len(groups)}")
    print(f"  Groups with >1 image (augmented): {len(multi_groups)}")
    if group_sizes:
        print(f"  Avg copies per group: {sum(group_sizes)/len(group_sizes):.2f}")
        print(f"  Max copies in a group: {max(group_sizes)}")
        print(f"  Min copies in a group: {min(group_sizes)}")
    
    # Show top groups by size
    top_groups = sorted(groups.items(), key=lambda x: -len(x[1]))[:5]
    print(f"  Top 5 groups by size:")
    for gname, gfiles in top_groups:
        print(f"    '{gname}' -> {len(gfiles)} copies")
    
    return {
        'total_images': len(image_files),
        'class_instances': dict(class_counter),
        'images_per_class': dict(images_per_class),
        'no_label': no_label,
        'empty_label': empty_label,
        'oob_count': oob_count,
        'groups': groups,
        'total_instances': total_instances,
    }

if __name__ == '__main__':
    print("SWART Dataset Audit")
    print("="*60)
    
    # Check data.yaml class order
    print("\ndata.yaml class order:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")
    print("  -> Matches required order: electronic(0), glass(1), metal(2), organic(3), paper(4), plastic(5)")
    
    results = {}
    for split in ['train', 'valid']:
        img_dir = DATASET_ROOT / split / 'images'
        lbl_dir = DATASET_ROOT / split / 'labels'
        results[split] = audit_split(split, img_dir, lbl_dir)
    
    # Combined stats
    print(f"\n{'='*60}")
    print(f"  COMBINED STATS")
    print(f"{'='*60}")
    total_imgs = sum(r['total_images'] for r in results.values())
    print(f"Total images: {total_imgs}")
    
    combined_classes = Counter()
    for r in results.values():
        combined_classes.update(r['class_instances'])
    print(f"\nCombined class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name:12s} -> {combined_classes.get(i, 0):6d} instances")
    
    # Cross-split group analysis
    print(f"\nCross-split group overlap check:")
    train_groups = set(results['train']['groups'].keys())
    valid_groups = set(results['valid']['groups'].keys())
    overlap = train_groups & valid_groups
    print(f"  Train groups: {len(train_groups)}")
    print(f"  Valid groups: {len(valid_groups)}")
    print(f"  Overlapping groups: {len(overlap)}")
    if overlap:
        overlap_list = sorted(overlap)[:10]
        print(f"  Examples: {overlap_list}")
