"""
SWART - Evaluasi model di test set (Langkah 6)
================================================
Membandingkan model lama (best2_main.onnx) dengan model baru (yolo11n, yolo26n).

Pemakaian:
    python eval_swart.py                           # evaluasi semua model
    python eval_swart.py --model runs/yolo11n_320   # evaluasi satu model
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def eval_ultralytics(model_path, data_yaml, imgsz=320, split='test'):
    """Evaluasi model .pt atau .onnx menggunakan Ultralytics."""
    from ultralytics import YOLO
    
    model = YOLO(str(model_path))
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Data: {data_yaml} | Split: {split} | imgsz: {imgsz}")
    print(f"{'='*60}\n")
    
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        split=split,
        verbose=True,
        plots=True,
    )
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='swart.yaml')
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--model', default=None, help='Specific model dir to evaluate')
    a = ap.parse_args()
    
    if a.model:
        # Evaluate single model
        best = Path(a.model) / 'weights' / 'best.pt'
        if not best.exists():
            best = Path(a.model)  # might be direct .pt or .onnx path
        eval_ultralytics(best, a.data, a.imgsz, a.split)
    else:
        # Evaluate all available models
        models = []
        
        # Model lama (ONNX)
        old_model = Path('weights/best2_main.onnx')
        if old_model.exists():
            models.append(('YOLOv9-s (lama)', old_model))
        
        # Model baru
        for name in ['yolo11n_320', 'yolo26n_320']:
            best = Path(f'runs/detect/runs/{name}/weights/best.pt')
            if not best.exists():
                best = Path(f'runs/{name}/weights/best.pt')
            if best.exists():
                models.append((name, best))
        
        if not models:
            print("[ERROR] Tidak ada model yang ditemukan. Jalankan train_swart.py dulu.")
            sys.exit(1)
        
        print(f"[INFO] Evaluasi {len(models)} model di split '{a.split}':\n")
        for name, path in models:
            print(f"  - {name}: {path}")
        
        results_all = {}
        for name, path in models:
            try:
                r = eval_ultralytics(path, a.data, a.imgsz, a.split)
                results_all[name] = r
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
        
        # Print comparison
        if results_all:
            print(f"\n{'='*80}")
            print(f"{'Model':<25} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
            print(f"{'-'*80}")
            for name, r in results_all.items():
                box = r.box
                print(f"{name:<25} {box.map50:>8.3f} {box.map:>10.3f} "
                      f"{np.mean(box.p):>8.3f} {np.mean(box.r):>8.3f}")
            print(f"{'='*80}")


if __name__ == '__main__':
    main()

