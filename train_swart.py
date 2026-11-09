"""
SWART - Training script untuk YOLO11n dan YOLO26n
==================================================
Jalankan SETELAH dataset_clean/ sudah siap (dari resplit_dataset.py).

Pemakaian:
    python train_swart.py                    # train kedua model
    python train_swart.py --model yolo11n    # hanya YOLO11n
    python train_swart.py --model yolo26n    # hanya YOLO26n
"""
import argparse
import os
import sys
from pathlib import Path

def train_model(model_name, data_yaml, imgsz=320, epochs=150, patience=30, batch=16, seed=0, device=None):
    """Train satu model YOLO."""
    from ultralytics import YOLO
    
    model = YOLO(f'{model_name}.pt')
    
    run_name = f'{model_name}_{imgsz}'
    print(f"\n{'='*60}")
    print(f"Training: {model_name} | imgsz={imgsz} | epochs={epochs} | batch={batch}")
    print(f"Data: {data_yaml} | Device: {device or 'auto'}")
    print(f"Run name: {run_name}")
    print(f"{'='*60}\n")
    
    kwargs = dict(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        patience=patience,
        batch=batch,
        close_mosaic=15,
        seed=seed,
        deterministic=True,
        workers=4,
        project='runs',
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
    if device is not None:
        kwargs['device'] = device
    
    results = model.train(**kwargs)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['yolo11n', 'yolo26n', 'both'], default='both')
    ap.add_argument('--data', default='swart.yaml')
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16, help='Turunkan jika OOM (RTX 2050 = 4GB VRAM)')
    ap.add_argument('--device', default='0', help='Device CUDA (misal 0, 0,1), mps, atau cpu')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    
    data_path = Path(a.data)
    if not data_path.exists():
        sys.exit(f"[ERROR] {a.data} tidak ditemukan. Jalankan resplit_dataset.py dulu.")
    
    models = ['yolo11n', 'yolo26n'] if a.model == 'both' else [a.model]
    
    for m in models:
        try:
            train_model(m, str(data_path), a.imgsz, a.epochs, a.patience, a.batch, a.seed, a.device)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                new_batch = a.batch // 2
                print(f"\n[WARN] OOM dengan batch={a.batch}. Mencoba batch={new_batch}...")
                if new_batch >= 4:
                    train_model(m, str(data_path), a.imgsz, a.epochs, a.patience, new_batch, a.seed, a.device)
                else:
                    print(f"[ERROR] Batch size terlalu kecil. Coba kurangi imgsz atau gunakan Colab.")
                    raise
            else:
                raise
    
    print("\n[DONE] Training selesai. Cek hasil di runs/")
    print("Langkah selanjutnya:")
    print("  1. python eval_swart.py              # evaluasi di test set")
    print("  2. yolo export model=runs/yolo11n_320/weights/best.pt format=onnx imgsz=320 simplify=True")


if __name__ == '__main__':
    main()

