# Panduan Training Model SWART (YOLO11n & YOLO26n)

Panduan ini ditujukan bagi anggota tim yang ingin menjalankan proses training model SWART di komputer / server masing-masing.

---

## 1. Persyaratan Sistem & Instalasi

### A. Clone Repo & Masuk ke Folder Project
```bash
git clone <URL_REPOSITORY>
cd SWART3.0
```

### B. Install Dependensi Python
Rekomendasi Python **3.10 - 3.13**:
```bash
pip install ultralytics opencv-python numpy onnx onnxruntime
```

**Untuk GPU NVIDIA (Wajib CUDA untuk training cepat):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
*Verifikasi GPU di terminal:*
```bash
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Persiapan Dataset (Zero-Leakage & Bak Kosong Sintetis)

Jika folder `dataset_clean/` belum ada di komputer teman:
1. Pastikan dataset mentah ada di `dataset_raw/` (atau download versi 3 dari Roboflow swart/swart).
2. Jalankan script clustering dHash & split zero-leakage:
```bash
python resplit_final.py
```
*Script ini otomatis mengelompokkan data berdasarkan D4 8-orientasi dHash, menghilangkan kebocoran data (0% leakage), dan menyintesis latar belakang bak kosong.*

---

## 3. Menjalankan Training

### Opsi A: Training YOLO11n (Rekomendasi Utama Raspberry Pi)
```bash
python train_swart.py --model yolo11n --imgsz 320 --epochs 150 --patience 30 --batch 16 --device 0
```

### Opsi B: Training YOLO26n
```bash
python train_swart.py --model yolo26n --imgsz 320 --epochs 150 --patience 30 --batch 16 --device 0
```

### Opsi C: Training Keduanya Otomatis
```bash
python train_swart.py --model both --imgsz 320 --epochs 150 --patience 30 --batch 16 --device 0
```

> **Tips Penyesuaian Hardware:**
> - Jika GPU memiliki VRAM 8 GB atau lebih: ubah `--batch 32` atau `--batch 64`.
> - Jika VRAM 4 GB (seperti RTX 2050/3050 laptop): gunakan `--batch 16` (hanya memakan ~0.8 GB VRAM).
> - Jika menggunakan CPU: ganti `--device 0` menjadi `--device cpu`.

---

## 4. Evaluasi & Export Model

Setelah training selesai, bobot terbaik akan tersimpan di `runs/detect/runs/<model>_320/weights/best.pt`.

### A. Evaluasi mAP di Test Set
```bash
python eval_swart.py
```

### B. Export ke ONNX (Format untuk Raspberry Pi)
```bash
yolo export model=runs/detect/runs/yolo11n_320/weights/best.pt format=onnx imgsz=320 simplify=True
```
Salin file `.onnx` yang dihasilkan ke:
```bash
cp runs/detect/runs/yolo11n_320/weights/best.onnx weights/yolo11n_best.onnx
```
*(PENTING: Jangan menghapus atau menimpa `weights/best2_main.onnx` lama)*

### C. Uji False-Positive Bak Kosong
```bash
python eval_empty_fp.py --weights weights/yolo11n_best.onnx --images dataset_clean/test_empty/images --imgsz 320
```

