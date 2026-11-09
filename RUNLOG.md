# SWART — Run Log

Catatan semua versi, perintah, seed, dan hasil selama proses retrain.

---

## Langkah 0 — Audit Lingkungan

**Tanggal:** 2026-11-07

### Versi Software

| Komponen | Versi |
|----------|-------|
| Python | 3.13.3 |
| ultralytics | 8.4.164 |
| torch | 2.7.0 |
| torchaudio | 2.7.0 |
| torchvision | 0.22.0 |
| onnxruntime | 1.30.0 |
| opencv-python | 4.11.0.86 |
| numpy | 2.3.3 |
| pandas | 2.3.2 |
| NVIDIA Driver | 566.36 |
| CUDA | 12.7 |

### GPU

- **NVIDIA GeForce RTX 2050** — 4 GB VRAM, CUDA 12.7
- Training dapat dilakukan di lokal (tidak perlu Colab)

### File Pendukung

| File | Status |
|------|--------|
| `swart_core.py` | ✅ Ada |
| `prune_onnx.py` | ✅ Ada |
| `calibrate.py` | ❌ TIDAK ADA — perlu dibuat |
| `detect_pi_fast.py` | ✅ Ada |
| `test_yolov9_iriun_fast.py` | ✅ Ada |
| `bench_models.py` | ❌ TIDAK ADA — perlu dibuat |
| `capture_dataset.py` | ✅ Ada |
| `check_leakage.py` | ✅ Ada |
| `eval_empty_fp.py` | ✅ Ada |
| `weights/best2_main.onnx` | ✅ Ada (27.20 MB, pruned, input 320) |
| `weights/best_main.onnx` | ✅ Ada (27.19 MB, pruned, input 224) |

### Model Weights Tersedia (SEMUA DIPERTAHANKAN UNTUK BACKUP)

| File | Ukuran | Keterangan |
|------|--------|------------|
| `best.pt` | 19.36 MB | PyTorch v1 |
| `best2.pt` | 19.36 MB | PyTorch v2 |
| `best.onnx` | 36.89 MB | ONNX v1 (full, unpruned) |
| `best2.onnx` | 36.90 MB | ONNX v2 (full, unpruned) |
| `best2_main.onnx` | 27.20 MB | Pruned, input 320 (MODEL UTAMA SAAT INI) |
| `best_main.onnx` | 27.19 MB | Pruned, input 224 |
| `SWART.onnx` | 36.88 MB | Model SWART (kelas: electronic) |
| `wasteaiupdate.pt` | 11.74 MB | Model update |

### Dataset

- **Dataset 6 kelas TIDAK ADA di lokal** — perlu di-download ulang dari Roboflow
- `dataset_empty/` belum ada
- Training lama dilakukan di Colab, dataset di-download dari Roboflow API

### Training Lama (dari opt.yaml)

- Model: YOLOv9-s (`yolov9-s.yaml`)
- Input: 320 px
- Epochs: 50, Batch: 104
- Augmentasi: mosaic=1.0, mixup=0.15, scale=0.9
- Seed: 0

### Hal yang Belum Diverifikasi

- [x] Dataset asli — DITEMUKAN: Roboflow workspace `swart`, project `swart`
- [ ] Angka mAP50 0.899 / mAP50-95 0.758 dari training lama belum bisa dikonfirmasi ulang
- [ ] Apakah dataset augmentasi bocor antara train/valid

---

## Langkah 0b — Download Dataset

**Tanggal:** 2026-11-07

### Dataset Roboflow

| Info | Detail |
|------|--------|
| Workspace | `swart` |
| Project | `swart` (id: swart/swart) |
| Gambar asli | 36.963 (22.048 train + 14.915 valid) |
| Kelas (6) | electronic, glass, metal, organic, paper, plastic |
| Lisensi | CC BY 4.0 |
| Versi tersedia | 3 (raw), 8 (augmented 58K), 9 (augmented 62K) |

### Instance per Kelas (raw dataset)

| Kelas | Instance |
|-------|----------|
| glass | 46.784 |
| plastic | 37.844 |
| electronic | 44.204 |
| metal | 29.886 |
| organic | 64.413 |
| paper | 15.272 |

### Download

- Downloaded **Version 3** (36.963 gambar, TANPA augmentasi) format YOLOv8
- Disimpan di `dataset_raw/` (29.419 train + 7.544 valid)
- Versi ini di-resize ke 640×640 (stretch) oleh Roboflow preprocessing

### Model Baru untuk Training

| Model | Parameter | Status |
|-------|-----------|--------|
| YOLO11n | 2.6M | ✅ yolo11n.pt downloaded |
| YOLO26n | 2.6M | ✅ yolo26n.pt downloaded |

Perbandingan: YOLOv9-s = 7.2M params → YOLO11n/26n = 2.6M params (2.8× lebih kecil)

---

## Langkah 1 — Keputusan Hardware

**Tanggal:** 2026-11-07

Keputusan dicatat di `HARDWARE_FREEZE.md`:
- Bak putih dipertahankan
- Pencahayaan: kondisi yang ada
- Kamera: tetap (640×480 MJPG)
- Exposure: auto

---

## Langkah 2-4 — Resplit Dataset Bersih (Bebas Kebocoran) & Latar Belakang Sintetis

**Tanggal:** 2026-11-08

### Pemecahan Masalah Kebocoran Data (Data Leakage)
- Ditemukan bahwa dataset lama dan split awal memiliki kebocoran signifikan (~42,8% pasangan dengan jarak Hamming $\le 10$ akibat rotasi/augmentasi offline Roboflow).
- **Solusi**: Dibuat `resplit_final.py`:
  1. Pengelompokan berbasis nama bersih (menghapus akhiran `_0`, `_90`, `_180`, `_270` dan roboflow hash) -> 21.425 grup independen.
  2. Perhitungan perceptual dHash untuk 8 orientasi D4 (4 rotasi + flip).
  3. Perhitungan jarak Hamming berpasangan penuh antar seluruh 21.425 grup menggunakan akselerasi GPU (PyTorch CUDA di RTX 2050) -> 798.750 pasangan terhubung digabungkan (Union-Find) menjadi 6.794 klaster independen.
  4. Stratified Split 80/10/10 berbasis klaster:
     - **Train**: 36.703 gambar (34.303 gambar objek sampah + 2.400 latar belakang bak kosong sintetis).
     - **Valid**: 1.343 gambar objek sampah independen (**Overlap Klaster Train $\leftrightarrow$ Val: 0 / 0.0%**).
     - **Test**: 1.317 gambar independen.
     - **Test Empty**: 300 gambar latar belakang bak kosong sintetis untuk menguji false-positive deteksi bak kosong.
  5. Cache hash disimpan di `group_hashes_cache.npz`.

---

## Langkah 5 — Training Model YOLO11n

**Tanggal:** 2026-11-08

- **Akselerator**: NVIDIA GeForce RTX 2050 (4 GB VRAM) dengan PyTorch 2.14.1+cu126.
- **Konfigurasi**:
  - Model: `yolo11n.pt` (pretrained, 2.6M parameter, 6.5 GFLOPs pada 320 px).
  - Resolusi input: $320 \times 320$.
  - Batch size: 16.
  - Epochs: 150 (Patience 30).
  - Optimizer: MuSGD (lr=0.01, momentum=0.9).
  - AMP (Automatic Mixed Precision): Aktif.
  - close_mosaic: 15 epoch terakhir.
  - Direktori output: `runs/detect/runs/yolo11n_320/`.
- **Status**: Sedang berlangsung. Iterasi batch berjalan pada ~10 it/s (~160 gambar/detik), estimasi ~3,6 menit per epoch.


