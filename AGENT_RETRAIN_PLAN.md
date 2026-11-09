# SWART — Rencana Training Ulang (untuk agent)

> **Untuk pengguna:** taruh file ini di folder proyek bersama file-file pendukung (lihat bagian 4), lalu tempel prompt ini ke agent:
>
> *"Baca `AGENT_RETRAIN_PLAN.md` sampai habis, lalu kerjakan berurutan dari Langkah 0. Berhenti dan tanya saya di setiap titik bertanda **BERHENTI**. Jangan mengarang data atau angka; kalau sesuatu tidak ada, minta ke saya."*

Dokumen ini berdiri sendiri: agent tidak punya riwayat percakapan sebelumnya. Semua yang perlu diketahui ada di sini.

---

## 1. Konteks proyek

SWART = tempat sampah pintar. Kamera di atas area scan (bak putih hasil 3D print, ±13,2 × 10 cm) memotret sampah; model deteksi mengklasifikasi ke 6 kelas; Raspberry Pi 4 menggerakkan 2 servo + LED untuk memilah.

- **Kelas (urutan WAJIB persis ini):** `electronics, glass, metal, organic, paper, plastic`
- **Perangkat target:** Raspberry Pi 4 (CPU Cortex-A72, tanpa GPU/NPU). Inference lewat `onnxruntime` (CPU).
- **Model sekarang:** YOLOv9-s, input 320 px, dilatih 50 epoch (repo YOLOv9 `train_dual.py`), dataset Roboflow dengan 6 kelas. Hasil lama: mAP50 0,899, mAP50-95 0,758 (angka ini kemungkinan terlalu optimistis, lihat Langkah 3).
- **Resolusi kamera saat dipakai:** 640×480, letterbox ke 320×320 (kode ada di `swart_core.py`).

## 2. Masalah yang diselesaikan

1. **Bak putih kosong terbaca `paper`** (confidence ±0,37–0,48). Dataset tidak punya satu pun gambar bak kosong, jadi model tidak pernah belajar bahwa "putih polos = tidak ada objek". Menaikkan threshold tidak membantu (kertas asli juga ±0,4).
2. **Terlalu berat untuk Pi 4.** YOLOv9-s = 7,2 M parameter / 26,7 GFLOPs di 640 px. YOLO11n = 2,6 M / 6,5 GFLOPs; YOLO26n = 2,4 M / 5,5 GFLOPs (tabel resmi Ultralytics). Di 320 px FLOPs turun ±4×.
3. **Validasi mungkin bocor.** Jumlah gambar train/valid di blueprint (50.904 / 11.511, rasio 81,6/18,4 dari total *setelah* augmentasi) tidak cocok dengan klaim "split dulu, baru augmentasi" (18% dari 36.963 = 6.653). Perlu dibuktikan, bukan diasumsikan.

## 3. Aturan kerja (wajib)

- **Jangan** menaruh API key/secret di kode, log, atau commit. Roboflow API key lama sempat tertulis di notebook dan harus dianggap bocor: minta pengguna membuat ulang, lalu baca dari variabel lingkungan `ROBOFLOW_API_KEY`.
- **Jangan** memasang `ultralytics`/`torch` di Raspberry Pi 4 (ada laporan crash "Illegal instruction" di Pi 4). Training & ekspor dilakukan di PC/Colab. Pi hanya menjalankan `onnxruntime`/`opencv`/`numpy`.
- **Jangan** memakai set val/test untuk memilih hyperparameter lalu melaporkannya sebagai hasil akhir. Pilih di val, laporkan di test.
- **Jangan** mengaugmentasi gambar bak kosong secara offline (duplikat nyaris identik = bocor + tidak menambah informasi).
- Label kosong = latar belakang. Gambar tanpa objek **tidak perlu dilabeli**; file `.txt` kosong (atau tanpa file) sudah cukup (Ultralytics: "No labels are required for background images").
- Catat semua versi (python, ultralytics, torch, onnxruntime), seed, dan perintah persis ke `RUNLOG.md`.
- Bila ragu soal sebuah angka, ukur; jangan menulis angka dari ingatan. Tandai yang belum diverifikasi sebagai "belum diverifikasi".
- Lisensi: Ultralytics berlisensi AGPL-3.0. Beri tahu pengguna sebelum dipakai di produk yang dipublikasikan.

## 4. File pendukung yang harus ada di folder proyek

`swart_core.py`, `prune_onnx.py`, `calibrate.py`, `detect_pi_fast.py`, `test_yolov9_iriun_fast.py`, `bench_models.py`, `capture_dataset.py`, `check_leakage.py`, `eval_empty_fp.py`, `weights/best2_main.onnx` (model lama, cabang auxiliary sudah dipangkas, input 320), `weights/best_main.onnx` (input 224).

Skrip-skrip ini sudah diuji di sandbox (x86, 1 core), **belum** di Raspberry Pi sungguhan.

## 5. Langkah-langkah

### Langkah 0 — Audit lingkungan dan minta prasyarat

1. Cek GPU (`nvidia-smi`). Tanpa GPU, training di CPU terlalu lambat; tawarkan Google Colab (T4) dan siapkan notebook.
2. Buat virtualenv, `pip install ultralytics onnx onnxruntime opencv-python numpy pandas`. Catat versinya.
3. Tanya pengguna dan tunggu jawaban (**BERHENTI**):
   - Di mana dataset 6 kelas asli? Minta **export seluruh gambar tanpa augmentasi** (Roboflow: buat versi tanpa augmentasi, gabungkan train/valid/test). Jika hanya ada export beraugmentasi, catat itu (lihat Langkah 4).
   - Compute apa yang dipakai (GPU lokal / Colab)?
   - Apakah pengguna bisa melabeli ±50–100 gambar per kelas untuk set uji lapangan (Langkah 7)? Agent tidak bisa melakukan ini sendiri.

### Langkah 1 — Bekukan keputusan hardware (**BERHENTI**)

Gambar bak kosong hanya berlaku untuk kondisi hardware yang sama. Minta pengguna memutuskan dan menulis ke `HARDWARE_FREEZE.md`:

| Keputusan | Catatan |
|---|---|
| Warna permukaan bak: tetap putih atau diganti hitam/non-putih? | Blueprint sendiri mengusulkan non-putih (Pengembangan Selanjutnya no. 3). Ini perbaikan akar masalah. |
| Pencahayaan: lampu tambahan/tetap? | Pencahayaan konsisten mengurangi variasi. |
| Posisi kamera & area scan (crop) | Mengubah crop setelah training mengubah skala objek. |
| Exposure: kunci (`--lock-exposure`) atau auto? | Pakai pilihan yang sama saat merekam data dan saat deployment. |

Jika pengguna **akan** mengubah hardware nanti, jangan rekam bak kosong sekarang. Selesaikan hardware dulu. Dataset 6 kelas lama kemungkinan direkam di bak putih; jika permukaan berubah, tanyakan apakah perlu merekam ulang sebagian.

### Langkah 2 — Rekam dataset bak kosong (dijalankan pengguna di Pi)

Agent menyiapkan perintah dan checklist; pengguna menjalankannya di Raspberry Pi (bak kosong) lalu menyalin folder `dataset_empty/` ke proyek.

Pakai `capture_dataset.py`. Satu sesi = satu kondisi. Target minimal per sesi 100–150 frame:

| Tag sesi | Kondisi |
|---|---|
| `normal_pagi`, `normal_siang`, `normal_malam` | cahaya ruangan sebenarnya di jam berbeda |
| `lampu_nyala`, `lampu_mati`, `redup`, `terang_jendela` | variasi cahaya & sinar matahari lewat jendela |
| `servo_putar` | servo/platform di berbagai posisi (home, putar kiri, putar kanan) sambil merekam |
| `led_putih_nyala` | LED putih aktif seperti saat sortir |
| `tangan_lewat` | tangan/lengan masuk-keluar area scan **tanpa benda** (negatif keras) |
| `kamera_panas` | setelah alat menyala ±30 menit (kamera hangat, noise berbeda) |
| `kotor_noda` | bak apa adanya (noda, debu, bayangan) |

```bash
python capture_dataset.py --source 0 --tag normal_pagi --count 150 --interval 1.5
python capture_dataset.py --source 0 --tag servo_putar --count 150 --interval 0.7
python capture_dataset.py --source 0 --tag cek_model_lama --weights weights/best2_main.onnx --count 200   # mencatat false positive ke dataset_empty/hardneg/
```

**Target jumlah (rekomendasi, bukan angka dari sumber):** ±10% dari jumlah gambar train berobjek, **minimal 15 sesi berbeda**. Keberagaman sesi lebih penting daripada jumlah frame (frame berurutan di bak statis nyaris kembar). Panduan umum Ultralytics menyarankan 0–10% gambar latar; kasus ini khusus (satu latar yang selalu sama dan terbukti membingungkan model), jadi mulai dari 10% dan evaluasi. Naikkan sampai ±15% hanya jika false positive di Langkah 6 masih tinggi.

Sisihkan **minimal 3 sesi sebagai test kosong** dan 3 sesi sebagai val kosong (tidak dipakai training sama sekali). Pemisahan per **sesi**, bukan per frame acak.

### Langkah 3 — Audit dataset lama

1. Inventaris: jumlah gambar & instance per kelas, urutan kelas di `data.yaml` (harus sama dengan bagian 1), format label (YOLO txt), gambar tanpa label, label di luar [0,1].
2. Deteksi augmentasi: nama file Roboflow berbentuk `<nama>_jpg.rf.<hash>.jpg`; semua file dengan `<nama>` yang sama adalah salinan dari satu gambar asli. Hitung berapa grup dan rata-rata salinan per grup.
3. Ukur kebocoran dataset lama: `python check_leakage.py --train <train/images> --val <valid/images> --thr 10 --csv leak_lama.csv`. Pakai `--thr 10` (lebih ketat dari default 6; pada uji sintetis, default 6 melewatkan 1 dari 10 kebocoran). Laporkan persentasenya. **Jangan mengubah kesimpulan di blueprint sendiri**: sampaikan ke pengguna.

### Langkah 4 — Susun ulang split (berbasis kelompok)

Tujuan: tidak ada gambar yang sama/mirip di dua split.

1. Kelompokkan gambar berobjek: (a) per `<nama>` sebelum `_jpg.rf.`; (b) gabungkan kelompok yang dHash-nya berjarak ≤ 10 dari 64 bit (union-find; boleh pakai fungsi `dhash` dan `orientations` di `check_leakage.py`, termasuk flip/rotasi). Gambar berurutan dari satu benda yang dijatuhkan akan masuk satu kelompok.
2. Bagi **per kelompok** ke train/val/test ≈ 80/10/10, dijaga proporsi kelas kira-kira sama (stratified per kelas dominan kelompok). Simpan `split_manifest.csv` (file, kelompok, split, kelas).
3. Val dan test berisi hanya **satu gambar asli per kelompok** (bukan salinan augmentasi). Salinan augmentasi (jika ada) hanya boleh ada di train.
4. Gambar kosong: split per sesi. Train ≈ 10% dari jumlah train berobjek; val kosong ≈ 10% dari val; sesi test kosong dipisah di folder `dataset_empty_test/` (dipakai `eval_empty_fp.py`, bukan ikut mAP).
5. Susun `swart.yaml` (path train/val/test, `nc: 6`, `names` sesuai urutan wajib).
6. Verifikasi: `check_leakage.py` untuk train vs val dan train vs test dengan `--thr 10`. Wajib ≤ 2%. Jika lebih, ulangi langkah 1–2 dengan ambang kelompok lebih longgar.

### Langkah 5 — Training

Latih **dua kandidat dengan data, seed, dan epoch sama**, di 320 px:

```bash
yolo detect train model=yolo11n.pt data=swart.yaml imgsz=320 epochs=150 patience=30 batch=32 \
     close_mosaic=15 seed=0 deterministic=True workers=8 project=runs name=yolo11n_320
yolo detect train model=yolo26n.pt data=swart.yaml imgsz=320 epochs=150 patience=30 batch=32 \
     close_mosaic=15 seed=0 deterministic=True workers=8 project=runs name=yolo26n_320
```

- Turunkan `batch` jika kehabisan memori GPU. Biarkan augmentasi default Ultralytics lainnya. Training lama memakai `scale 0.9`, `close_mosaic 0`, 50 epoch dan belum konvergen (mAP masih naik di epoch 49). Itu yang diubah di sini.
- Jika nama bobot YOLO26 berbeda di versi Ultralytics yang terpasang, baca dokumentasi resminya; jangan menebak.
- Pilihan model akhir ditentukan di Langkah 6 dan 8, bukan di sini.

### Langkah 6 — Evaluasi (di test, bukan val)

Untuk **model lama dan kedua model baru**, di set test yang sama:

1. Metrik per kelas: P, R, AP50, AP50-95, dan confusion matrix. Untuk model lama, pakai `ultralytics` (`YOLO('weights/best2_main.onnx').val(...)`) bila bisa dimuat; jika tidak, tulis skrip evaluasi sendiri dan jelaskan caranya.
2. **False positive bak kosong:** `python eval_empty_fp.py --weights <model.onnx> --images dataset_empty_test/ --imgsz 320`, sekali tanpa `--calib` dan sekali dengan `--calib calib/empty_ref.npz`.
3. Ambil dari kurva F1 nilai confidence terbaik per model (catat; model lama di 0,222).
4. **Loop hard-negative (maks. 2 putaran):** jalankan `capture_dataset.py --weights <model baru> ...` pada sesi bak kosong baru; frame di `dataset_empty/hardneg/` dipindah ke **train** sebagai sesi baru (bukan ke val/test), latih ulang model terpilih, ukur lagi.

### Langkah 7 — Set uji lapangan (butuh pengguna)

Metrik di val/test dari dataset yang sama belum membuktikan kinerja di lapangan. Minta pengguna:

- Merekam ≥ 50–100 gambar per kelas dengan sampah **nyata** di alat, di hari dan kondisi cahaya **berbeda** dari data training (`capture_dataset.py` bisa dipakai dengan tag sesuai, lalu dilabeli).
- Melabeli dengan Roboflow/CVAT/Label Studio, satu kotak per objek, kelas sesuai urutan wajib.
- Set ini **tidak boleh** dipakai untuk training. Jika dilewati, nyatakan di laporan bahwa akurasi lapangan belum terverifikasi.

### Langkah 8 — Ekspor dan benchmark

1. `yolo export model=<best.pt> format=onnx imgsz=320 simplify=True dynamic=False`.
2. Periksa `onnxruntime.InferenceSession(...).get_outputs()[0].shape`.
   - Bentuk `[1, 4+6, N]` = kompatibel dengan `swart_core.OnnxDetector.detect` (tanpa perubahan).
   - Bentuk lain (mis. head end-to-end/NMS-free YOLO26, sering `[1, 300, 6]`) = **ubah pascapemrosesan** di `swart_core.py` (tidak perlu NMS, kolom: x1,y1,x2,y2,conf,kelas) dan cocokkan dengan dokumentasi ekspor Ultralytics. Belum diverifikasi oleh penulis dokumen ini.
3. Uji paritas: untuk 50 gambar test, deteksi ONNX via `swart_core` harus sama dengan `model.predict` PyTorch (kotak ±2 px, confidence ±0,02, kelas sama). Laporkan selisih.
4. Salin ONNX ke Pi dan jalankan di Pi 4 (pengguna): `python bench_models.py weights/best2_main.onnx <yolo11n.onnx> <yolo26n.onnx> --imgsz 320 --threads 4`. Pastikan pendingin terpasang (skrip memperingatkan jika suhu ≥ 80 °C).
5. NCNN (opsional): panduan Ultralytics menyebut NCNN tercepat di ARM, tetapi benchmark resminya Pi 5. Ukur di Pi 4 sebelum memutuskan; mengubah pipeline hanya jika hasilnya jelas lebih baik.

### Langkah 9 — Integrasi dan uji akhir

1. Simpan model terpilih di `weights/` (pertahankan `best2_main.onnx` untuk rollback). Perbarui perintah di `waste.sh` agar memakai `detect_pi_fast.py --weights <model baru>`.
2. Jalankan ulang `calibrate.py` di bak kosong yang sekarang. Pertahankan filter kalibrasi sebagai pengaman kedua walau model baru sudah bagus.
3. `python detect_pi_fast.py --dry-run --show` lalu uji dengan hardware: ≥ 20 benda nyata per kelas, catat benar/salah + kasus bak kosong 10 menit. Stel `--conf`, `--vote-min`, `--class-conf` seperlunya.
4. Tulis `HASIL.md`: tabel perbandingan model lama vs baru (Langkah 6, 8, 9), keputusan, dan apa yang belum terverifikasi.

## 6. Kriteria lulus (usulan; pengguna boleh mengubah angkanya)

| Kriteria | Ambang usulan |
|---|---|
| Kebocoran train↔val/test (`check_leakage`, thr 10) | ≤ 2% |
| False positive di sesi bak kosong (test), conf 0,25, **tanpa** filter kalibrasi | ≤ 0,5% frame |
| AP50-95 per kelas di test baru | tidak turun > 2 poin dibanding model lama **di test yang sama** |
| Latensi di Pi 4 yang sama (`bench_models.py`) | ≥ 2× lebih cepat dari `best2_main.onnx` |
| Akurasi uji lapangan nyata | ditetapkan pengguna; wajib dilaporkan per kelas |

Jika ada kriteria yang gagal, jangan menyatakan selesai. Laporkan angka, penyebab yang paling mungkin, dan langkah berikutnya.

## 7. Hasil akhir yang harus diserahkan

`swart.yaml`, `split_manifest.csv`, `HARDWARE_FREEZE.md`, `RUNLOG.md`, `HASIL.md`, folder `runs/` (kedua kandidat), `best.onnx` terpilih (+ `best.pt`), keluaran `eval_empty_fp.py` (lama vs baru), keluaran `bench_models.py` dari Pi, dan daftar hal yang belum terverifikasi.

## 8. Jebakan yang sudah diketahui

- `electronics` (model `best.onnx`/`best2.onnx`) vs `electronic` (`SWART.onnx`): pakai `electronics` di `swart.yaml`; `detect_pi_fast.py` sudah menyamakan keduanya.
- `best2.pt` di arsip lama memakai dataset berbeda dari `best2.onnx` (kelas `electronic` vs `electronics`): jangan menganggap keduanya satu model.
- Loss validasi di `results.csv` lama bernilai 0 sepanjang training (perilaku `train_dual.py`, menurut ingatan, belum diverifikasi): jangan memakainya sebagai sinyal overfitting. Pakai set val/test terpisah.
- Auto-exposure bisa mengubah putih bak. Rekam dan jalankan dengan pengaturan exposure yang sama.
- Kamera webcam 640×480 MJPG: rekam data kosong pada resolusi dan pipeline yang sama dengan deployment.
