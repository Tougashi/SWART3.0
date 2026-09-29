# 🍓 SWART - Raspberry Pi (IoT Controller)

Folder ini berisi kode yang berjalan di **Raspberry Pi** untuk mengontrol hardware SWART (Smart Waste Sorting).

## 📂 Struktur File

| File | Deskripsi |
|------|-----------|
| `detectv2.py` | **Script utama** — deteksi YOLOv9 + kontrol servo/LED (dipakai di `waste.sh`) |
| `detect9.py` | Versi terbaru — deteksi ultralytics YOLO + servo + LED + sensor ultrasonik |
| `detectv1.py` | Versi awal — campuran import yolov7/v9, perlu cleanup |

## 🔧 Hardware yang Digunakan

### Servo Motor
- **Servo 360°** (GPIO 21) — putar platform kiri/kanan
- **Servo 180°** (GPIO 26) — posisi A/B/C untuk mengarahkan sampah

### LED Indikator (12 buah)
| LED | Warna | GPIO | Keterangan |
|-----|-------|------|-----------|
| REDAL-REDCR | Merah | 17,27,22,5,6,13 | Indikator tempat penuh |
| WHITEAL-WHITECR | Putih | 2,3,4,10,9,11 | Indikator kategori aktif |

### Sensor Ultrasonik (6 buah, di detect9.py)
- AL, BL, CL (kiri) — trigger/echo pairs
- AR, BR, CR (kanan) — trigger/echo pairs

## 🚀 Cara Menjalankan

```bash
# Dari Raspberry Pi:
sudo pigpiod
conda activate yolov9
python detectv2.py --weights /path/to/best.onnx --imgsz=224 --source 0
```

## 📦 Dependencies Khusus RPi

```
RPi.GPIO
gpiozero
pigpio
torch / ultralytics
opencv-python
```
