# ⚖️ SWART - Model Weights

Semua file model weights yang telah di-training.

## 📂 File Weights

| File | Ukuran | Format | Keterangan |
|------|--------|--------|-----------|
| `best.pt` | ~20 MB | PyTorch | Model terbaik (versi 1) |
| `best2.pt` | ~20 MB | PyTorch | Model terbaik (versi 2) |
| `best.onnx` | ~38 MB | ONNX | Export ONNX dari best.pt |
| `best2.onnx` | ~38 MB | ONNX | Export ONNX dari best2.pt |
| `SWART.onnx` | ~38 MB | ONNX | Model SWART final |
| `wasteaiupdate.pt` | ~12 MB | PyTorch | Model update |

## 💡 Penggunaan

```bash
# PyTorch
python detect.py --weights weights/best.pt --source 0

# ONNX (untuk Raspberry Pi - lebih ringan)
python detectv2.py --weights weights/best.onnx --imgsz=224 --source 0
```

## 📝 Catatan
- File `.onnx` direkomendasikan untuk Raspberry Pi (inference tanpa PyTorch)
- File `.pt` untuk training lanjutan atau validasi
