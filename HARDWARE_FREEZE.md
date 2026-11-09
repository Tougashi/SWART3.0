# SWART — Keputusan Hardware (HARDWARE_FREEZE.md)

> **Catatan:** Dokumen ini dibuat untuk mencatat keputusan hardware yang mempengaruhi training dan deployment.
> Karena waktu mepet, keputusan dibuat berdasarkan kondisi yang ada saat ini.

**Tanggal:** 2026-11-07

---

## Keputusan

| Keputusan | Pilihan | Catatan |
|-----------|---------|---------|
| Warna permukaan bak | **Tetap putih** | Perubahan ke non-putih ideal tapi butuh rekam ulang. Masalah false positive ditangani via model + software filter. |
| Pencahayaan | **Kondisi yang ada** | Tidak ada lampu tambahan dipasang. Model dilatih dengan gambar sintetis multi-kondisi. |
| Posisi kamera & area scan | **Tetap seperti sekarang** | Kamera di atas bak, 640×480 MJPG, letterbox ke 320×320. |
| Exposure | **Auto exposure** | Kamera default. Jika bisa dikunci nanti, rekalibrasi direkomendasikan. |

## Implikasi

- Gambar bak kosong sintetis dibuat mensimulasikan permukaan putih dengan variasi cahaya
- Model dilatih dengan background images (label kosong) agar belajar "putih polos = tidak ada objek"
- Filter kalibrasi (`calibrate.py`) tetap dipertahankan sebagai pengaman kedua
- Jika hardware berubah di masa depan, perlu rekam ulang data bak kosong dan rekalibrasi

## Hal yang Belum Terverifikasi

- [ ] Efek auto-exposure vs locked exposure pada false positive rate
- [ ] Apakah noda/debu pada bak putih seiring waktu mempengaruhi deteksi
- [ ] Performa di kondisi cahaya ekstrem (sangat terang / sangat redup)

