"""
SWART - Rekam dataset BAK KOSONG (gambar latar / negatif) langsung dari kamera di alat
=======================================================================================
Gambar tanpa objek TIDAK perlu dilabeli. Skrip ini menyimpan frame mentah (tanpa gambar kotak/teks) dengan
resolusi & pipeline yang SAMA dengan saat alat dipakai, satu folder per sesi, lengkap dengan file label kosong
(format YOLO: file .txt kosong = "tidak ada objek").

Pemakaian (di Raspberry Pi, bak benar-benar kosong, lalu jalankan):
    python capture_dataset.py --source 0 --tag siang_lampu_nyala --count 150 --interval 1.5
    python capture_dataset.py --source 0 --tag redup --count 150
    python capture_dataset.py --source 0 --tag servo_putar --count 150 --interval 0.7   # putar servo/platform sambil merekam
    python capture_dataset.py --source 0 --tag tangan_lewat --count 150 --interval 0.7  # tangan masuk-keluar area scan, TANPA benda

Cari false positive model yang sekarang sambil merekam (hard negative): tambahkan --weights model.onnx
    python capture_dataset.py --source 0 --tag cek_model --weights weights/best2_main.onnx --count 200
    -> frame yang membuat model mendeteksi sesuatu (padahal bak kosong) juga disalin ke dataset_empty/hardneg/

PENTING:
  * Satu sesi = satu kondisi (cahaya, jam, posisi servo, suhu kamera). Pemisahan train/val/test dilakukan PER SESI.
  * Pakai opsi kamera yang sama dengan deployment (mis. --lock-exposure jika di detect_pi_fast.py juga dipakai).
  * Jangan rekam bak kosong sebelum keputusan hardware final (warna permukaan, lampu, posisi kamera). Ubah hardware
    = rekam ulang.
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from swart_core import CameraStream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='0')
    ap.add_argument('--tag', required=True, help='nama kondisi sesi, mis. siang_lampu_nyala (tanpa spasi)')
    ap.add_argument('--out', default='dataset_empty')
    ap.add_argument('--count', type=int, default=150, help='jumlah frame per sesi')
    ap.add_argument('--interval', type=float, default=1.5, help='detik antar frame (acak +-30%% agar tidak seragam)')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--delay', type=float, default=5.0, help='detik menunggu sebelum mulai (agar tangan menjauh)')
    ap.add_argument('--lock-exposure', action='store_true')
    ap.add_argument('--quality', type=int, default=95, help='kualitas JPEG')
    ap.add_argument('--weights', default=None, help='(opsional) ONNX untuk mencatat false positive')
    ap.add_argument('--conf', type=float, default=0.15, help='confidence minimum untuk dianggap false positive')
    a = ap.parse_args()

    tag = ''.join(c if c.isalnum() or c in '-_' else '_' for c in a.tag)
    src = int(a.source) if a.source.isnumeric() else a.source
    backend = cv2.CAP_V4L2 if (sys.platform.startswith('linux') and isinstance(src, int)) else None
    cam = CameraStream(src, a.width, a.height, 30, backend)
    if not cam.opened:
        sys.exit('[ERROR] Kamera gagal dibuka')
    cam.start()
    if a.lock_exposure:
        cam.lock_exposure()

    det = None
    if a.weights:
        from swart_core import OnnxDetector
        det = OnnxDetector(a.weights, 320, a.conf, 0.45, None, verbose=False)
        det.warmup()

    session = f"session_{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    root = Path(a.out)
    img_dir, lab_dir = root / session / 'images', root / session / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    hard_dir = root / 'hardneg'
    if det:
        hard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Sesi {session} -> {img_dir}\n[INFO] Pastikan bak KOSONG. Mulai dalam {a.delay:.0f} detik...")
    time.sleep(a.delay)

    saved, fp, last_id, t_next = 0, 0, -1, time.time()
    w = h = None
    try:
        while saved < a.count:
            fid, frame = cam.read()
            if frame is None or fid == last_id or time.time() < t_next:
                time.sleep(0.01)
                continue
            last_id = fid
            t_next = time.time() + a.interval * random.uniform(0.7, 1.3)
            h, w = frame.shape[:2]
            name = f"{tag}_{saved:04d}"
            path = img_dir / f"{name}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            (lab_dir / f"{name}.txt").write_text('')          # label kosong = latar belakang
            saved += 1
            if det:
                hits = det.detect(frame, conf=a.conf)
                if hits:
                    fp += 1
                    cv2.imwrite(str(hard_dir / f"{session}_{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                    print(f"  [FP] {name}: " + ', '.join(f"{det.names[c]} {s:.2f}" for *_, s, c in hits))
            if saved % 25 == 0:
                print(f"[INFO] {saved}/{a.count}" + (f" | false positive model: {fp}" if det else ''))
    except KeyboardInterrupt:
        print('\n[INFO] Dihentikan; menyimpan metadata sesi...')
    finally:
        cam.release()
        meta = {'session': session, 'tag': tag, 'frames': saved, 'resolution': [w, h],
                'interval_s': a.interval, 'lock_exposure': a.lock_exposure,
                'started': datetime.now().isoformat(timespec='seconds'),
                'model_false_positives': fp if det else None, 'model': a.weights}
        (root / session / 'meta.json').write_text(json.dumps(meta, indent=2))
        print(f"[OK] {saved} frame tersimpan di {root / session}")


if __name__ == '__main__':
    main()
