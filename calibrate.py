"""
SWART - Kalibrasi referensi bak kosong
=======================================
Rekam frame bak kosong, hitung referensi statistik (mean + std foreground mask),
dan simpan ke file .npz. Dipakai oleh detect_pi_fast.py dan eval_empty_fp.py untuk
mengurangi false positive pada bak kosong.

Pemakaian:
    python calibrate.py --source 0 --out calib/empty_ref.npz --frames 60
    python calibrate.py --source 0 --out calib/empty_ref.npz --frames 60 --lock-exposure
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from swart_core import CameraStream


class EmptyReference:
    """Referensi bak kosong: simpan mean + std warna untuk mendeteksi 'foreground'."""

    def __init__(self, mean, std, roi_mask=None):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.roi_mask = roi_mask  # opsional: hanya area scan yang dievaluasi

    @classmethod
    def from_frames(cls, frames, roi_mask=None):
        """Buat referensi dari list frame BGR."""
        stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0).clip(min=5.0)  # minimal 5 untuk toleransi noise
        return cls(mean, std, roi_mask)

    def foreground(self, frame, z_thr=3.0):
        """Hitung foreground mask: piksel yang jauh dari referensi kosong."""
        diff = np.abs(frame.astype(np.float32) - self.mean) / self.std
        fg = (diff.max(axis=2) > z_thr).astype(np.uint8) * 255
        if self.roi_mask is not None:
            fg = cv2.bitwise_and(fg, self.roi_mask)
        return fg, (fg > 0).sum() / max(fg.size, 1)

    def center_in_roi(self, box, shape):
        """Cek apakah center bounding box ada di dalam ROI."""
        if self.roi_mask is None:
            return True
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        h, w = shape[:2]
        cy, cx = min(cy, h - 1), min(cx, w - 1)
        return bool(self.roi_mask[cy, cx] > 0)

    def box_change(self, fg_mask, box, shape, margin=5):
        """Fraksi piksel foreground di dalam bounding box."""
        x1, y1, x2, y2 = box
        h, w = shape[:2]
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
        region = fg_mask[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        return float((region > 0).sum()) / region.size

    def save(self, path):
        data = {'mean': self.mean, 'std': self.std}
        if self.roi_mask is not None:
            data['roi_mask'] = self.roi_mask
        np.savez_compressed(str(path), **data)
        print(f"[OK] Referensi disimpan: {path}")

    @classmethod
    def load(cls, path):
        d = np.load(str(path))
        roi = d['roi_mask'] if 'roi_mask' in d else None
        return cls(d['mean'], d['std'], roi)


def main():
    ap = argparse.ArgumentParser(description='Kalibrasi referensi bak kosong')
    ap.add_argument('--source', default='0', help='index kamera atau path video')
    ap.add_argument('--out', default='calib/empty_ref.npz', help='path output .npz')
    ap.add_argument('--frames', type=int, default=60, help='jumlah frame untuk referensi')
    ap.add_argument('--interval', type=float, default=0.5, help='detik antar frame')
    ap.add_argument('--delay', type=float, default=3.0, help='detik menunggu sebelum mulai')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--lock-exposure', action='store_true')
    a = ap.parse_args()

    src = int(a.source) if a.source.isnumeric() else a.source
    backend = cv2.CAP_V4L2 if (sys.platform.startswith('linux') and isinstance(src, int)) else None
    cam = CameraStream(src, a.width, a.height, 30, backend)
    if not cam.opened:
        sys.exit('[ERROR] Kamera gagal dibuka')
    cam.start()

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Pastikan bak KOSONG. Mulai dalam {a.delay:.0f} detik...")
    time.sleep(a.delay)

    frames = []
    last_id = -1
    t_next = time.time()
    try:
        while len(frames) < a.frames:
            fid, frame = cam.read()
            if frame is None or fid == last_id or time.time() < t_next:
                time.sleep(0.01)
                continue
            last_id = fid
            t_next = time.time() + a.interval
            frames.append(frame.copy())
            if len(frames) % 10 == 0:
                print(f"  ... {len(frames)}/{a.frames}")
    except KeyboardInterrupt:
        print(f"\n[INFO] Dihentikan, menggunakan {len(frames)} frame")
    finally:
        cam.release()

    if len(frames) < 5:
        sys.exit(f"[ERROR] Hanya {len(frames)} frame, terlalu sedikit (minimal 5)")

    ref = EmptyReference.from_frames(frames)
    ref.save(out_path)
    print(f"[INFO] Referensi bak kosong: {len(frames)} frame, resolusi {frames[0].shape[1]}x{frames[0].shape[0]}")


if __name__ == '__main__':
    main()

