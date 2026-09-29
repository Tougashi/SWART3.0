"""
SWART 3.0 - Test YOLOv9 dengan Iriun Webcam (versi cepat, ONNX Runtime)
=======================================================================
Beda utama dari test_yolov9_iriun.py:
  * Inference di THREAD SENDIRI -> video/HUD tetap mulus (dulu setiap frame inference membekukan tampilan;
    "frame skip" hanya membuat lag-nya berkedip-kedip).
  * Pakai cabang UTAMA YOLOv9 + model yang sudah dipangkas (prune_onnx.py) -> ~22-40% lebih cepat.
  * Ukuran input dibaca dari model. (Dulu: ONNX dinamis diam-diam jatuh ke 640.)
  * Preprocess/NMS dipercepat, HUD tanpa menyalin seluruh frame, statistik hanya dihitung per inference.
  * Tanpa torch -> start cepat. Untuk .pt: export dulu ke ONNX lalu jalankan prune_onnx.py.

Pemakaian:
    python test_yolov9_iriun_fast.py --weights weights/best2_main.onnx --source 1
    python test_yolov9_iriun_fast.py --weights weights/best_main.onnx --source 1     # input 224 (lebih ringan)
    python test_yolov9_iriun_fast.py --source video.mp4 --weights weights/best2_main.onnx

Kontrol: q/ESC keluar | s screenshot | p pause | r reset statistik | +/- conf | g mode hemat on/off
          c toggle color filter | v toggle vote overlay
"""
import argparse
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2

from swart_core import CameraStream, ColorVerifier, InferenceWorker, OnnxDetector

COLORS = {'paper': (0, 200, 255), 'metal': (180, 180, 180), 'electronic': (255, 100, 0),
          'electronics': (255, 100, 0), 'plastic': (0, 255, 100), 'glass': (255, 200, 100),
          'organic': (50, 200, 50)}


def find_camera(preferred, backend, max_scan=5):
    print(f"[INFO] Mencari kamera (preferred: {preferred})...")
    for idx in [preferred] + [i for i in range(max_scan) if i != preferred]:
        cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                print(f"[OK] Kamera di index {idx}")
                return idx
    return None


def draw_box(img, box, label, conf, col):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ly = max(y1 - th - 8, 0)
    cv2.rectangle(img, (x1, ly), (x1 + tw + 8, ly + th + 8), col, -1)
    cv2.putText(img, text, (x1 + 4, ly + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def darken(img, x1, y1, x2, y2, alpha=0.55):
    """Panel gelap semi-transparan hanya pada ROI kecil (bukan menyalin seluruh frame)."""
    roi = img[y1:y2, x1:x2]
    if roi.size:
        cv2.addWeighted(roi, 1 - alpha, roi * 0, alpha, 0, roi)


def norm_name(name):
    """'electronics' dan 'electronic' dianggap sama."""
    return 'electronic' if name.startswith('electronic') else name


def frame_label(dets, names):
    """Tepat 1 jenis kelas di frame -> label; kosong / >1 jenis berbeda -> None."""
    kinds = {norm_name(names.get(d[5], str(d[5]))) for d in dets}
    return kinds.pop() if len(kinds) == 1 else None


class Voter:
    """Temporal voting: butuh N dari M frame terakhir setuju sebelum label dianggap stabil."""
    def __init__(self, window=5, need=3):
        self.buf, self.need = deque(maxlen=window), need

    def push(self, label):
        self.buf.append(label)

    def decision(self):
        c = Counter(x for x in self.buf if x)
        if not c:
            return None
        label, n = c.most_common(1)[0]
        return label if n >= self.need else None

    def clear(self):
        self.buf.clear()


def draw_hud(img, disp_fps, w_fps, ms, conf, stats, mode, paused, size,
             color_on=False, vote_label=None):
    h, w = img.shape[:2]
    darken(img, 0, 0, 280, 142)
    fcol = (0, 255, 0) if w_fps >= 15 else (0, 200, 255) if w_fps >= 8 else (0, 0, 255)
    cv2.putText(img, f"Infer: {w_fps:.1f}/s  ({ms:.0f} ms)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fcol, 2)
    cv2.putText(img, f"Video: {disp_fps:.0f} FPS | input {size}px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"Conf: {conf:.2f} | Mode: {mode}", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cf_col = (0, 255, 0) if color_on else (0, 0, 200)
    cv2.putText(img, f"ColorFilter: {'ON' if color_on else 'OFF'}", (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cf_col, 1)
    cv2.putText(img, f"Deteksi: {stats['with_det']}/{stats['total']} inference", (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    if vote_label:
        cv2.putText(img, f"Vote: {vote_label}", (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLORS.get(vote_label, (220, 220, 220)), 2)
    if stats['counts']:
        ph = (len(stats['counts']) + 1) * 24 + 8
        darken(img, w - 200, 0, w, ph)
        cv2.putText(img, "TERDETEKSI", (w - 190, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, (n, c) in enumerate(sorted(stats['counts'].items(), key=lambda x: -x[1])):
            cv2.putText(img, f"{n}: {c}", (w - 190, 44 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COLORS.get(n, (220, 220, 220)), 1)
    if paused:
        cv2.putText(img, "PAUSED", (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/best2_main.onnx')
    ap.add_argument('--source', default='1', help='index kamera atau path video')
    ap.add_argument('--imgsz', type=int, default=320, help='hanya dipakai jika ONNX berukuran dinamis')
    ap.add_argument('--conf-thres', type=float, default=0.25)
    ap.add_argument('--iou-thres', type=float, default=0.45)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--gate', action='store_true', help='aktifkan mode hemat (inference turun saat scene diam)')
    ap.add_argument('--no-color-filter', action='store_true', help='matikan color verifier HSV')
    ap.add_argument('--color-verbose', action='store_true', help='cetak log tiap kali color filter override/drop')
    ap.add_argument('--vote-window', type=int, default=5, help='jumlah frame terakhir untuk voting')
    ap.add_argument('--vote-min', type=int, default=3, help='minimal suara kelas sama untuk vote stabil')
    a = ap.parse_args()

    wpath = Path(a.weights)
    if not wpath.exists():
        alt = Path(__file__).parent / 'weights' / wpath.name
        wpath = alt if alt.exists() else wpath
    if wpath.suffix != '.onnx' or not wpath.exists():
        sys.exit(f"[ERROR] Butuh file .onnx yang ada: {a.weights}")

    print("=" * 60 + "\n  SWART 3.0 - YOLOv9 Detection (FAST)\n" + "=" * 60)
    det = OnnxDetector(wpath, a.imgsz, a.conf_thres, a.iou_thres, a.threads)
    det.warmup()

    color_filter = ColorVerifier(det.names, enabled=not a.no_color_filter,
                                 verbose=a.color_verbose)
    voter = Voter(a.vote_window, a.vote_min)
    color_on = color_filter.enabled
    print(f"[INFO] Color filter: {'ON' if color_on else 'OFF'} | "
          f"Vote: {a.vote_min}/{a.vote_window}")

    backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else None
    if a.source.isnumeric():
        idx = find_camera(int(a.source), backend)
        if idx is None:
            sys.exit("[ERROR] Tidak ada kamera. Pastikan Iriun Webcam aktif.")
        cam = CameraStream(idx, a.width, a.height, 30, backend)
    else:
        cam = CameraStream(a.source, a.width, a.height, 30, None, mjpg=False)
    if not cam.opened:
        sys.exit("[ERROR] Kamera gagal dibuka")
    cam.start()

    worker = InferenceWorker(cam, det, idle_interval=0.5 if a.gate else 0.0)
    worker.start()

    win = "SWART 3.0 - YOLOv9 (FAST)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    stats = {'counts': defaultdict(int), 'total': 0, 'with_det': 0}
    shots = Path(__file__).parent / 'screenshots'
    seq, last_fid, paused, gate_on = 0, -1, False, a.gate
    dets, ms = [], 0.0
    t_prev, disp_fps = time.perf_counter(), 0.0
    frame = None
    vote_label = None

    print("[START] Tekan 'q' untuk keluar. 'c' toggle color filter, '+'/'-' conf.\n")
    try:
        while True:
            if not paused:
                _, frame = cam.read()
                seq_new, d, ms_new, fid = worker.wait_new(seq, timeout=0.0)
                if seq_new != seq:                      # ada hasil inference baru -> update statistik
                    seq, ms = seq_new, ms_new
                    # color filter: verifikasi/koreksi label
                    if color_on and frame is not None:
                        dets = color_filter.verify(frame, d)
                    else:
                        dets = d
                    stats['total'] += 1
                    if dets:
                        stats['with_det'] += 1
                        for x in dets:
                            stats['counts'][det.names.get(x[5], str(x[5]))] += 1
                    # temporal voting
                    voter.push(frame_label(dets, det.names))
                    vote_label = voter.decision()
            if frame is None:
                time.sleep(0.01)
                continue

            vis = frame.copy()                          # jangan gambar di frame bersama (dibaca thread inference)
            for x1, y1, x2, y2, c, k in dets:
                n = det.names.get(k, str(k))
                draw_box(vis, (x1, y1, x2, y2), n, c, COLORS.get(n, (200, 200, 200)))

            now = time.perf_counter()
            disp_fps = 0.9 * disp_fps + 0.1 / max(now - t_prev, 1e-6) if disp_fps else 30.0
            t_prev = now
            draw_hud(vis, disp_fps, worker.fps, worker.ms_avg or ms, det.conf, stats,
                     ('hemat' if gate_on else 'penuh') + ('/aktif' if worker.active else '/diam' if gate_on else ''),
                     paused, det.w, color_on=color_on, vote_label=vote_label)
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                shots.mkdir(exist_ok=True)
                fp = shots / f"det_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(fp), vis)
                print(f"[SAVE] {fp}")
            elif key == ord('p'):
                paused = not paused
                worker.paused = paused
            elif key == ord('r'):
                stats.update(total=0, with_det=0)
                stats['counts'].clear()
            elif key in (ord('+'), ord('=')):
                det.conf = min(det.conf + 0.05, 0.60)   # di atas ~0.5 F1 model ini anjlok
            elif key == ord('-'):
                det.conf = max(det.conf - 0.05, 0.05)
            elif key == ord('g'):
                gate_on = not gate_on
                worker.idle_interval = 0.5 if gate_on else 0.0
            elif key == ord('c'):
                color_on = not color_on
                color_filter.enabled = color_on
                voter.clear()
                print(f"[INFO] Color filter: {'ON' if color_on else 'OFF'}")
            elif key == ord('v'):
                voter.clear()
                print("[INFO] Vote di-reset")
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        cam.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60 + "\n  RINGKASAN\n" + "=" * 60)
        print(f"  Model            : {wpath.name} ({det.w}x{det.h})")
        print(f"  Rata2 inference  : {worker.ms_avg:.1f} ms  (~{worker.fps:.1f}/s)")
        print(f"  Color filter     : {'ON' if color_on else 'OFF'}")
        print(f"  Vote             : {a.vote_min}/{a.vote_window}")
        print(f"  Inference total  : {stats['total']} | dgn deteksi: {stats['with_det']}")
        for n, c in sorted(stats['counts'].items(), key=lambda x: -x[1]):
            print(f"    {n:12s}: {c}")


if __name__ == '__main__':
    main()
