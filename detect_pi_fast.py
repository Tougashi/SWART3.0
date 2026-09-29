"""
SWART - Deteksi + kontrol servo/LED di Raspberry Pi (versi optimal, pengganti detectv2.py)
=========================================================================================
Pemakaian:
    sudo pigpiod                       # sekali per boot (kalau belum jalan)
    python detect_pi_fast.py --weights weights/best2_main.onnx --source 0
    python detect_pi_fast.py --weights weights/best_main.onnx --imgsz 224 --source 0   # lebih ringan
    python detect_pi_fast.py --dry-run --show                                          # tes tanpa hardware

Yang berubah dari detectv2.py (sumber lag di Raspberry):
  1. mulai(s) dulu dipanggil di SETIAP frame -> minimal ~4.7 detik time.sleep per frame, walau kosong.
     Sekarang servo/LED jalan di thread sendiri & hanya dipicu saat objek stabil terdeteksi.
  2. Tidak lagi menyimpan video ke SD card & tidak wajib membuka jendela (headless default).
  3. Tidak butuh torch/repo yolov9: cukup onnxruntime + opencv (start lebih cepat, RAM lebih kecil).
  4. Memakai cabang utama YOLOv9 (bukan auxiliary) dan model yang sudah dipangkas (prune_onnx.py).
  5. Voting antar-frame (bukan 1 frame langsung memicu servo) -> lebih tahan salah deteksi.
  6. Bug logika lama: kelas ganda ("2 plastics,") dan nama 'electronics' vs 'electronic' membuat frame
     dianggap kosong. Sekarang keduanya ditangani.

Urutan gerak, pin, dan timing servo/LED SAMA dengan detectv2.py.
"""
import argparse
import sys
import threading
import time
from collections import Counter, deque

import cv2

from swart_core import CameraStream, ColorVerifier, InferenceWorker, OnnxDetector

# ----------------------------------------------------------------------------
# Hardware (sama dengan detectv2.py)
# ----------------------------------------------------------------------------
PIN_SERVO_360, PIN_SERVO_180 = 21, 26
LED_PINS = {'REDAL': 17, 'REDBL': 27, 'REDCL': 22, 'REDAR': 5, 'REDBR': 6, 'REDCR': 13,
            'WHITEAL': 2, 'WHITEBL': 3, 'WHITECL': 4, 'WHITEAR': 10, 'WHITEBR': 9, 'WHITECR': 11}

# kelas -> (LED putih, posisi servo 180, arah servo 360)
ROUTES = {
    'organic':    ('WHITEAL', 'a', 'kanan'),
    'metal':      ('WHITEBL', 'b', 'kanan'),
    'electronic': ('WHITECL', 'c', 'kanan'),
    'paper':      ('WHITEAR', 'a', 'kiri'),
    'glass':      ('WHITEBR', 'b', 'kiri'),
    'plastic':    ('WHITECR', 'c', 'kiri'),
}
S180 = {'a': 1, 'b': 0, 'c': -1}
S360 = {'kiri': 1, 'kanan': -1, 'stop': 0}

# Timing asli detectv2.py
T_BEFORE_ROTATE = 0.5    # jeda setelah servo180 bergerak, sebelum platform berputar
T_ROTATE = 2.11          # durasi servo360 berputar
T_AFTER_ROTATE = 0.5     # tambahan putar setelah LED mati (di kode lama: sleep 0.5 di mulai())
T_HOME_SETTLE = 2.11     # jeda setelah servo360 dihentikan (kembali ke home)


def norm(name):
    """'electronics' (best.onnx/best2.onnx) dan 'electronic' (SWART.onnx) dianggap sama."""
    return 'electronic' if name.startswith('electronic') else name


class Sorter:
    """Menjalankan urutan servo/LED di thread terpisah supaya inference & kamera tidak terblok."""

    def __init__(self, dry_run=False):
        self.dry = dry_run
        self._busy = threading.Event()
        self.at_home = False
        if not dry_run:
            import RPi.GPIO as GPIO
            from gpiozero import Servo
            from gpiozero.pins.pigpio import PiGPIOFactory
            self.GPIO = GPIO
            factory = PiGPIOFactory()
            c, cb = 0.1, 0.55
            self.s180 = Servo(PIN_SERVO_180, min_pulse_width=(1.0 - c - 0.2) / 1000,
                              max_pulse_width=(2.0 + c) / 1000, pin_factory=factory)
            self.s360 = Servo(PIN_SERVO_360, min_pulse_width=(1.0 - cb) / 1000,
                              max_pulse_width=(2.0 + cb) / 1000, pin_factory=factory)
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            for pin in LED_PINS.values():
                GPIO.setup(pin, GPIO.OUT)
        self._home()

    @property
    def busy(self):
        return self._busy.is_set()

    def _set180(self, pos):
        if not self.dry:
            self.s180.value = S180[pos]

    def _set360(self, d):
        if not self.dry:
            self.s360.value = S360[d]

    def _led(self, key, on):
        if not self.dry:
            self.GPIO.output(LED_PINS[key], self.GPIO.HIGH if on else self.GPIO.LOW)

    def _home(self):
        self._set180('b')
        self._set360('stop')
        if not self.at_home:            # jeda settle hanya saat benar-benar perlu (startup / setelah gerak)
            time.sleep(T_HOME_SETTLE)
        self.at_home = True

    def trigger(self, label):
        if label not in ROUTES or self._busy.is_set():
            return False
        self._busy.set()
        threading.Thread(target=self._run, args=(label,), daemon=True).start()
        return True

    def _run(self, label):
        led, pos180, dir360 = ROUTES[label]
        try:
            self.at_home = False
            self._led(led, True)
            self._set180(pos180)
            time.sleep(T_BEFORE_ROTATE)
            self._set360(dir360)
            time.sleep(T_ROTATE)
            self._led(led, False)
            time.sleep(T_AFTER_ROTATE)
            self._home()
        finally:
            self._busy.clear()

    def cleanup(self):
        try:
            self._set360('stop')
            self._set180('b')
            if not self.dry:
                for k in LED_PINS:
                    self._led(k, False)
                self.GPIO.cleanup()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Voting antar-frame
# ----------------------------------------------------------------------------
def frame_label(dets, names):
    """Tepat 1 jenis kelas di frame -> label; kosong / >1 jenis berbeda -> None (aturan 'kosong' lama)."""
    kinds = {norm(names.get(d[5], str(d[5]))) for d in dets}
    return kinds.pop() if len(kinds) == 1 else None


class Voter:
    def __init__(self, window, need):
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


COLORS = {'paper': (0, 200, 255), 'metal': (180, 180, 180), 'electronic': (255, 100, 0),
          'plastic': (0, 255, 100), 'glass': (255, 200, 100), 'organic': (50, 200, 50)}


def draw(frame, dets, names, info):
    for x1, y1, x2, y2, conf, c in dets:
        name = norm(names.get(c, str(c)))
        col = COLORS.get(name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.putText(frame, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser(description='SWART Raspberry Pi - deteksi cepat + servo')
    ap.add_argument('--weights', default='weights/best2_main.onnx')
    ap.add_argument('--source', default='0', help='index kamera atau path video')
    ap.add_argument('--imgsz', type=int, default=320, help='hanya dipakai jika ONNX berukuran dinamis')
    ap.add_argument('--conf', type=float, default=0.25, help='F1 terbaik model ini di conf ~0.22')
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--threads', type=int, default=None, help='thread ONNX (default: jumlah core - 1)')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--vote-window', type=int, default=5, help='jumlah hasil inference terakhir yang dihitung')
    ap.add_argument('--vote-min', type=int, default=3, help='minimal suara kelas yang sama untuk memicu servo')
    ap.add_argument('--cooldown', type=float, default=0.5, help='jeda (detik) setelah servo selesai')
    ap.add_argument('--idle-interval', type=float, default=0.5,
                    help='saat diam & kosong, inference tiap N detik (0 = selalu full speed)')
    ap.add_argument('--show', action='store_true', help='tampilkan jendela (butuh DISPLAY)')
    ap.add_argument('--dry-run', action='store_true', help='tanpa GPIO/servo (untuk tes di PC)')
    ap.add_argument('--log-every', type=float, default=5.0)
    ap.add_argument('--no-color-filter', action='store_true', help='matikan color verifier HSV')
    ap.add_argument('--color-verbose', action='store_true', help='log tiap override/drop color filter')
    a = ap.parse_args()

    det = OnnxDetector(a.weights, a.imgsz, a.conf, a.iou, a.threads)
    det.warmup()

    color_filter = ColorVerifier(det.names, enabled=not a.no_color_filter,
                                 verbose=a.color_verbose)

    src = int(a.source) if a.source.isnumeric() else a.source
    backend = cv2.CAP_V4L2 if (sys.platform.startswith('linux') and isinstance(src, int)) else None
    cam = CameraStream(src, a.width, a.height, a.fps, backend)
    if not cam.opened:
        sys.exit('[ERROR] Kamera gagal dibuka')
    cam.start()

    sorter = Sorter(a.dry_run)
    worker = InferenceWorker(cam, det, idle_interval=a.idle_interval)
    worker.start()
    voter = Voter(a.vote_window, a.vote_min)

    print(f"[START] siap | vote {a.vote_min}/{a.vote_window} | idle-interval {a.idle_interval}s | "
          f"color-filter {'ON' if color_filter.enabled else 'OFF'} | "
          f"{'DRY-RUN' if a.dry_run else 'HARDWARE'}")
    seq, cooldown_until, last_log, n_trig = 0, 0.0, time.time(), 0
    last_frame_for_color = None
    try:
        while True:
            seq, dets, ms, fid = worker.wait_new(seq, timeout=0.03 if a.show else 0.25)

            # simpan frame terakhir untuk color filter
            _, cur_frame = cam.read()
            if cur_frame is not None:
                last_frame_for_color = cur_frame

            # color filter: verifikasi/koreksi label
            if color_filter.enabled and last_frame_for_color is not None and dets:
                dets = color_filter.verify(last_frame_for_color, dets)

            if sorter.busy:
                voter.clear()
                cooldown_until = time.time() + a.cooldown
            elif time.time() >= cooldown_until and fid:
                voter.push(frame_label(dets, det.names))
                label = voter.decision()
                if label and sorter.trigger(label):
                    n_trig += 1
                    print(f"[SORT] {label} (trigger #{n_trig})")
                    voter.clear()

            if a.show:
                _, frame = cam.read()
                if frame is not None:
                    vis = frame.copy()
                    draw(vis, dets, det.names,
                         f"{worker.fps:.1f} FPS | {worker.ms_avg:.0f} ms | {'AKTIF' if worker.active else 'HEMAT'}"
                         f"{' | SORTING' if sorter.busy else ''}")
                    cv2.imshow('SWART', vis)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break

            if time.time() - last_log >= a.log_every:
                last_log = time.time()
                print(f"[STAT] inference {worker.fps:.1f}/s | {worker.ms_avg:.0f} ms | "
                      f"{'aktif' if worker.active else 'hemat'} | sortir {n_trig}x")
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C")
    finally:
        worker.stop()
        cam.release()
        sorter.cleanup()
        cv2.destroyAllWindows()
        print("[INFO] Selesai.")


if __name__ == '__main__':
    main()
