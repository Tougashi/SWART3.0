"""
SWART - modul inti (dipakai oleh detect_pi_fast.py dan test_yolov9_iriun_fast.py)
==================================================================================
Hanya butuh: opencv-python, numpy, onnxruntime  (TIDAK butuh torch / repo yolov9).

Perbaikan utama dibanding skrip lama:
  1. Pakai output cabang UTAMA YOLOv9 (bukan cabang auxiliary) -> sama dengan hasil val_dual.py.
  2. Cabang auxiliary dibuang dari file ONNX (lihat prune_onnx.py) -> compute turun 22-40%.
  3. Kamera, inference, dan tampilan/aktuator berjalan di thread terpisah -> tidak saling blok.
  4. Preprocess/NMS dipercepat (cv2.dnn.blobFromImage, cv2.dnn.NMSBoxes, canvas letterbox dipakai ulang).
  5. Ukuran input dibaca dari model (tidak lagi jatuh ke 640 kalau ONNX dinamis).
  6. Adaptive rate: saat scene diam & tidak ada objek, inference turun ke ~2 Hz (hemat CPU/panas di Pi).
"""

import ast
import os
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

FALLBACK_NAMES = {0: 'electronics', 1: 'glass', 2: 'metal', 3: 'organic', 4: 'paper', 5: 'plastic'}


# ----------------------------------------------------------------------------
# Kamera: thread khusus, selalu menyimpan frame TERBARU (tidak ada antrean/backlog)
# ----------------------------------------------------------------------------
class CameraStream:
    def __init__(self, source=0, width=640, height=480, fps=30, backend=None, mjpg=True):
        self.cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
        if mjpg:  # MJPG jauh lebih ringan di USB (terutama Raspberry Pi) daripada YUYV mentah
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._is_file = isinstance(source, str) and not source.isnumeric()

        ok, frame = self.cap.read()
        if ok:
            self._frame, self._id = frame, 1
        self.opened = bool(ok)

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self._is_file:  # video file: ulang dari awal (untuk testing)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame  # array baru tiap read -> aman dibagi (jangan dimodifikasi in-place)
                self._id += 1
            if self._is_file:
                time.sleep(1 / 30)

    def read(self):
        """Return (frame_id, frame). frame adalah referensi bersama: .copy() dulu sebelum digambari."""
        with self._lock:
            return self._id, self._frame

    @property
    def resolution(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self):
        self._stop.set()
        time.sleep(0.1)
        self.cap.release()


# ----------------------------------------------------------------------------
# Detektor ONNX (YOLOv9 anchor-free, output [1, 4+nc, N])
# ----------------------------------------------------------------------------
class OnnxDetector:
    def __init__(self, path, imgsz=320, conf=0.25, iou=0.45, threads=None, max_det=50, verbose=True):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1                       # model ini berantai; inter>1 hanya rebutan CPU
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 4) - 1)  # sisakan 1 core
        self.sess = ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])

        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        if isinstance(h, int) and isinstance(w, int):
            self.h, self.w = h, w                          # ONNX statis: ukuran model yang berlaku
        else:
            self.h = self.w = int(imgsz)                   # ONNX dinamis: pakai --imgsz (bukan 640 diam-diam)

        outs = self.sess.get_outputs()
        self.n_outputs = len(outs)
        self.out_name = outs[-1].name                      # dual-head: output terakhir = cabang UTAMA
        if self.n_outputs > 1 and verbose:
            print("[WARN] Model masih memuat cabang auxiliary (lebih lambat ~25-40%). "
                  "Jalankan: python prune_onnx.py <model.onnx>")

        self.names = dict(FALLBACK_NAMES)
        meta = self.sess.get_modelmeta().custom_metadata_map or {}
        if 'names' in meta:
            try:
                self.names = ast.literal_eval(meta['names'])
            except Exception:
                pass

        self.conf, self.iou, self.max_det = conf, iou, max_det
        self._canvas = None
        self._canvas_key = None
        if verbose:
            print(f"[INFO] Model {os.path.basename(str(path))} | input {self.w}x{self.h} | "
                  f"threads {so.intra_op_num_threads} | kelas {list(self.names.values())}")

    # -- preprocess: letterbox ke canvas yang dipakai ulang + blobFromImage (C++) --
    def _prep(self, frame):
        fh, fw = frame.shape[:2]
        r = min(self.h / fh, self.w / fw)
        nw, nh = int(round(fw * r)), int(round(fh * r))
        dw, dh = (self.w - nw) / 2, (self.h - nh) / 2
        key = (fh, fw)
        if self._canvas_key != key:
            self._canvas = np.full((self.h, self.w, 3), 114, np.uint8)
            self._canvas_key = key
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        resized = frame if (nw, nh) == (fw, fh) else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self._canvas[top:top + nh, left:left + nw] = resized
        blob = cv2.dnn.blobFromImage(self._canvas, 1 / 255.0, (self.w, self.h), swapRB=True)
        return blob, r, dw, dh

    def detect(self, frame, conf=None):
        """Return list of (x1, y1, x2, y2, conf, cls_id) dalam koordinat frame asli."""
        conf_thr = self.conf if conf is None else conf
        blob, r, dw, dh = self._prep(frame)
        out = self.sess.run([self.out_name], {self.input_name: blob})[0]

        p = out[0]
        if p.shape[0] < p.shape[1]:
            p = p.T                                        # -> [N, 4+nc]
        scores = p[:, 4:]
        cls = scores.argmax(1)
        confs = scores[np.arange(len(scores)), cls]
        keep = confs > conf_thr
        if not keep.any():
            return []
        p, cls, confs = p[keep], cls[keep], confs[keep]
        if len(confs) > 300:                               # batasi kandidat sebelum NMS
            top = np.argpartition(-confs, 300)[:300]
            p, cls, confs = p[top], cls[top], confs[top]

        fh, fw = frame.shape[:2]
        x1 = np.clip((p[:, 0] - p[:, 2] / 2 - dw) / r, 0, fw)
        y1 = np.clip((p[:, 1] - p[:, 3] / 2 - dh) / r, 0, fh)
        x2 = np.clip((p[:, 0] + p[:, 2] / 2 - dw) / r, 0, fw)
        y2 = np.clip((p[:, 1] + p[:, 3] / 2 - dh) / r, 0, fh)

        off = cls.astype(np.float32) * 4096.0              # trik offset: NMS per-kelas
        rects = np.stack([x1 + off, y1 + off, x2 - x1, y2 - y1], 1).astype(np.float32)
        idx = cv2.dnn.NMSBoxes(rects.tolist(), confs.astype(np.float32).tolist(), conf_thr, self.iou)
        idx = np.array(idx).reshape(-1)[:self.max_det]
        return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), float(confs[i]), int(cls[i])) for i in idx]

    def warmup(self, n=3):
        dummy = np.full((480, 640, 3), 114, np.uint8)
        for _ in range(n):
            self.detect(dummy)


# ----------------------------------------------------------------------------
# Deteksi gerakan murah (64x48 grayscale) untuk adaptive rate
# ----------------------------------------------------------------------------
class MotionGate:
    def __init__(self, thr=2.0, alpha=0.05):
        self.thr, self.alpha, self.bg = thr, alpha, None

    def score(self, frame):
        g = cv2.cvtColor(cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.bg is None:
            self.bg = g
            return 0.0
        s = float(np.abs(g - self.bg).mean())
        self.bg += self.alpha * (g - self.bg)
        return s

    def moving(self, frame):
        return self.score(frame) > self.thr


# ----------------------------------------------------------------------------
# Color-based post-filter: verifikasi/koreksi label berdasar warna HSV
# ----------------------------------------------------------------------------
# Setiap kelas punya "profil" HSV yang dicocokkan dengan distribusi warna
# di dalam bounding box. Profil berisi daftar rentang HSV + bobot.
# Skor diukur sebagai persentase piksel yang masuk ke profil. Jika skor
# kelas asal terlalu rendah DAN ada kelas lain yang jauh lebih cocok,
# label bisa di-override. Jika semua skor rendah, deteksi bisa di-drop.

class ColorVerifier:
    """Post-processing filter: verifikasi prediksi kelas berdasar warna HSV."""

    # Profil HSV per kelas: list of (H_low, S_low, V_low, H_high, S_high, V_high, weight)
    # H: 0-179, S: 0-255, V: 0-255 (OpenCV convention)
    # Weight dipakai jika ada beberapa sub-profil (misal organic bisa hijau ATAU cokelat).
    PROFILES = {
        'paper': [
            # putih/krem: saturasi rendah, value tinggi
            (0,   0,  160, 179, 50,  255, 1.0),
            # kuning/cokelat muda (karton)
            (15,  20, 120, 35,  120, 255, 0.7),
        ],
        'organic': [
            # hijau (daun, sayuran)
            (30,  30,  30, 85,  255, 255, 1.0),
            # cokelat (tanah, kulit buah)
            (8,   30,  20, 25,  200, 180, 0.9),
            # kuning (buah masak)
            (20,  50,  80, 35,  255, 255, 0.6),
            # merah tua (buah busuk, tomat)
            (0,   50,  20, 10,  255, 180, 0.5),
            (170, 50,  20, 179, 255, 180, 0.5),
        ],
        'glass': [
            # transparan / putih reflektif: saturasi rendah, value sedang-tinggi
            (0,   0,  100, 179, 40,  255, 1.0),
            # hijau botol kaca (bir, dll)
            (35,  40,  30, 85,  255, 200, 0.7),
            # cokelat botol kaca
            (8,   30,  30, 25,  180, 180, 0.6),
        ],
        'metal': [
            # abu-abu metalik: saturasi sangat rendah, value sedang
            (0,   0,   40, 179, 35, 200, 1.0),
            # reflektif terang (aluminium)
            (0,   0,  180, 179, 25, 255, 0.8),
            # gelap (besi berkarat, cokelat tua)
            (5,   20,  20, 20, 120, 120, 0.5),
        ],
        'plastic': [
            # warna cerah / saturasi tinggi (botol warna-warni)
            (0,   80,  80, 179, 255, 255, 0.8),
            # putih (kantong plastik)
            (0,   0,  180, 179, 40,  255, 0.9),
            # transparan (mirip glass, tapi umumnya lebih value)
            (0,   0,  120, 179, 35,  255, 0.5),
        ],
        'electronics': [
            # hitam/gelap (casing, PCB)
            (0,   0,    0, 179, 80, 80, 1.0),
            # hijau PCB
            (35,  40,  30, 85, 255, 180, 0.8),
            # abu-abu (housing, metal bracket)
            (0,   0,   60, 179, 30, 180, 0.6),
        ],
        # alias
        'electronic': None,  # di-set sama dengan 'electronics' di __init__
    }

    # Batas keputusan
    MIN_MATCH = 0.10       # skor minimum agar kelas asli dianggap valid
    OVERRIDE_GAP = 0.20    # selisih skor minimum untuk override ke kelas lain
    DROP_MAX = 0.08        # jika skor terbaik < ini, deteksi di-drop

    # Pasangan kelas yang sering rancu — hanya override antar pasangan ini
    CONFUSABLE = {
        frozenset({'paper', 'organic'}),
        frozenset({'glass', 'metal'}),
        frozenset({'paper', 'glass'}),
        frozenset({'paper', 'metal'}),
        frozenset({'organic', 'glass'}),
        frozenset({'plastic', 'glass'}),
    }

    def __init__(self, names, enabled=True, sample_frac=0.5, verbose=False):
        """
        names: dict {cls_id: 'name'} dari model
        sample_frac: fraksi piksel di center crop yang disampling (hemat CPU)
        """
        self.names = names
        self.id_to_name = {k: self._norm(v) for k, v in names.items()}
        self.name_to_ids = {}
        for k, v in self.id_to_name.items():
            self.name_to_ids.setdefault(v, []).append(k)
        self.enabled = enabled
        self.sample_frac = sample_frac
        self.verbose = verbose

        # pastikan alias 'electronic' = 'electronics'
        if self.PROFILES.get('electronic') is None and 'electronics' in self.PROFILES:
            self.PROFILES['electronic'] = self.PROFILES['electronics']

        # pre-compile mask ranges sebagai numpy arrays untuk kecepatan
        self._compiled = {}
        for name, ranges in self.PROFILES.items():
            if ranges is None:
                continue
            self._compiled[name] = [
                (np.array([r[0], r[1], r[2]], dtype=np.uint8),
                 np.array([r[3], r[4], r[5]], dtype=np.uint8),
                 r[6])
                for r in ranges
            ]

    @staticmethod
    def _norm(name):
        return 'electronic' if name.startswith('electronic') else name

    def _score_roi(self, hsv_roi, name):
        """Hitung weighted match score untuk kelas `name` pada ROI HSV."""
        compiled = self._compiled.get(name)
        if not compiled:
            return 0.0
        total = hsv_roi.shape[0]
        if total == 0:
            return 0.0
        best = 0.0
        for lo, hi, weight in compiled:
            mask = np.all((hsv_roi >= lo) & (hsv_roi <= hi), axis=1)
            score = (mask.sum() / total) * weight
            best = max(best, score)
        return best

    def verify(self, frame, dets):
        """
        Filter dan koreksi deteksi berdasar warna.
        Input:  frame (BGR), dets list of (x1,y1,x2,y2,conf,cls_id)
        Output: dets baru (bisa berkurang, bisa berubah cls_id)
        """
        if not self.enabled or not dets:
            return dets

        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = []
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            name = self.id_to_name.get(cls_id, str(cls_id))

            # crop ROI & center-sample untuk hemat CPU
            roi = hsv_full[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                result.append(det)
                continue

            # center crop 70% untuk menghindari background di pinggir bbox
            rh, rw = roi.shape[:2]
            margin_x, margin_y = int(rw * 0.15), int(rh * 0.15)
            roi = roi[margin_y:rh - margin_y, margin_x:rw - margin_x]
            if roi.size == 0:
                result.append(det)
                continue

            pixels = roi.reshape(-1, 3)
            if self.sample_frac < 1.0 and len(pixels) > 100:
                idx = np.random.choice(len(pixels), int(len(pixels) * self.sample_frac), replace=False)
                pixels = pixels[idx]

            # skor kelas asli
            orig_score = self._score_roi(pixels, name)

            # jika skor asli cukup tinggi, terima apa adanya
            if orig_score >= self.MIN_MATCH:
                result.append(det)
                continue

            # cari kelas alternatif terbaik (hanya di antara kelas-kelas confusable)
            best_alt_name, best_alt_score = None, 0.0
            for other_name in self._compiled:
                if other_name == name:
                    continue
                if frozenset({name, other_name}) not in self.CONFUSABLE:
                    continue
                s = self._score_roi(pixels, other_name)
                if s > best_alt_score:
                    best_alt_score = s
                    best_alt_name = other_name

            # keputusan
            if best_alt_score > self.DROP_MAX and (best_alt_score - orig_score) >= self.OVERRIDE_GAP:
                # override ke kelas alternatif
                new_ids = self.name_to_ids.get(best_alt_name, [])
                new_id = new_ids[0] if new_ids else cls_id
                if self.verbose:
                    print(f"  [COLOR] {name}({orig_score:.2f}) -> {best_alt_name}({best_alt_score:.2f})")
                result.append((x1, y1, x2, y2, conf, new_id))
            elif orig_score < self.DROP_MAX and best_alt_score < self.DROP_MAX:
                # semua skor terlalu rendah — drop deteksi ini
                if self.verbose:
                    print(f"  [COLOR] DROP {name}({orig_score:.2f}) best_alt={best_alt_score:.2f}")
                continue
            else:
                # skor asli tidak cukup bagus tapi tidak ada alternatif jelas — tetap terima
                result.append(det)

        return result


# ----------------------------------------------------------------------------
# Worker inference: selalu proses frame terbaru, tanpa memblok kamera/tampilan/servo
# ----------------------------------------------------------------------------
class InferenceWorker(threading.Thread):
    def __init__(self, cam, detector, idle_interval=0.5, hold=3.0, motion_thr=2.0):
        super().__init__(daemon=True)
        self.cam, self.det = cam, detector
        self.idle_interval = idle_interval   # 0 = gating dimatikan (selalu full speed)
        self.hold = hold                     # detik tetap "aktif" setelah ada gerakan/deteksi
        self.gate = MotionGate(motion_thr)
        self.paused = False
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._seq = 0
        self.result = ([], 0.0, 0)           # (dets, ms, frame_id)
        self.ms_avg = 0.0
        self.fps = 0.0
        self.active = True
        self._active_until = time.time() + hold

    def run(self):
        last_id, last_t, last_infer = -1, time.perf_counter(), 0.0
        while not self._stop.is_set():
            fid, frame = self.cam.read()
            if frame is None or fid == last_id or self.paused:
                time.sleep(0.003)
                continue
            last_id = fid
            now = time.time()

            if self.idle_interval > 0:
                if self.gate.moving(frame):
                    self._active_until = now + self.hold
                self.active = now < self._active_until
                if not self.active and (now - last_infer) < self.idle_interval:
                    continue                                   # mode hemat: lewati frame ini
            else:
                self.active = True

            t0 = time.perf_counter()
            dets = self.det.detect(frame)
            ms = (time.perf_counter() - t0) * 1000
            last_infer = time.time()
            if dets:
                self._active_until = last_infer + self.hold    # ada objek -> tetap full speed

            self.ms_avg = ms if self.ms_avg == 0 else 0.9 * self.ms_avg + 0.1 * ms
            t = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(t - last_t, 1e-6)) if self.fps else 1.0 / max(t - last_t, 1e-6)
            last_t = t
            with self._cv:
                self.result = (dets, ms, fid)
                self._seq += 1
                self._cv.notify_all()

    def wait_new(self, seq, timeout=0.2):
        """Tunggu hasil baru; return (seq, dets, ms, frame_id). Hemat CPU dibanding polling."""
        with self._cv:
            if self._seq == seq:
                self._cv.wait(timeout)
            d, ms, fid = self.result
            return self._seq, d, ms, fid

    def stop(self):
        self._stop.set()
