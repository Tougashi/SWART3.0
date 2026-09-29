"""
SWART 3.0 - Test YOLOv9 Detection dengan Iriun Webcam (OPTIMIZED)
=================================================================
Script untuk testing deteksi sampah menggunakan YOLOv9
dengan Iriun Webcam di PC/Laptop — DIOPTIMASI untuk CPU.

Optimasi:
    - Threaded camera capture (non-blocking)
    - Frame skipping (inference setiap N frame)
    - ONNX Runtime backend (3-5x lebih cepat dari PyTorch di CPU)
    - Reduced image size default (320px)
    - Optimized HUD rendering

Penggunaan:
    python test_yolov9_iriun.py
    python test_yolov9_iriun.py --weights weights/best.pt --source 1
    python test_yolov9_iriun.py --weights weights/best.onnx --source 1
    python test_yolov9_iriun.py --weights weights/best.pt --imgsz 224 --skip 2

Kontrol keyboard:
    q / ESC  = Keluar
    s        = Screenshot
    p        = Pause/Resume
    r        = Reset statistik
    +/-      = Confidence threshold
    1/2/3    = Frame skip (1=tiap frame, 2=tiap 2 frame, dst)
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np

# ============================================================
# Patch torch.load untuk PyTorch 2.6+ compatibility
# ============================================================
try:
    import torch
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Setup path untuk repo yolov9
FILE = Path(__file__).resolve()
ROOT = FILE.parent / 'yolov9'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Warna per kategori (BGR)
# ============================================================
CATEGORY_COLORS = {
    'paper':       (0, 200, 255),
    'metal':       (180, 180, 180),
    'electronic':  (255, 100, 0),
    'electronics': (255, 100, 0),   # alias dari ONNX model
    'plastic':     (0, 255, 100),
    'glass':       (255, 200, 100),
    'organic':     (50, 200, 50),
}
DEFAULT_COLOR = (200, 200, 200)


# ============================================================
# Threaded Camera Capture — non-blocking frame read
# ============================================================
class CameraStream:
    """Membaca frame kamera di thread terpisah agar tidak blocking."""

    def __init__(self, source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer

        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

        # Baca frame pertama
        self.ret, self.frame = self.cap.read()

    def start(self):
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    @property
    def is_opened(self):
        return self.cap.isOpened() and self.ret

    @property
    def resolution(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()


# ============================================================
# ONNX Runtime Inference — jauh lebih cepat dari PyTorch di CPU
# ============================================================
class ONNXDetector:
    """Inference menggunakan ONNX Runtime (optimal untuk CPU)."""

    def __init__(self, onnx_path, conf_thres=0.25, iou_thres=0.45):
        print(f"[INFO] Loading ONNX model: {onnx_path}")
        providers = ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()
        sess_options.inter_op_num_threads = os.cpu_count()

        self.session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_h = input_shape[2] if isinstance(input_shape[2], int) else 640
        self.input_w = input_shape[3] if isinstance(input_shape[3], int) else 640

        # Coba baca nama kelas dari metadata
        meta = self.session.get_modelmeta()
        self.names = {}
        if meta.custom_metadata_map and 'names' in meta.custom_metadata_map:
            import ast
            try:
                self.names = ast.literal_eval(meta.custom_metadata_map['names'])
            except:
                pass

        if not self.names:
            self.names = {
                0: 'electronic', 1: 'glass', 2: 'metal',
                3: 'organic', 4: 'paper', 5: 'plastic'
            }

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        print(f"[INFO] ONNX input shape: {input_shape}")
        print(f"[INFO] Kelas: {list(self.names.values())}")
        print(f"[INFO] Threads: {os.cpu_count()}")

    def preprocess(self, frame, target_size=None):
        """Letterbox + normalize."""
        if target_size:
            h, w = target_size
        else:
            h, w = self.input_h, self.input_w

        img, ratio, (dw, dh) = self._letterbox(frame, (h, w))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img, ratio, (dw, dh)

    def _letterbox(self, img, new_shape=(640, 640)):
        """Resize gambar dengan letterbox (maintain aspect ratio)."""
        shape = img.shape[:2]  # h, w
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    def detect(self, frame, conf_thres=None, imgsz=None):
        """Jalankan deteksi dan return list of (box, class_name, confidence)."""
        if conf_thres is None:
            conf_thres = self.conf_thres

        # Selalu gunakan native model size (ONNX biasanya fixed)
        target = (self.input_h, self.input_w)
        img, ratio, (dw, dh) = self.preprocess(frame, target)

        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        pred = outputs[0]  # shape: (1, N, 5+num_classes) atau (1, N, 6)

        # Post-process
        detections = self._postprocess(pred, frame.shape, ratio, dw, dh, conf_thres)
        return detections

    def _postprocess(self, pred, orig_shape, ratio, dw, dh, conf_thres):
        """
        NMS dan filter deteksi.
        YOLOv9 ONNX output format: [batch, num_attrs, num_preds]
          - num_attrs = 4 (cx, cy, w, h) + num_classes
          - Tidak ada objectness score (anchor-free format)
        """
        detections = []

        if pred.ndim == 3:
            # Shape [1, 10, 1029] → perlu transpose ke [1029, 10]
            pred = pred[0]  # remove batch → [10, 1029]
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T  # transpose → [1029, 10]

        num_cols = pred.shape[1]
        num_classes = num_cols - 4  # 4 kolom box (cx, cy, w, h)

        if num_classes <= 0:
            return detections

        # Class scores (tanpa objectness — langsung class confidence)
        cls_scores = pred[:, 4:]  # [N, num_classes]
        cls_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(len(cls_scores)), cls_ids]

        # Filter by confidence
        mask = confs > conf_thres
        pred = pred[mask]
        cls_ids = cls_ids[mask]
        confs = confs[mask]

        if len(pred) == 0:
            return detections

        # Convert center xywh → xyxy
        boxes = pred[:, :4].copy()
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes[:, 0] = cx - w / 2  # x1
        boxes[:, 1] = cy - h / 2  # y1
        boxes[:, 2] = cx + w / 2  # x2
        boxes[:, 3] = cy + h / 2  # y2

        # Scale boxes ke ukuran asli
        boxes[:, 0] = (boxes[:, 0] - dw) / ratio
        boxes[:, 1] = (boxes[:, 1] - dh) / ratio
        boxes[:, 2] = (boxes[:, 2] - dw) / ratio
        boxes[:, 3] = (boxes[:, 3] - dh) / ratio

        # Clip
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_shape[0])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_shape[1])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_shape[0])

        # NMS
        indices = self._nms(boxes, confs, self.iou_thres)

        for i in indices:
            cls_id = int(cls_ids[i])
            cls_name = self.names.get(cls_id, f'class_{cls_id}')
            box = boxes[i].astype(int).tolist()
            conf = float(confs[i])
            detections.append((box, cls_name, conf))

        return detections


    def _nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression sederhana."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep


# ============================================================
# PyTorch Detector (fallback, lebih lambat di CPU)
# ============================================================
class PyTorchDetector:
    """Inference menggunakan YOLOv9 repo + PyTorch."""

    def __init__(self, weights_path, conf_thres=0.25, iou_thres=0.45, device='cpu'):
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device

        self.device = select_device(device)
        print(f"[INFO] Loading PyTorch model: {weights_path}")
        print(f"[INFO] Device: {self.device}")

        self.model = DetectMultiBackend(str(weights_path), device=self.device, fp16=False)
        self.stride = int(self.model.stride)
        self.names = self.model.names
        self.pt = self.model.pt

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Warmup
        dummy = torch.zeros(1, 3, 320, 320).to(self.device)
        self.model(dummy)

        print(f"[INFO] Kelas: {list(self.names.values())}")

    def detect(self, frame, conf_thres=None, imgsz=None):
        """Jalankan deteksi."""
        from utils.general import non_max_suppression, scale_boxes
        from utils.augmentations import letterbox

        if conf_thres is None:
            conf_thres = self.conf_thres

        sz = imgsz or 320
        img = letterbox(frame, (sz, sz), stride=self.stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model(img)

        if isinstance(pred, (list, tuple)):
            if isinstance(pred[0], (list, tuple)):
                pred = pred[0][1]
            else:
                pred = pred[0]

        pred = non_max_suppression(pred, conf_thres, self.iou_thres, max_det=100)

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    cls_id = int(cls)
                    cls_name = self.names.get(cls_id, f'class_{cls_id}')
                    box = [int(v) for v in xyxy]
                    detections.append((box, cls_name, float(conf)))

        return detections


# ============================================================
# FPS & Stats
# ============================================================
class FPSCounter:
    def __init__(self, window=30):
        self.times = deque(maxlen=window)

    def tick(self):
        self.times.append(time.perf_counter())

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        elapsed = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0


class DetectionStats:
    def __init__(self):
        self.counts = defaultdict(int)
        self.total = 0
        self.with_det = 0

    def update(self, classes):
        self.total += 1
        if classes:
            self.with_det += 1
        for c in classes:
            self.counts[c] += 1

    def reset(self):
        self.counts.clear()
        self.total = self.with_det = 0

    @property
    def rate(self):
        return (self.with_det / self.total * 100) if self.total else 0.0


# ============================================================
# Drawing functions (optimized — single overlay)
# ============================================================
def draw_box(frame, box, label, conf, color):
    """Bounding box dengan corner accents."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents
    cl = min(18, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, 3)
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, 3)
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, 3)
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, 3)

    # Label
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ly = max(y1 - th - 8, 0)
    cv2.rectangle(frame, (x1, ly), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_hud(frame, fps, conf, stats, skip, paused, backend):
    """HUD overlay — single overlay operation untuk performa."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel kiri atas
    cv2.rectangle(overlay, (0, 0), (230, 130), (0, 0, 0), -1)

    # Panel bawah
    cv2.rectangle(overlay, (0, h - 28), (w, h), (0, 0, 0), -1)

    # Panel statistik (kanan atas) jika ada deteksi
    if stats.counts:
        n = len(stats.counts)
        ph = (n + 1) * 26 + 10
        cv2.rectangle(overlay, (w - 200, 0), (w, ph), (0, 0, 0), -1)

    if paused:
        cv2.rectangle(overlay, (w // 2 - 80, h // 2 - 25), (w // 2 + 80, h // 2 + 25), (0, 0, 0), -1)

    # Single blend
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Teks FPS
    fps_color = (0, 255, 0) if fps >= 15 else (0, 200, 255) if fps >= 8 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_color, 2)

    # Info
    cv2.putText(frame, f"Conf: {conf:.2f}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Skip: {skip} | Det: {stats.rate:.0f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, f"Backend: {backend}", (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)

    # Statistik kanan atas
    if stats.counts:
        cv2.putText(frame, "DETEKSI:", (w - 190, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        for i, (name, cnt) in enumerate(sorted(stats.counts.items(), key=lambda x: -x[1])):
            y = (i + 2) * 26
            c = CATEGORY_COLORS.get(name, DEFAULT_COLOR)
            cv2.circle(frame, (w - 185, y - 4), 5, c, -1)
            cv2.putText(frame, f"{name}: {cnt}", (w - 172, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Paused
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Hints bawah
    cv2.putText(frame, "Q:Quit S:Screenshot P:Pause R:Reset +/-:Conf 1/2/3:Skip",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)


# ============================================================
# Camera finder
# ============================================================
def find_camera(preferred=1, max_scan=5):
    """Cari kamera yang tersedia."""
    print(f"\n[INFO] Mencari kamera (preferred: {preferred})...")

    for idx in [preferred] + [i for i in range(max_scan) if i != preferred]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK] Kamera ditemukan di index {idx}")
                cap.release()
                return idx
            cap.release()

    print("[ERROR] Tidak ada kamera terdeteksi!")
    print("  - Pastikan Iriun Webcam aktif dan terhubung")
    return None


# ============================================================
# Auto export .pt → .onnx
# ============================================================
def export_pt_to_onnx(pt_path, imgsz=320):
    """Export model PyTorch ke ONNX untuk inference yang lebih cepat."""
    onnx_path = pt_path.with_suffix('.onnx')
    if onnx_path.exists():
        print(f"[INFO] ONNX sudah ada: {onnx_path}")
        return onnx_path

    print(f"[INFO] Exporting {pt_path.name} → ONNX (ini hanya sekali)...")
    try:
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device

        device = select_device('cpu')
        model = DetectMultiBackend(str(pt_path), device=device, fp16=False)

        dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
        torch.onnx.export(
            model.model, dummy, str(onnx_path),
            opset_version=12,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},
                          'output': {0: 'batch'}}
        )
        print(f"[OK] Exported: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return onnx_path
    except Exception as e:
        print(f"[WARN] Export gagal: {e}")
        print("[INFO] Fallback ke PyTorch backend...")
        return None


# ============================================================
# Main Detection Loop
# ============================================================
def run_detection(opt):
    print("=" * 60)
    print("  SWART 3.0 - YOLOv9 Detection (OPTIMIZED)")
    print("=" * 60)

    weights_path = Path(opt.weights)
    if not weights_path.exists():
        alt = FILE.parent / 'weights' / weights_path.name
        if alt.exists():
            weights_path = alt
        else:
            print(f"[ERROR] Weights tidak ditemukan: {opt.weights}")
            sys.exit(1)

    imgsz = opt.imgsz[0] if isinstance(opt.imgsz, list) else opt.imgsz

    # ---- Pilih backend ----
    backend_name = "unknown"
    detector = None

    if weights_path.suffix == '.onnx' and HAS_ONNX:
        # Langsung pakai ONNX
        detector = ONNXDetector(weights_path, opt.conf_thres, opt.iou_thres)
        backend_name = "ONNX Runtime"

    elif weights_path.suffix == '.pt':
        # Coba cari/buat ONNX dulu (lebih cepat)
        onnx_path = weights_path.with_suffix('.onnx')

        if HAS_ONNX and onnx_path.exists():
            print(f"[INFO] Ditemukan ONNX version: {onnx_path.name}")
            print(f"[INFO] Menggunakan ONNX (lebih cepat dari PyTorch di CPU)")
            detector = ONNXDetector(onnx_path, opt.conf_thres, opt.iou_thres)
            backend_name = "ONNX Runtime"

        elif HAS_ONNX and HAS_TORCH:
            # Auto export
            onnx_path = export_pt_to_onnx(weights_path, imgsz)
            if onnx_path and onnx_path.exists():
                detector = ONNXDetector(onnx_path, opt.conf_thres, opt.iou_thres)
                backend_name = "ONNX Runtime (auto-exported)"
            elif HAS_TORCH:
                detector = PyTorchDetector(weights_path, opt.conf_thres, opt.iou_thres, opt.device)
                backend_name = "PyTorch (CPU)"

        elif HAS_TORCH:
            detector = PyTorchDetector(weights_path, opt.conf_thres, opt.iou_thres, opt.device)
            backend_name = "PyTorch (CPU)"

    if detector is None:
        print("[ERROR] Tidak bisa load model! Pastikan torch atau onnxruntime terinstall.")
        sys.exit(1)

    print(f"[INFO] Backend: {backend_name}")
    print(f"[INFO] Image size: {imgsz}px")

    # ---- Setup kamera ----
    source = opt.source
    if source.isnumeric():
        cam_idx = find_camera(int(source))
        if cam_idx is None:
            sys.exit(1)
    else:
        cam_idx = source

    print(f"[INFO] Starting threaded camera (index: {cam_idx})...")
    stream = CameraStream(cam_idx if isinstance(cam_idx, int) else int(cam_idx), 640, 480)

    if not stream.is_opened:
        print("[ERROR] Kamera gagal dibuka!")
        sys.exit(1)

    stream.start()
    w, h = stream.resolution
    print(f"[INFO] Resolusi kamera: {w}x{h}")

    # ---- Init ----
    fps_counter = FPSCounter(30)
    stats = DetectionStats()
    conf_thres = opt.conf_thres
    skip = opt.skip
    paused = False
    frame_count = 0
    last_detections = []
    screenshot_dir = FILE.parent / 'screenshots'

    window_name = "SWART 3.0 - YOLOv9 Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(w, 1280), min(h, 720))

    print(f"[INFO] Frame skip: {skip} (inference tiap {skip} frame)")
    print(f"\n[START] Deteksi dimulai! Tekan 'q' untuk keluar.\n")

    try:
        while True:
            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    paused = False
                elif key == ord('q') or key == 27:
                    break
                continue

            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            fps_counter.tick()
            frame_count += 1

            # ---- Inference (dengan frame skipping) ----
            if frame_count % skip == 0:
                detections = detector.detect(frame, conf_thres=conf_thres, imgsz=imgsz)
                last_detections = detections

                detected_names = [d[1] for d in detections]
                stats.update(detected_names)

                if detected_names:
                    print(f"[DET] {', '.join(detected_names)} | FPS: {fps_counter.fps:.1f}")
            else:
                stats.update([])

            # ---- Draw detections (selalu gambar terakhir) ----
            for box, name, conf_val in last_detections:
                color = CATEGORY_COLORS.get(name, DEFAULT_COLOR)
                draw_box(frame, box, name, conf_val, color)

            # ---- HUD ----
            draw_hud(frame, fps_counter.fps, conf_thres, stats, skip, paused, backend_name)

            cv2.imshow(window_name, frame)

            # ---- Keyboard ----
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                screenshot_dir.mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fp = screenshot_dir / f"det_{ts}.jpg"
                cv2.imwrite(str(fp), frame)
                print(f"[SAVE] {fp}")
            elif key == ord('p'):
                paused = True
                print("[INFO] Paused")
            elif key == ord('r'):
                stats.reset()
                print("[INFO] Stats reset")
            elif key == ord('+') or key == ord('='):
                conf_thres = min(conf_thres + 0.05, 0.95)
                print(f"[INFO] Conf: {conf_thres:.2f}")
            elif key == ord('-'):
                conf_thres = max(conf_thres - 0.05, 0.05)
                print(f"[INFO] Conf: {conf_thres:.2f}")
            elif key == ord('1'):
                skip = 1
                print("[INFO] Skip: 1 (setiap frame)")
            elif key == ord('2'):
                skip = 2
                print("[INFO] Skip: 2")
            elif key == ord('3'):
                skip = 3
                print("[INFO] Skip: 3")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C")

    finally:
        print("\n" + "=" * 60)
        print("  RINGKASAN DETEKSI")
        print("=" * 60)
        print(f"  Backend              : {backend_name}")
        print(f"  Image size           : {imgsz}px")
        print(f"  Total frame          : {stats.total}")
        print(f"  Frame dgn deteksi    : {stats.with_det}")
        print(f"  Detection rate       : {stats.rate:.1f}%")
        print(f"  Rata-rata FPS        : {fps_counter.fps:.1f}")

        if stats.counts:
            print(f"\n  Deteksi per kategori:")
            for name, cnt in sorted(stats.counts.items(), key=lambda x: -x[1]):
                print(f"    {name:12s} : {cnt:5d}")

        print("=" * 60)

        stream.release()
        cv2.destroyAllWindows()
        print("[INFO] Selesai.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SWART 3.0 - YOLOv9 Optimized Test')
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                        help='Path model (.pt atau .onnx)')
    parser.add_argument('--source', type=str, default='1',
                        help='Kamera index (0/1/2) atau path video')
    parser.add_argument('--imgsz', '--img-size', nargs='+', type=int, default=[320],
                        help='Inference size (default: 320 untuk CPU)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IOU threshold NMS')
    parser.add_argument('--max-det', type=int, default=100,
                        help='Max deteksi per frame')
    parser.add_argument('--device', default='', help='Device (cpu/cuda)')
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='Filter kelas')
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true',
                        help='OpenCV DNN untuk ONNX')
    parser.add_argument('--skip', type=int, default=2,
                        help='Frame skip (inference tiap N frame, default: 2)')

    opt = parser.parse_args()
    print(f"\nConfig: {vars(opt)}\n")
    run_detection(opt)
