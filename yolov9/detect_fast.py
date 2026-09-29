"""
SWART 3.0 - YOLOv9 ONNX Optimized for Raspberry Pi
==================================================
Versi Super Ringan (Threaded):
- Deteksi menggunakan ONNXRuntime (CPU execution).
- UI Minimal (Hanya Kamera + Bounding Box).
- Pergerakan Servo & LED menggunakan Threading (Asynchronous) agar kamera TIDAK LAG.
"""

import cv2
import numpy as np
import time
import threading
import sys
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("❌ onnxruntime belum terinstall. Jalankan: pip install onnxruntime")
    sys.exit(1)

# ==========================================
# KONFIGURASI HARDWARE (GPIO & SERVO)
# ==========================================
try:
    import RPi.GPIO as GPIO
except ImportError:
    import FakeRPi.GPIO as GPIO
    import FakeRPi
    sys.modules['RPi'] = FakeRPi
    sys.modules['RPi.GPIO'] = GPIO

from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

# --- Setup Servo ---
factory = PiGPIOFactory()
myGPIO360 = 21
myGPIO180 = 26
myCorrection = 0.1
maxPW = (2.0 + myCorrection) / 1000
minPW = (1.0 - myCorrection - 0.2) / 1000

myCorrectionb = 0.55
maxPWb = (2.0 + myCorrectionb) / 1000
minPWb = (1.0 - myCorrectionb) / 1000

servo360 = Servo(myGPIO360, min_pulse_width=minPWb, max_pulse_width=maxPWb, pin_factory=factory)
servo180 = Servo(myGPIO180, min_pulse_width=minPW, max_pulse_width=maxPW, pin_factory=factory)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# --- Setup LED ---
LEDREDAL = 17; LEDREDBL = 27; LEDREDCL = 22
LEDREDAR = 5;  LEDREDBR = 6;  LEDREDCR = 13
LEDWHITEAL = 2; LEDWHITEBL = 3; LEDWHITECL = 4
LEDWHITEAR = 10; LEDWHITEBR = 9; LEDWHITECR = 11

led_pins = [LEDREDAL, LEDREDBL, LEDREDCL, LEDREDAR, LEDREDBR, LEDREDCR,
            LEDWHITEAL, LEDWHITEBL, LEDWHITECL, LEDWHITEAR, LEDWHITEBR, LEDWHITECR]

for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)

# Flag agar mesin tidak diganggu jika sedang memilah sampah
is_moving = False

# ==========================================
# FUNGSI HARDWARE
# ==========================================
def servo360kiri():
    servo360.value = 1
    time.sleep(2.11)
def servo360stop():
    servo360.value = 0
    time.sleep(2.11)
def servo360kanan():
    servo360.value = -1
    time.sleep(2.11)

def servo180_a(): servo180.value = 1
def servo180_b(): servo180.value = 0
def servo180_c(): servo180.value = -1

def ledon(pin): GPIO.output(pin, GPIO.HIGH)
def ledoff(pin): GPIO.output(pin, GPIO.LOW)

def pos_awal():
    servo180_b()
    servo360stop()

def arahkan(s):
    if s == 'organic': ledon(LEDWHITEAL)
    elif s == 'metal': ledon(LEDWHITEBL)
    elif s == 'electronic': ledon(LEDWHITECL)
    elif s == 'paper': ledon(LEDWHITEAR)
    elif s == 'glass': ledon(LEDWHITEBR)
    elif s == 'plastic': ledon(LEDWHITECR)
    
    if s in ('organic', 'paper'): servo180_a()
    elif s in ('plastic', 'electronic'): servo180_c()
    else: servo180_b()
    
    time.sleep(0.5)
    
    if s in ('metal', 'electronic', 'organic'): servo360kanan()
    elif s in ('paper', 'glass', 'plastic'): servo360kiri()
    
    if s == 'organic': ledoff(LEDWHITEAL)
    elif s == 'metal': ledoff(LEDWHITEBL)
    elif s == 'electronic': ledoff(LEDWHITECL)
    elif s == 'paper': ledoff(LEDWHITEAR)
    elif s == 'glass': ledoff(LEDWHITEBR)
    elif s == 'plastic': ledoff(LEDWHITECR)

def hardware_worker(kategori):
    """
    Fungsi ini berjalan di thread terpisah.
    Kamera dan ONNX akan tetap berjalan lancar saat fungsi ini aktif.
    """
    global is_moving
    is_moving = True
    print(f"[MESIN] Memulai pemilahan sampah: {kategori}")
    pos_awal()
    arahkan(kategori)
    time.sleep(0.5)
    pos_awal()
    print(f"[MESIN] Selesai. Siap untuk sampah berikutnya.")
    is_moving = False

# ==========================================
# YOLOV9 ONNX UTILITIES (LIGHTWEIGHT)
# ==========================================
def letterbox(img, new_shape=320, color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)

def postprocess(output, conf_threshold, iou_threshold, num_classes):
    # Support YOLOv9 & YOLOv8 ONNX format
    predictions = output[1] if len(output) > 1 else output[0]
    if len(predictions.shape) == 3: predictions = predictions[0]
    if predictions.shape[0] < predictions.shape[1]: predictions = predictions.T
    
    results = []
    if predictions.shape[1] == 5 + num_classes:
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > conf_threshold]
        if len(predictions) == 0: return results
        
        boxes = predictions[:, :4]
        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_confs = class_scores[np.arange(len(class_scores)), class_ids]
        scores = obj_conf * class_confs
        
        mask2 = scores > conf_threshold
        boxes = boxes[mask2]; scores = scores[mask2]; class_ids = class_ids[mask2]
        if len(boxes) == 0: return results
        
        boxes = xywh2xyxy(boxes)
        keep = nms(boxes, scores, iou_threshold)
        
        for k in keep:
            results.append({"box": boxes[k], "score": scores[k], "class": class_ids[k]})
    else:
        # Format tanpa obj_conf
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        class_scores = np.max(scores, axis=1)
        
        mask = class_scores > conf_threshold
        boxes = boxes[mask]; class_scores = class_scores[mask]; class_ids = class_ids[mask]
        if len(boxes) == 0: return results
        
        boxes = xywh2xyxy(boxes)
        keep = nms(boxes, class_scores, iou_threshold)
        
        for k in keep:
            results.append({"box": boxes[k], "score": class_scores[k], "class": class_ids[k]})
            
    return results

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    global is_moving
    
    # --- PENGATURAN ---
    weights_path = "SWART.onnx"
    conf_thres = 0.50  # Dipertinggi sedikit agar tidak banyak false-detect saat mesin gerak
    iou_thres = 0.45
    source = "http://10.208.250.70:8080/video"
   # Webcam default
    
    # Kelas sampah sesuai model SWART
    class_names = {
        0: 'electronic', 1: 'glass', 2: 'metal',
        3: 'organic', 4: 'paper', 5: 'plastic'
    }
    num_classes = len(class_names)
    
    print("\n🔄 Memuat model ONNX...")
    session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Auto-detect ukuran input langsung dari model
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4 and isinstance(input_shape[2], int):
        imgsz = max(input_shape[2], input_shape[3])
    else:
        imgsz = 128  # Fallback
        
    print(f"✅ Model berhasil dimuat! Ukuran Input: {imgsz}x{imgsz}")
    
    import urllib.request
    
    # Untuk IP Webcam, OpenCV sering gagal di Raspberry Pi. 
    # Kita buat sistem fallback manual yang super stabil:
    is_http = isinstance(source, str) and source.startswith('http')
    if is_http:
        print("🔗 Menggunakan koneksi HTTP langsung ke Kamera HP...")
        # Ganti /video jadi /shot.jpg untuk ditarik frame-by-frame
        shot_url = source.replace('/video', '/shot.jpg')
    else:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n🚀 Menjalankan SWART 3.0 (Tekan 'q' pada jendela kamera untuk keluar)...")
    
    while True:
        if is_http:
            try:
                img_resp = urllib.request.urlopen(shot_url, timeout=2)
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                frame = cv2.imdecode(imgnp, -1)
                ret = True
            except Exception as e:
                print(f"Koneksi terputus: {e}")
                ret = False
        else:
            if not cap.isOpened(): break
            ret, frame = cap.read()
            
        if not ret or frame is None:
            time.sleep(1) # Tunggu sebentar jika koneksi ngelag
            continue
        
        h_orig, w_orig = frame.shape[:2]
        
        # 1. Preprocess Gambar
        img_resized, ratio, (dw, dh) = letterbox(frame, imgsz)
        img_input = img_resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        # 2. Deteksi ONNX
        outputs = session.run(None, {input_name: img_input})
        detections = postprocess(outputs, conf_thres, iou_thres, num_classes)
        
        detected_category = None
        
        # 3. Proses Hasil dan Gambar Bounding Box
        for det in detections:
            box = det["box"]
            
            # Kembalikan koordinat ke ukuran gambar asli
            x1 = int(max(0, min((box[0] - dw) / ratio, w_orig)))
            y1 = int(max(0, min((box[1] - dh) / ratio, h_orig)))
            x2 = int(max(0, min((box[2] - dw) / ratio, w_orig)))
            y2 = int(max(0, min((box[3] - dh) / ratio, h_orig)))
            
            cls_id = int(det["class"])
            score = det["score"]
            label_name = class_names.get(cls_id, f"Class {cls_id}")
            
            if detected_category is None:
                detected_category = label_name
                
            # Gambar Bounding Box (Ringan & Cukup Jelas)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label_name} {score:.2f}"
            
            # Teks bayangan hitam (agar terbaca jelas di kondisi terang)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            # Teks hijau muda
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            
        # 4. Asynchronous Hardware Control (Kamera tidak akan delay!)
        if detected_category and not is_moving:
            t = threading.Thread(target=hardware_worker, args=(detected_category,))
            t.daemon = True
            t.start()
            
        # UI Status Indikator di Kiri Atas
        if is_moving:
            status_text = "[ MESIN SEDANG MEMILAH ]"
            color = (0, 0, 255) # Merah (BGR)
        else:
            status_text = "[ MESIN SIAP ]"
            color = (255, 0, 0) # Biru (BGR)
            
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # 5. Tampilkan ke Layar
        cv2.imshow("SWART 3.0", frame)
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
