import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np

from ultralytics import YOLO

from gpiozero import LED, DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Servo

# Inisialisasi PiGPIOFactory
factory = PiGPIOFactory()

# Setup servo
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

# Pinout ultrasonic
TRIGAL, ECHOAL = 14, 15
TRIGBL, ECHOBL = 18, 23
TRIGCL, ECHOCL = 24, 25
TRIGAR, ECHOAR = 8, 7
TRIGBR, ECHOBR = 1, 12
TRIGCR, ECHOCR = 16, 20

# Pin LED
led_pins = {
    'REDAL': 17, 'REDBL': 27, 'REDCL': 22, 'REDAR': 5, 'REDBR': 6, 'REDCR': 13,
    'WHITEAL': 2, 'WHITEBL': 3, 'WHITECL': 4, 'WHITEAR': 10, 'WHITEBR': 9, 'WHITECR': 11
}

# Setup LED dan sensor jarak
leds = {name: LED(pin, pin_factory=factory) for name, pin in led_pins.items()}
distance_sensors = {
    'AL': DistanceSensor(echo=ECHOAL, trigger=TRIGAL, pin_factory=factory),
    'BL': DistanceSensor(echo=ECHOBL, trigger=TRIGBL, pin_factory=factory),
    'CL': DistanceSensor(echo=ECHOCL, trigger=TRIGCL, pin_factory=factory),
    'AR': DistanceSensor(echo=ECHOAR, trigger=TRIGAR, pin_factory=factory),
    'BR': DistanceSensor(echo=ECHOBR, trigger=TRIGBR, pin_factory=factory),
    'CR': DistanceSensor(echo=ECHOCR, trigger=TRIGCR, pin_factory=factory)
}

# Full level
ALN, ALF = 10.5, 7.5
BLN, BLF = 11, 5
CLN, CLF = 12, 9
ARN, ARF = 9, 6
BRN, BRF = 10, 9
CRN, CRF = 10, 6
ALS = BLS = CLS = ARS = BRS = CRS = True

# ... (fungsi-fungsi lain tetap sama)

def servo360kiri():
    value2= -0.9
    servo360.value=value2
    time.sleep(2.11)
def servo360stop():
    value2= 0
    servo360.value=value2
    time.sleep(2.11)
def servo360kanan():
    value2= 0.9
    servo360.value=value2
    time.sleep(2.11)
    
      
def servo180_a():
    value2= 1
    servo180.value=value2
def servo180_b():
    value2= 0
    servo180.value=value2
def servo180_c():
    value2= -1
    servo180.value=value2

def check_single_true(*variables):
    true_count = sum(variables)
    return true_count == 1

def scek(s):
    global paper, metal, electronic, plastic, glass, organic
    paper = False
    metal = False
    electronic = False
    plastic = False
    glass = False
    organic = False
    
    data = s.split()
    if 'paper,' in data:
        paper = True
    elif 'electronic,' in data:
        electronic = True
    elif 'metal,' in data:
        metal = True
    elif 'plastic,' in data:
        plastic = True
    elif 'glass,' in data:
        glass = True
    elif 'organic,' in data:
        organic = True
    
    cek1 = check_single_true(paper,electronic,metal, plastic, glass, organic)
    
    if cek1 :
        if 'paper,' in data:
            hasil = 'paper'
        elif 'electronic,' in data:
            hasil = 'electronic'
        elif 'metal,' in data:
            hasil = 'metal'
        elif 'plastic,' in data:
            hasil = 'plastic'
        elif 'glass,' in data:
            hasil = 'glass'
        elif 'organic,' in data:
            hasil = 'organic'
    else:
        hasil = 'kosong'
        
    return hasil

def cekpenuh():
    global ALS, BLS, CLS, ARS, BRS, CRS
    for name, sensor in distance_sensors.items():
        distance = sensor.distance * 100  # konversi ke cm
        if name == 'AL' and distance < ALF:
            leds['REDAL'].on()
            ALS = False
        elif name == 'BL' and distance < BLF:
            leds['REDBL'].on()
            BLS = False
        # ... (lanjutkan untuk sensor lainnya)
        else:
            leds[f'RED{name}'].off()
            globals()[f'{name}S'] = True

def arahkan(s):
    if(s=='organic'):
        leds['WHITEAL'].on()
    elif(s=='metal'):
        leds['WHITEBL'].on()
    elif(s=='electronic'):
        leds['WHITECL'].on()
    elif(s=='paper'):
        leds['WHITEAR'].on()
    elif(s=='glass'):
        leds['WHITEBR'].on()
    elif(s=='plastic'):
        leds['WHITECR'].on()
        
    if (s == 'organic') or (s == 'paper'):
        servo180_a()
    elif (s == 'plastic') or (s == 'electronic'):
        servo180_c()
    else:
        servo180_b()
    
    time.sleep(0.5)
 
    if (s == 'metal') or (s == 'electronic') or (s == 'organic'):
        servo360kiri()
    elif (s == 'paper') or (s == 'glass') or (s == 'plastic'):
        servo360kanan()
    
    if(s=='organic'):
        leds['WHITEAL'].off()
    elif(s=='metal'):
        leds['WHITEBL'].off()
    elif(s=='electronic'):
        leds['WHITECL'].off()
    elif(s=='paper'):
        leds['WHITEAR'].off()
    elif(s=='glass'):
        leds['WHITEBR'].off()
    elif(s=='plastic'):
        leds['WHITECR'].off()

def pos_awal():
    servo180_b()
    servo360stop()

def mulai(s):
    pos_awal()
    kategori = scek(s)
    
    if kategori != 'kosong':
        arahkan(kategori)
    
    time.sleep(0.5)
    pos_awal()
    cekpenuh()
    
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Inisialisasi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model YOLOv9
    model = YOLO(weights)
    model.to(device)

    # Dataloader
    if webcam:
        dataset = cv2.VideoCapture(int(source) if source.isnumeric() else source)
    else:
        dataset = cv2.VideoCapture(source)

    # Get names and colors
    names = model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Jalankan inferensi
    while True:
        ret, frame = dataset.read()
        if not ret:
            break

        # Inferensi
        results = model(frame)

        # Proses deteksi
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                label = f'{names[cls]} {conf:.2f}'
                color = colors[cls]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Stream hasil
        if view_img:
            cv2.imshow('YOLOv9 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Proses hasil deteksi
        detected_classes = [names[cls] for cls in classes]
        s = ', '.join(detected_classes)
        mulai(s)

    dataset.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov9.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)

        detect()
    except KeyboardInterrupt:
        print("Program dihentikan oleh Pengguna.")
    finally:
        # Membersihkan resources
        for led in leds.values():
            led.close()
        for sensor in distance_sensors.values():
            sensor.close()
        servo360.close()
        servo180.close()