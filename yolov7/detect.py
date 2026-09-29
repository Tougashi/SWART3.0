import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import RPi.GPIO as GPIO
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

GPIO.setmode(GPIO.BCM)

factory = PiGPIOFactory()
myGPIO360=21
myGPIO180=26
myCorrection=0.1
maxPW=(2.0+myCorrection)/1000
minPW=(1.0-myCorrection-0.2)/1000

myCorrectionb=0.55
maxPWb=(2.0+myCorrectionb)/1000
minPWb=(1.0-myCorrectionb)/1000

servo360 = Servo(myGPIO360,min_pulse_width=minPWb,max_pulse_width=maxPWb, pin_factory=factory)
servo180 = Servo(myGPIO180,min_pulse_width=minPW,max_pulse_width=maxPW, pin_factory=factory)


#Pinout ultrasonic
TRIGAL=14
TRIGBL=18
TRIGCL=24
TRIGAR=8
TRIGBR=1
TRIGCR=16
ECHOAL=15
ECHOBL=23
ECHOCL=25
ECHOAR=7
ECHOBR=12
ECHOCR=20

LEDREDAL=17
LEDREDBL=27
LEDREDCL=22
LEDREDAR=5
LEDREDBR=6
LEDREDCR=13
LEDWHITEAL=2
LEDWHITEBL=3
LEDWHITECL=4
LEDWHITEAR=10
LEDWHITEBR=9
LEDWHITECR=11

# Setup the GPIO pin as an output
GPIO.setup(LEDREDAL, GPIO.OUT)
GPIO.setup(LEDREDBL, GPIO.OUT)
GPIO.setup(LEDREDCL, GPIO.OUT)
GPIO.setup(LEDREDAR, GPIO.OUT)
GPIO.setup(LEDREDBR, GPIO.OUT)
GPIO.setup(LEDREDCR, GPIO.OUT)
GPIO.setup(LEDWHITEAL, GPIO.OUT)
GPIO.setup(LEDWHITEBL, GPIO.OUT)
GPIO.setup(LEDWHITECL, GPIO.OUT)
GPIO.setup(LEDWHITEAR, GPIO.OUT)
GPIO.setup(LEDWHITEBR, GPIO.OUT)
GPIO.setup(LEDWHITECR, GPIO.OUT)




#Full level
ALN = 10.5
ALF = 7.5
BLN = 11
BLF = 5
CLN = 12
CLF = 9
ARN = 9
ARF = 6
BRN = 10
BRF = 9
CRN = 10
CRF = 6
ALS = True
BLS = True
CLS = True
ARS = True
BRS = True
CRS = True

def servo360a():
    value2= 1
    servo360.value=value2
    time.sleep(2.11)
def servo360b():
    value2= 0
    servo360.value=value2
    time.sleep(2.11)
def servo360c():
    value2= -1
    servo360.value=value2
    time.sleep(2.11)
    
      
def servo180a():
    value2= 1
    servo180.value=value2
def servo180b():
    value2= 0
    servo180.value=value2
def servo180c():
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

def ukur(trig,echo):
    #print("distance measurement in progress")
    GPIO.setup(trig,GPIO.OUT)
    GPIO.setup(echo,GPIO.IN)
    GPIO.output(trig,False)
    #print("waiting for sensor to settle")
    time.sleep(0.1)
    GPIO.output(trig,True)
    time.sleep(0.00001)
    GPIO.output(trig,False)
    while GPIO.input(echo)==0:
        pulse_start=time.time()
    while GPIO.input(echo)==1:
        pulse_end=time.time()
    pulse_duration=pulse_end-pulse_start
    distance=pulse_duration*17150
    distance=round(distance,2)
    
    # print("distance:",distance,"cm")
    # time.sleep(2)  
    return distance   

def ledon(led_pin):
    GPIO.output(led_pin, GPIO.HIGH)  # Turn the LED strip on

def ledoff(led_pin):
    GPIO.output(led_pin, GPIO.LOW)
        

def cekpenuh():
    AL = ukur(TRIGAL,ECHOAL)
    BL = ukur(TRIGBL,ECHOBL)
    CL = ukur(TRIGCL,ECHOCL)
    AR = ukur(TRIGAR,ECHOAR)
    BR = ukur(TRIGBR,ECHOBR)
    CR = ukur(TRIGCR,ECHOCR)


    if (AL<ALF):
        ledon(LEDREDAL)
        ALS = False
        
    else:
        ledoff(LEDREDAL)
        ALS = True
    
    if (BL<BLF):
        ledon(LEDREDBL)
        BLS = False
    else:
        ledoff(LEDREDBL)
        BLS = True

    if (CL<CLF):
        ledon(LEDREDCL)
        CLS = False
    else:
        ledoff(LEDREDCL)
        CLS = True

    if(AR<ARF):
        ledon(LEDREDAR)
        ARS = False
    else:
        ledoff(LEDREDAR)
        ARS = True

    if(BR<BRF):
        ledon(LEDREDBR)
        BRS = False
    else:
        ledoff(LEDREDBR)
        BRS = True

    if(CR<CRF):
        ledon(LEDREDCR)
        CRS = False
    else:
        ledoff(LEDREDCR)
        CRS = True


def arahkan(s):

    if(s=='organic'):
        ledon(LEDWHITEAL)
    elif(s=='metal'):
        ledon(LEDWHITEBL)
    elif(s=='electronic'):
        ledon(LEDWHITECL)
    elif(s=='paper'):
        ledon(LEDWHITEAR)
    elif(s=='glass'):
        ledon(LEDWHITEBR)
    elif(s=='plastic'):
        ledon(LEDWHITECR)

    if (s == 'organic') or (s == 'paper'):
        servo180a()
    elif (s == 'plastic') or (s == 'electronic'):
        servo180c()
    else:
        servo180b()
    
    time.sleep(0.5)

    if (s == 'metal') or (s == 'electronic') or (s == 'organic'):
        servo360c()
    elif (s == 'paper') or (s == 'glass') or (s == 'plastic'):
        servo360a()   

    if(s=='organic'):
        ledoff(LEDWHITEAL)
    elif(s=='metal'):
        ledoff(LEDWHITEBL)
    elif(s=='electronic'):
        ledoff(LEDWHITECL)
    elif(s=='paper'):
        ledoff(LEDWHITEAR)
    elif(s=='glass'):
        ledoff(LEDWHITEBR)
    elif(s=='plastic'):
        ledoff(LEDWHITECR)    


def pos_awal():
    servo180b()
    servo360b()


def mulai(s):
    pos_awal()
    kategori = scek(s)
    
    if kategori != 'kosong':
        arahkan(kategori)
    
    time.sleep(0.5)
    pos_awal()
    cekpenuh()
    

#Pinout ultrasonic
# TRIGAL=21
# TRIGBL=21
# TRIGCL=21
# TRIGAR=21
# TRIGBR=21
# TRIGCR=21
# ECHOAL=20
# ECHOBL=20
# ECHOCL=20
# ECHOAR=20
# ECHOBR=20
# ECHOCR=20




    


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS                                                                                                                                       )
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            #gerak
            mulai(s)
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #GPIO.cleanup()
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
