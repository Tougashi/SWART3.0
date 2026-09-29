import RPi.GPIO as GPIO
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

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
    
def servo360a():
    value2= 1
    servo360.value=value2
def servo360b():
    value2= 0
    servo360.value=value2
def servo360c():
    value2= -1
    servo360.value=value2
      
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

def arahkan(s):
    if (s == 'organic') or (s == 'plastic'):
        servo180_c()
    elif (s == 'paper') or (s == 'electronic'):
        servo180_a()
    else:
        servo180_b()
    
    time.sleep(2)
 
    
    if (s == 'organic') or (s == 'glass') or (s=='paper'):
        servo360kiri()
        servo360stop()
    else:
        servo360kanan()
        servo360stop()

def mulai(s):
    servo360b()
    #time.sleep(2)
    servo180_b()
   
    
    
    
mulai('paper')