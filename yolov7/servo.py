from gpiozero import AngularServo
from time import sleep

servo =AngularServo(11, min_angle=0, max_angle=360, min_pulse_width=0.0005, max_pulse_width=0.0025)

while (True):
    for angle in range(0, 360, 5):  # 0 - 180 degrees, 5 degrees at a time.
        servo.angle = angle
        sleep(0.05)
    for angle in range(360, 0, -5): # 180 - 0 degrees, 5 degrees at a time.
        servo.angle = angle
        sleep(0.05)
