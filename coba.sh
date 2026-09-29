#!/bin/bash

sudo su
conda init bash
conda activate yolov5
cd yolov7
sudo pigpiod
python detect.py --weights wasteaifinal.pt --source 0

