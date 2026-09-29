#!/bin/bash

sleep 30
echo 'script started' >> /home/admin/log.txt

source /home/admin/miniforge3/bin/activate yolov9
echo 'environtment activated' >> /home/admin/log.txt

sleep 2
cd /home/admin/yolov9
echo 'changed direktory' >> /home/admin/log.txt

export DISPLAY=:0

sleep 2
sudo chmod 666 /dev/gpiomem

sleep 2
python /home/admin/yolov9/detectv2.py --weight /home/admin/yolov9/best.onnx --imgsz=224  --source 0
echo 'program started' >> /home/admin/log.txt
