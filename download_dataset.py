"""Download SWART dataset dari Roboflow."""
import os
os.environ['ROBOFLOW_API_KEY'] = 'wynrCu4rrnAqpIrQUHsq'

from roboflow import Roboflow

rf = Roboflow(api_key='wynrCu4rrnAqpIrQUHsq')

# Download Version 3 (tanpa augmentasi, 36963 gambar asli) 
# Ini yang kita butuhkan untuk resplit yang bersih
project = rf.workspace("swart").project("swart")
version = project.version(3)
print("[INFO] Downloading Version 3 (raw, no augmentation) as yolov8...")
dataset = version.download("yolov8", location="d:/Farid2/Project/SWART3.0/dataset_raw")
print(f"[OK] Downloaded to: {dataset.location}")
