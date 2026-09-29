"""
SWART 2.0 - YOLOv9-s Inference Script (PyTorch)
=============================================
"""

import argparse
import sys
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import cv2
import torch

# Add YOLOv9 root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (check_img_size, non_max_suppression, scale_boxes, Profile)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(ROOT / "best.pt"), help="Path to weights file")
    parser.add_argument("--source", type=str, default="0", help="Source: 0 for webcam, or path to image/folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' atau 'cuda:0' (default: cpu)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Error: File weights tidak ditemukan: {weights_path}")
        sys.exit(1)

    print("=" * 60)
    print("  SWART 2.0 - YOLOv9-s Object Detection (PyTorch)")
    print("=" * 60)
    print(f"  Weights  : {weights_path}")
    print(f"  Source   : {args.source}")
    print(f"  Conf     : {args.conf}")
    print(f"  IoU      : {args.iou}")
    print(f"  Img Size : {args.imgsz}")
    print(f"  Device   : {args.device}")
    print("=" * 60)

    # Initialize
    device = select_device(args.device)
    print("\n🔄 Memuat model PyTorch...")
    model = DetectMultiBackend(weights_path, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((args.imgsz, args.imgsz), s=stride)

    print(f"✅ Model berhasil dimuat!")
    print(f"\n📋 Kelas yang dideteksi ({len(names)} kelas):")
    for idx, name in names.items():
        print(f"   [{idx}] {name}")

    # Dataloader
    source = str(args.source)
    is_webcam = source.isnumeric()
    
    if is_webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        print(f"\n🚀 Menjalankan deteksi pada Webcam (Tekan 'q' untuk keluar)...")
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        print(f"\n🚀 Menjalankan deteksi pada file...")

    # Run inference
    model.warmup(imgsz=(1 if pt else len(dataset), 3, *imgsz))
    dt = (Profile(), Profile(), Profile())
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.float()  # uint8 to fp32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
            pred = non_max_suppression(pred, args.conf, args.iou, None, False, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            if is_webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            annotator = Annotator(im0, line_width=2, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results string
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            cv2.imshow("SWART 2.0 - " + str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                if is_webcam and isinstance(dataset, LoadStreams):
                    # Trying to close thread properly
                    for thread in dataset.threads:
                        if thread.is_alive():
                            thread.join(timeout=1)
                cv2.destroyAllWindows()
                return

        print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

if __name__ == "__main__":
    main()
