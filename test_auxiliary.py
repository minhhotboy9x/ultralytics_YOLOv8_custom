import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics import YOLO
import time

model = YOLO('yolov8n.yaml')
# model2 = YOLO('yolov8n.pt')
if __name__ == '__main__':
    model.train(data='coco_minitrain_10k.yaml', batch=4, epochs=300)