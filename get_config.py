import numpy as np
import torch
import torch.nn as nn
import os
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.detect.val import DetectionValidator
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.cfg import get_cfg
from ultralytics import YOLO
# train trong train.py được gọi ở main
# => _do_train() trong trainer.py dùng để train và sẽ call đến train trong module.train() để set train mode
# 

# print(type(DEFAULT_CFG))
# args = dict(task = 'detect', mode = 'train', model='yolov8n.pt', 
#                 data='coco128.yaml', epochs=100, device='0')
# print(args)
# model = DetectionTrainer(overrides=args)
# print(model.model)

# class Descriptor:
#     def __init__(self) -> None:
#         self.a = 10

# def my_function(self: Descriptor):
#     print(self.a)
#     return "Hello, world!"

# descriptor = Descriptor()
# descriptor.b = 20
# descriptor.__setattr__("hello", my_function.__get__(descriptor))
# print(descriptor.hello()) 

# model = YOLO("yolov8n.yaml") 
RANK = int(os.getenv('RANK', -1))
print(RANK)
