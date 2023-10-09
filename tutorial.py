import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.detect.val import DetectionValidator
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.cfg import get_cfg
from ultralytics import YOLO
from mmcv.cnn import constant_init, kaiming_init

# model = YOLO('yolov8s.pt')
if __name__ == '__main__':
   import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = MyModel()
layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Conv2d(3, 3, 1, 1)
        )
# Khởi tạo trọng số của tầng cuối cùng của model bằng giá trị 0
layer2.inited = True
constant_init(layer2[-1], val=0)
print(layer2[-1].bias)
# print(layer2[0].weight)



