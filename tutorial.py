import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.detect.val import DetectionValidator
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.cfg import get_cfg
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
if __name__ == '__main__':
    # tensor([1, 2, 3, 1, 2, 3])
    y = torch.tensor([[[[1.0, 2],
                        [1.0, 2]],
                        [[2.0, 2],
                        [2.0, 2]]]])
    m = nn.Softmax2d()
    print(m(y))

