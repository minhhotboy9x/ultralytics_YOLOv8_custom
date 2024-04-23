import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.cfg import get_cfg
from ultralytics import YOLO
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import torch.optim as optim
from ultralytics.nn import modules_quantized
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
from ptflops import get_model_complexity_info
from torchvision.ops import DeformConv2d
import torchvision.models as models

model = YOLO('runs/detect/train35/weights/best.pt')
model2 = YOLO('yolov8n.pt')
if __name__ == '__main__':
    print(torch.cuda.device_count())
    