import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.cfg import get_cfg
from ultralytics import YOLO
from torchvision.transforms import GaussianBlur
import torch.optim as optim

model = YOLO('yolov8s.yaml')
if __name__ == '__main__':
    dump_image = torch.zeros(1, 3, 640, 640)
    img, feature = model.model(dump_image, augment=False, mask_id = [16, 20, 24])
    print(img[0].shape)
    for i in feature:
        print(i.shape)
    
