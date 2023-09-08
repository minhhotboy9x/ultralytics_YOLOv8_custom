import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.detect.val import DetectionValidator
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.cfg import get_cfg
from ultralytics import YOLO
CFG = get_cfg(DEFAULT_CFG)

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')
    model.val('coco.yaml')

