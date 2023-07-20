import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK

# model = YOLO('yolov8s.pt') #train 0
# model = YOLO('yolov8m.pt') #train 1
# model = YOLO('runs/detect/train/weights/last.pt') #train 0
model = YOLO('runs/detect/train2/weights/last.pt') #train 1

if __name__ == '__main__':
    # model.train(data = 'AIC-HCMC-2020.yaml', epochs = 300, batch=32, device = 0) #train0
    # model.train(data = 'AIC-HCMC-2020.yaml', epochs = 300, batch=16, device = 1) #train1
    model.train(resume=True)