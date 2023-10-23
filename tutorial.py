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
    learning_rate = 0.0000001
    weight_decay = 0.0001
    optimizer = optim.SGD(model.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    new_param = nn.Parameter(torch.Tensor([1.0]))  # Tham số mới
    new_optimizer_group = {'params': new_param, 'lr':1}
    optimizer.add_param_group(new_optimizer_group)
    # Lấy tốc độ học của optimizer
    learning_rate = optimizer.param_groups[0]['lr']
    learning_rate1 = optimizer.param_groups[1]['lr']

    print("Tốc độ học của optimizer là:", learning_rate, ' ', learning_rate1)
