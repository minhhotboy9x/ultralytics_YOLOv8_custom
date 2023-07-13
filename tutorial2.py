import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK

# model = YOLO("yolov8n.pt")


x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0])

z = x.detach() + y

print(z)  # Kết quả: tensor([6.])

z.backward()  # Tính gradient của z
print(x.grad)  # Kết quả: tensor([3.])