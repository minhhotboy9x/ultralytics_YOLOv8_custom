import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.cfg import get_cfg
from ultralytics import YOLO
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import torch.optim as optim

# model = YOLO('yolov8s.yaml')
def mask_split(tensor, indices):
    sorter = torch.argsort(indices, dim=-1)
    _, counts = torch.unique(indices, return_counts=True, dim=-1)
    print(tensor[sorter], counts)
    return torch.split(tensor[sorter], counts.tolist(), dim=-1)

if __name__ == '__main__':
    # Tạo một tensor 3 chiều ngẫu nhiên
    tensor_2d = torch.tensor([[1, 2], [2, 3], [3, 4]])
    mask = torch.tensor([1, 0, 1], dtype=torch.bool)

    # Áp dụng mask để lấy các dòng tương ứng
    result_tensor = tensor_2d[mask]

    print("Tensor ban đầu:")
    print(tensor_2d)
    print("Mask:")
    print(mask)
    print("Kết quả:")
    print(result_tensor)