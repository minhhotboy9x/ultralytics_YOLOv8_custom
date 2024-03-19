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

class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(3, 4, 3, bias=False)
        self.conv1 = nn.Conv2d(3, 2, 3, bias=False)
        self.conv2 = nn.Conv2d(3, 2, 3, bias=False)
        conv_weight = self.conv.weight
        self.conv1.weight = nn.Parameter(conv_weight[:2, ...])
        self.conv2.weight = nn.Parameter(conv_weight[2:, ...])

    def forward(self, x):
        return self.conv(x) - torch.cat([self.conv1(x), self.conv2(x)], dim=1)
    
    def forward2(self, x):
        return ''


if __name__ == '__main__':
    input_example = torch.ones(1, 3, 10, 10)
    model = MyModel()
    print(model(input_example))
    