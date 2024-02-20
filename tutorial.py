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
        self.quant = torch.quantization.QuantStub()
        self.model = models.resnet18(pretrained=True)
        self.dequant = torch.quantization.DeQuantStub()
        self.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))

if __name__ == '__main__':
    input = torch.randn(1, 3, 224, 224)
    model = MyModel()
    print(model)
    torch.quantization.prepare_qat(model, inplace=True)
    output = model(input)
    loss = 10 - output.sum()
    loss.backward()
    print(model)
    quantized_model = torch.quantization.convert(model)
    # Lưu mô hình
    torch.save(model, 'test_quantized.pt')
