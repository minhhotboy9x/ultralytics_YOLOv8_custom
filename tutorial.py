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
    
def sum_tensors(feature_maps):
    stacked_tensor = torch.stack(feature_maps)  # Xếp các tensor trong list theo chiều mới (0)
    summed_tensor = torch.sum(stacked_tensor, dim=0)  # Tính tổng theo chiều mới (0)
    return summed_tensor


if __name__ == '__main__':
    
    # Tạo một danh sách chứa các tensor
    feature_maps = [torch.randn(1, 64, 80, 80), torch.randn(1, 64, 80, 80), torch.randn(1, 64, 80, 80)]

    # Tính tổng các tensor và kiểm tra kích thước
    output = sum_tensors(feature_maps)
    print("Kích thước feature map sau khi tính tổng:", output.size())
