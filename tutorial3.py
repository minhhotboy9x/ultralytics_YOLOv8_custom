import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.cfg import get_cfg
import onnx
from torch.onnx import export
import onnxruntime
from torch.quantization import QConfig

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(3, 3, 3)
        self.act = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        print('1: ', x[0, 0])
        x = self.conv(x)
        print('2: ', x[0, 0])
        x = self.act(x)
        print('3', x[0, 0])
        x = self.dequant(x)
        print('4', x[0, 0])
        return x

if __name__ == "__main__":
    model = YOLO('asset/trained_model/VOC/v8n_relu_VOC.pt')
    model.export(format='onnx', dynamic=True)

