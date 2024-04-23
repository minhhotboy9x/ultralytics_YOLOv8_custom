import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
from thop import profile
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck, C2fGhost, GhostBottleneck, CBAM, C2f_v2
import time

def reset_params(model):
    for name, child_module in model.named_children():
        if isinstance(child_module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            child_module.reset_parameters()
        else:
            reset_params(child_module)

def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            std = torch.sqrt(torch.tensor(2. / n))
            with torch.no_grad():
                m.weight.normal_(mean=0, std=std.item())
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

model = YOLO('asset/trained_model/VOC/v8s_relu_localprune_0,2_iter2_l2_VOC.pt')
# model = YOLO('yolov8n.yaml')

if __name__ == '__main__':
    initialize_weights(model.model)
    m = model.model.model[-1]
    m.bias_init()
    model.info()
    model.train(data = 'VOC.yaml', epochs = 300, batch=16, device = 0, project='test_prune') 
    metrics = model.val(data = 'VOC.yaml', batch=16, device=0, split='test', project='test_prune')
    print(metrics) 