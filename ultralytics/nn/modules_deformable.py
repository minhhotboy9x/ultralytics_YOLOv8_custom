# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
deformable modules
"""

import math

import torch
import torchvision.ops
import torch.nn as nn
from ultralytics.nn.modules import autopad, DFL, Conv
import torch.quantization as quant
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x
    
class DeConv(nn.Module):
    """Standard deformable convolution with args(ch_in, ch_out, kernel, stride, padding, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = DeformableConv2d(c1, c2, k, s, autopad(k, p, d), dilation=d, bias=False)
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

class DeDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(DeConv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(DeConv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        pass


class FeatureExtractor(nn.Module):
    def __init__(self, c1, c2, N=6):
        super().__init__()
        self.extractors = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c1, c2, 3, stride=1, padding=1), 
                          nn.BatchNorm2d(c2),
                          nn.ReLU()) if i==0
            else nn.Sequential(nn.Conv2d(c2, c2, 3, stride=1, padding=1),
                               nn.BatchNorm2d(c2),
                                nn.ReLU())
            for i in range(N))
    
    def forward(self, x): # output list 
        X_inters = []
        for module in self.extractors:
            x = module(x)
            X_inters.append(x)
        return X_inters # b, c2, h, w
    

class LayerAttention(nn.Module):
    def __init__(self, c_hidden, N=6):
        super().__init__()
        self.cross_layer = nn.Sequential(
            nn.Linear(c_hidden*N, N),
            nn.Sigmoid()
        )
    def forward(self, x): # input X_inter from feature extractor, ouput X_task
        X_task = []
        x_concated = torch.cat(x, dim=1) 
        x_inter = x_concated.mean(dim=(2, 3)) # GAP
        w = self.cross_layer(x_inter)
        w = w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print(w.shape, x[0].shape)
        for i in range(len(x)):
            X_task.append(x[i] * w[:, i])
        X_task = torch.cat(X_task, dim=1) 
        return X_task # b, c_hidden * N, h, w


class PrePredictor(nn.Module):
    def __init__(self, c_hidden, c_out, N=6):
        super().__init__()
        self.layer_attention = LayerAttention(c_hidden, N)
        self.conv1 = nn.Conv2d(c_hidden * N, c_hidden, 1, stride=1)
        self.bn = nn.BatchNorm2d(c_hidden)
        self.conv2 = nn.Conv2d(c_hidden, c_out, 3, stride=1, padding=1)
    
    def forward(self, x): # input list(tensor)
        x = self.layer_attention(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

class TDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        c_max = max(c2, c3)
        self.cv1 = nn.ModuleList(FeatureExtractor(x, c_max, 6) for x in ch)
        self.cv2 = nn.ModuleList(PrePredictor(c_max, 4 * self.reg_max, 6) for _ in ch)
        self.cv3 = nn.ModuleList(PrePredictor(c_max, self.nc, 6) for _ in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x_iter = self.cv1[i](x[i])
            x[i] = torch.cat((self.cv2[i](x_iter), self.cv3[i](x_iter)), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        pass