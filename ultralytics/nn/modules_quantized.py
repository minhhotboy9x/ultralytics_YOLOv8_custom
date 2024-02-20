# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Quantized modules
"""

import math

import torch
import torch.nn as nn
from ultralytics.nn.modules import autopad
import torch.quantization as quant
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

class Q_Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.ReLU()  # default activation
    default_qconfig = 'fbgemm'

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, qconfig=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.dequant = torch.quantization.DeQuantStub()
        self.n_qconfig = self.default_qconfig if qconfig is True else qconfig if isinstance(qconfig, str) else 'fbgemm'
        self.qconfig = torch.quantization.get_default_qat_qconfig(self.n_qconfig)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.dequant(self.act(self.bn(self.conv(self.quant(x)))))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.dequant(self.act(self.conv(self.quant(x))))


class Q_DWConv(Q_Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True, ):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act) 


class Q_DWConvTranspose2d(nn.Module):
    """Depth-wise transpose convolution."""
    default_qconfig = 'fbgemm'
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, qconfig=True):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.tconv = nn.ConvTranspose2d(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
        self.n_qconfig = self.default_qconfig if qconfig is True else qconfig if isinstance(qconfig, str) else 'fbgemm'
        self.dequant = torch.quantization.DeQuantStub()
        self.qconfig = torch.quantization.get_default_qat_qconfig(self.n_qconfig)

    def forward(self, x):
        return self.dequant(self.tconv(self.quant(x)))

class Q_ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.ReLU()  # default activation
    default_qconfig = 'fbgemm'

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True, qconfig=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.n_qconfig = self.default_qconfig if qconfig is True else qconfig if isinstance(qconfig, str) else 'fbgemm'
        self.dequant = torch.quantization.DeQuantStub()
        self.qconfig = torch.quantization.get_default_qat_qconfig(self.n_qconfig)

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.dequant(self.act(self.bn(self.conv_transpose(self.quant(x)))))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.dequant(self.act(self.conv_transpose(self.quant(x))))
    
class Q_TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""
    default_qconfig = 'fbgemm'
    def __init__(self, c, num_heads, qconfig=True):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        self.dequant = torch.quantization.DeQuantStub()
        self.n_qconfig = self.default_qconfig if qconfig is True else qconfig if isinstance(qconfig, str) else 'fbgemm'
        self.qconfig = torch.quantization.get_default_qat_qconfig(self.n_qconfig)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.quant(x)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        x = self.dequant(x)
        return x


class Q_TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Q_Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(Q_TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1) # w*h, b, c
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
    
class Q_Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, ):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Q_Conv(c1, c_, k[0], 1)
        self.cv2 = Q_Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Q_BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Q_Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Q_Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.ReLU()
        self.m = nn.Sequential(*(Q_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class Q_C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Q_Conv(c1, c_, 1, 1)
        self.cv2 = Q_Conv(c1, c_, 1, 1)
        self.cv3 = Q_Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Q_Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Q_C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Q_Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Q_Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Q_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class Q_C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Q_Conv(c1, self.c, 1, 1)
        self.cv1 = Q_Conv(c1, self.c, 1, 1)
        self.cv2 = Q_Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Q_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Q_C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Q_Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Q_Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y

class Q_C3x(Q_C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Q_Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))

class Q_C3TR(Q_C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = Q_TransformerBlock(c_, c_, 4, n)

class Q_C3Ghost(Q_C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Q_GhostBottleneck(c_, c_) for _ in range(n)))


class Q_GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Q_Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Q_Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class Q_GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            Q_GhostConv(c1, c_, 1, 1),  # pw
            Q_DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            Q_GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(Q_DWConv(c1, c1, k, s, act=False), Q_Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)

class Q_SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Q_Conv(c1, c_, 1, 1)
        self.cv2 = Q_Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Q_SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Q_Conv(c1, c_, 1, 1)
        self.cv2 = Q_Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Q_Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Q_Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x)