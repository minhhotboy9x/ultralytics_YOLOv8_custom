import torch
import torch.nn as nn
import torch.quantization as quantization
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck, C2f_v2
from ultralytics.nn.modules_quantized import Q_Conv
from ultralytics.nn.modules_deformable import TDetect
from ultralytics.yolo.utils import (DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS)
from ultralytics.nn.tasks import torch_safe_load, guess_model_task
from torch.ao.quantization.qconfig import QConfig

def forward(self:Q_Conv, x):
    """Apply convolution if act=SiLu."""
    return self.act(self.dequant(self.bn(self.conv(self.quant(x)))))

def transfer_weights_qconv(conv, qconv):
    state_dict_conv = conv.state_dict()
    state_dict_qconv = qconv.state_dict()
    state_dict_qconv['conv.weight'] = state_dict_conv['conv.weight']
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        state_dict_qconv[f'bn.{bn_key}'] = state_dict_conv[f'bn.{bn_key}']
    for attr_name in dir(conv):
        attr_value = getattr(conv, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(qconv, attr_name, attr_value)
    qconv.load_state_dict(state_dict_qconv)

def replace_conv_with_qconv_v2_qat(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, Detect):
            continue
        elif isinstance(child_module, Conv):
            # Replace C2f with C2f_v2 while preserving its parameters
            conv2d = child_module.conv
            (c1, c2, k, s, p, g, d, act) = (conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size, 
                                conv2d.stride, conv2d.padding, conv2d.groups, conv2d.dilation[0], child_module.act)
            qconv = Q_Conv(c1, c2, k, s, p=p, g=g, d=d, act=act)
            setattr(module, name, qconv)
            transfer_weights_qconv(child_module, qconv) 
            qconv.eval()
            if not isinstance(act, nn.ReLU):
                torch.quantization.fuse_modules(qconv, [['conv', 'bn']], inplace=True)
                qconv.forward = forward.__get__(qconv)
            else:
                
                torch.quantization.fuse_modules(qconv, [['conv', 'bn', 'act']], inplace=True)
        else:
            replace_conv_with_qconv_v2_qat(child_module)

def replace_conv_with_qconv_v2_ptq(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, Detect):
            continue
        elif isinstance(child_module, Conv):
            # Replace C2f with C2f_v2 while preserving its parameters
            conv2d = child_module.conv
            (c1, c2, k, s, p, g, d, act) = (conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size, 
                                conv2d.stride, conv2d.padding, conv2d.groups, conv2d.dilation[0], child_module.act)
            qconv = Q_Conv(c1, c2, k, s, p=p, g=g, d=d, act=act)
            qconfig = QConfig(activation=quantization.HistogramObserver.with_args(reduce_range=True, qscheme=torch.per_tensor_affine),
                                weight=quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
                            )
            # qconfig = quantization.get_default_qconfig()
            qconv.qconfig = qconfig
            setattr(module, name, qconv)
            transfer_weights_qconv(child_module, qconv) 
            qconv.eval()
            if not isinstance(act, nn.ReLU):
                torch.quantization.fuse_modules(qconv, [['conv', 'bn']], inplace=True)
                qconv.forward = forward.__get__(qconv)
            else:
                torch.quantization.fuse_modules(qconv, [['conv', 'bn', 'act']], inplace=True)
        else:
            replace_conv_with_qconv_v2_ptq(child_module)
        
def attempt_load_one_weight_v2(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **ckpt['train_args']}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model'])  # quantized int8 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    # model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module compatibility updates
    # for m in model.modules():
    #     t = type(m)
    #     if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, TDetect):
    #         m.inplace = inplace  # torch 1.7.0 compatibility
    #     elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
    #         m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        
    return model, ckpt