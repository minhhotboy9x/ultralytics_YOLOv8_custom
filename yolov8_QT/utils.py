import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck, C2f_v2
from ultralytics.nn.modules_quantized import Q_Conv
from ultralytics.nn.modules_deformable import TDetect
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.nn.tasks import torch_safe_load, guess_model_task

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

def replace_conv_with_qconv_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, Detect):
            continue
        elif isinstance(child_module, Conv):
            # Replace C2f with C2f_v2 while preserving its parameters
            conv2d = child_module.conv
            c1 = conv2d.in_channels
            c2 = conv2d.out_channels
            k = conv2d.kernel_size
            s = conv2d.stride
            p = conv2d.padding
            g = conv2d.groups
            d = conv2d.dilation[0]
            qconv = Q_Conv(c1, c2, k, s, p=p, g=g, d=d)
            setattr(module, name, qconv)
            transfer_weights_qconv(child_module, qconv) 
            qconv.eval()
            torch.quantization.fuse_modules(qconv, [['conv', 'bn', 'act']], inplace=True)
        else:
            replace_conv_with_qconv_v2(child_module)
        
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