import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.modules import Conv2, C2f, C2f_v2, Conv
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.cfg import get_cfg


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

def convert_c2f_v2_to_c2f(c2f_v2):
    c1 = c2f_v2.cv1.conv.in_channels
    c2 = c2f_v2.cv2.conv.out_channels
    c = int(c2 * 0.5)
    n = int(c2f_v2.cv2.conv.in_channels / c) - 2
    c2f = C2f(c1, c2, n, shortcut=infer_shortcut(c2f_v2.m[0]))
    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()
    # for name, param in state_dict_v2.items():
    #     print(f"Parameter Name: {name}")
    cv0_state = state_dict_v2["cv0.conv.weight"]
    cv1_state = state_dict_v2["cv1.conv.weight"]
    state_dict['cv1.conv.weight'] = torch.concat([cv0_state, cv1_state], dim=0)

    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        cv0_bn = state_dict_v2[f'cv0.bn.{bn_key}']
        cv1_bn = state_dict_v2[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv1.bn.{bn_key}'] = torch.concat([cv0_bn, cv1_bn], dim=0)
    
    for key in state_dict:
        if not key.startswith('cv1.') and not key.startswith('cv0.'):
            state_dict[key] = state_dict_v2[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f_v2):
        attr_value = getattr(c2f_v2, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f, attr_name, attr_value)
    
    c2f.forward_split = c2f.forward_split_v2
    c2f.load_state_dict(state_dict)
    return c2f

    

def replace_c2f_v2_with_c2f(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f_v2):
            Conv.default_act = child_module.cv0.act
            c2f = convert_c2f_v2_to_c2f(child_module)
            setattr(module, name, c2f)
        else:
            replace_c2f_v2_with_c2f(child_module)

if __name__ == "__main__":
    model = YOLO('yolov8s_relu.pt')
    # model = YOLO('pruning/train23/step_0_finetune/weights/best.pt')
    # replace_c2f_v2_with_c2f(model.model)
    # result = model.val(data="VOC.yaml", split="test", batch=4)
    # print(model.model)
    model.export(format='onnx')
    # print(result)


