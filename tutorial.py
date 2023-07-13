import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.detect.val import DetectionValidator
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.cfg import get_cfg

CFG = get_cfg(DEFAULT_CFG)

if __name__ == '__main__':
    args = dict(task = 'detect', mode = 'train', model='yolov8n.pt', 
                data='AIC-HCMC-2020.yaml', epochs=100, device='0', resume='True')
    model = DetectionTrainer(overrides=args)
    model.train()
    # model.validate()
    # args2 = model.args   
    # args2.split = 'test'
    # val = DetectionValidator(save_dir=model.save_dir, args=args2)
    # val(model=model.best)  
