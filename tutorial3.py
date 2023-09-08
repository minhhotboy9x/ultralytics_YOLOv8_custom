import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.cfg import get_cfg
from ignite.metrics import SSIM


# model = YOLO("runs/detect/train3/weights/last.pt")  # or a segmentation model .i.e yolov8n-seg.pt
# model1 = YOLO("yolov8m.pt")  # or a segmentation model .i.e yolov8n-seg.pt
# model2 = YOLO("yolov8m.pt")  # or a segmentation model .i.e yolov8n-seg.pt

#inputs = 'datasets/AIC-HCMC-2020/images/val'  # list of numpy arrays
# results = model.val('ultralytics/datasets/AIC-HCMC-2020.yaml')  # generator of Results objects
# python yolov8_KD/KD_training_ver2.py --model KD/train2/weights/last.pt --teacher yolov8m.pt --resume True
# python yolov8_KD/KD_training_ver2.py --model yolov8s.pt --teacher yolov8m.pt --data coco.yaml --batch 8 --epochs 200
class MinMaxRescalingLayer(nn.Module):
    def __init__(self):
        super(MinMaxRescalingLayer, self).__init__()

    def forward(self, x, y):
        min_val = torch.min(x.min(-1)[0].min(-1)[0], y.min(-1)[0].min(-1)[0])
        max_val = torch.max(x.max(-1)[0].max(-1)[0], y.max(-1)[0].min(-1)[0])
        print(min_val.shape)
        rescaled_x = (x - min_val[:,:,None,None]) / (max_val[:,:,None,None] - min_val[:,:,None,None])
        rescaled_y = (y - min_val[:,:,None,None]) / (max_val[:,:,None,None] - min_val[:,:,None,None])
        return rescaled_x, rescaled_y

class DSSIM(nn.Module):
    def __init__(self, device = 'cpu'):
        super(DSSIM, self).__init__()
        self.device = device

    def forward(self, y_pred, y_true):
        data_range = torch.max(torch.max(y_pred, y_true) - torch.min(y_pred, y_true), torch.tensor(1.0))
        self.cal =  SSIM(data_range=data_range, device=self.device)
        self.cal.reset()
        self.cal.update((y_pred, y_true))
        loss = self.cal.compute()
        loss = 1-loss
        # print(f'------------{loss}--------------')
        return loss

if __name__ == "__main__":
    input1 = 1 + torch.randn(2, 10, 100, 100) * 10
    input2 = input1.clone()
    input2[0, 0, 0, 0] = 1.0
    scaler = MinMaxRescalingLayer()
    input1_scaled, input2_scaled= scaler(input1, input2)
    dssim = DSSIM()
    print(dssim(input1, input2))
    print(dssim(input1_scaled, input2_scaled))