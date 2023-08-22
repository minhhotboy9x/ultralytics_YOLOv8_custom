import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK

# model = YOLO('yolov8s.pt') 
# model = YOLO('yolov8m.pt')
# model = YOLO('yolov8s.yaml') 
model = YOLO('runs/detect/train3/weights/last.pt') # run 3


if __name__ == '__main__':
    # model.train(data = 'coco_minitrain_10k.yaml', epochs = 300, batch=32, device = 0) #run 3
    model.train(resume=True, device = 0)
    # model.val(data = 'AIC-HCMC-2020.yaml')

# KD
# python yolov8_KD/KD_training_ver2.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 300 --batch 32 --device 0 # KD 0
# python yolov8_KD/KD_training_ver2.py --model KD/train2/weights/last.pt --teacher yolov8m.pt --resume True --device 0 # KD 0
# python benchmarks/prunability/yolov8_pruning.py --model yolov8m.pt --data coco.yaml --target-prune-rate 0.5 --iterative-steps 20 --epochs 20 --batch 16 --device 1 # prune 0