import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK

# model = YOLO('yolov8s.pt') 
# model = YOLO('yolov8m.pt')
# model = YOLO('yolov8s.yaml') 
model = YOLO('yolov8.yaml') # run 3


if __name__ == '__main__':
    model.train(data = 'coco_minitrain_10k.yaml', epochs = 500, batch=32, device = 0, workers=2) # run 2
    # model.val('coco.yaml', device = 1)
    # model.val(data = 'AIC-HCMC-2020.yaml')

# KD
# python yolov8_KD/KD_training_ver2.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 300 --batch 32 --device 1 # KD 3
# python yolov8_KD/KD_training_ver2.py --model KD/train3/weights/last.pt --teacher yolov8m.pt --resume True --device 1 # KD 3
# python benchmarks/prunability/yolov8_pruning.py --model yolov8m.pt --data coco.yaml --target-prune-rate 0.5 --iterative-steps 20 --epochs 20 --batch 16 --device 1 # prune 0


# python yolov8_KD/KD_training.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 0 # KD_feature 2
# python yolov8_KD/KD_training.py --model KD_feature/train2/weights/last.pt --teacher yolov8m.pt --resume True --device 0 # KD_feature 2

# python yolov8_KD/KD_training.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 500 --type_kd_loss mse --batch 16 --device 0 # KD_feature 3 (mse)
# python yolov8_KD/KD_training.py --model KD_feature/train3/weights/last.pt --teacher yolov8m.pt --type_kd_loss mse --resume True --device 0 # KD_feature 3

# python yolov8_KD/KD_training.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 0 # KD_feature 4 (normalize ssim)
# python yolov8_KD/KD_training.py --model KD_feature/train4/weights/last.pt --teacher yolov8m.pt --resume True --device 0 # KD_feature 4

# python yolov8_KD/KD_training.py --model yolov8s.pt --teacher yolov8m.pt --project coco_kd  --data coco.yaml --epochs 500 --batch 16 --device 0 --worker 2 # coco_kd 1
# python yolov8_KD/KD_training.py --model coco_kd/train/weights/last.pt --teacher yolov8m.pt --resume True --device 0 # coco_kd 1

# test
# python yolov8_KD/KD_training.py --model yolov8s.pt --teacher yolov8l.pt --project coco_128  --data coco128.yaml --epochs 5 --batch 1 --device cpu
