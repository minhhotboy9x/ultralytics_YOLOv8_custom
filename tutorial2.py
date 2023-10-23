import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK

# model = YOLO('yolov8s.pt') 
# model = YOLO('yolov8m.yaml')
# model = YOLO('./yolov8s.yaml')
model = YOLO('runs/detect/train3/weights/last.pt')

if __name__ == '__main__':
    # model.train(data = 'coco_minitrain_10k.yaml', epochs = 500, batch=32, device = 1) # run 3
    model.train(data = 'coco_minitrain_10k.yaml', resume=True, batch=32, device = 1)
    # model.train(resume=True, device = 0) # run 0
# train 10k thuong

# KD
# python yolov8_KD/KD_training_fgd.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 0 # KD_feat 6
# python yolov8_KD/KD_training_fgd.py --model KD_feature/train6/weights/last.pt --teacher yolov8m.pt --resume True --device 0 # KD_feat 6

# KD
# python yolov8_KD/KD_training_ver3.py --model yolov8s.yaml --teacher yolov8m.pt --type_kd_loss dssim --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 0 # KD_feat_out 2
# python yolov8_KD/KD_training_ver3.py --model KD_feature_out/train2/weights/last.pt --teacher yolov8m.pt --type_kd_loss dssim --resume True --device 0 # KD_feat_out 2

# python yolov8_KD/KD_training.py --model yolov8s.pt --teacher yolov8m.pt --project coco_kd  --data coco.yaml --epochs 300 --batch 32 --device 1 --worker 4 # coco_kd 1
# python yolov8_KD/KD_training.py --model coco_kd/train/weights/last.pt --teacher yolov8m.pt --resume True --device 0 # coco_kd 1

# test
# python yolov8_KD/KD_training_ver6.py --model yolov8s.yaml --teacher yolov8l.pt --project coco_128 --type_kd_loss boxgauss --data coco128.yaml --epochs 50 --batch 16 --device 0

# prune
# python benchmarks/prunability/yolov8_pruning.py --model "asset/trained_model/coco_mini_10k/v8s_tea_v8m_coco10k_featbase_DSSIM(normalize)_[15,18,21].pt" --data coco_minitrain_10k.yaml --batch 16 --workers 2 --iterative-steps 16 --target-prune-rate 0.5 --epochs 40 --project yolov8s_coco10k_prune --device 1

# KD_fgd
# python yolov8_KD/KD_training_fgd.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 1 # KD_feat 6
# python yolov8_KD/KD_training_fgd.py --model KD_feature/train6/weights/last.pt --teacher yolov8m.pt --resume True --device 1 # KD_feat 6

# KD_fgfi_head
# python yolov8_KD/KD_training_ver5.py --model yolov8s.yaml --teacher yolov8m_coco10k.pt --data coco_minitrain_10k.yaml --type_kd_loss FGFI --epochs 500 --batch 16 --device 1 # KD_feat_out 4
# python yolov8_KD/KD_training_ver5.py --model KD_feature_out/train4/weights/last.pt --teacher yolov8m_coco10k.pt --type_kd_loss FGFI --resume True --device 1 # KD_feat_out 4
