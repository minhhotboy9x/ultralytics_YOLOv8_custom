import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK

# model = YOLO('yolov8s.pt') 
# model = YOLO('yolov8m.yaml')
# model = YOLO('yolov8s.yaml')
model = YOLO('KD_feature/train8/weights/best.pt')

if __name__ == '__main__':
    # model.train(data = 'coco_minitrain_10k.yaml', epochs = 500, batch=16, device = 0) # run 4
    # model.train(data = 'coco_minitrain_10k.yaml', resume=True, batch=16, device = 0)
    metrics = model.val(data = 'coco_minitrain_10k.yaml', device = 1) # run 0
    print(metrics)
    # print(metrics)
# train 10k thuong

# KD_v8s feat_adapt m dssim
# python yolov8_KD/KD_training_ver7.py --model yolov8s.yaml --teacher yolov8m_coco10k.pt --data coco_minitrain_10k.yaml --type_kd_loss dssim --epochs 500 --batch 16 --device 0 # KD_feat 10
# python yolov8_KD/KD_training_ver7.py --model KD_feature/train10/weights/last.pt --teacher yolov8m_coco10k.pt --type_kd_loss dssim --resume True --device 0 # KD_feat 10

# KD
# python yolov8_KD/KD_training_ver3.py --model yolov8s.yaml --teacher yolov8m.pt --type_kd_loss dssim --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 0 # KD_feat_out 2
# python yolov8_KD/KD_training_ver3.py --model KD_feature_out/train2/weights/last.pt --teacher yolov8m.pt --type_kd_loss dssim --resume True --device 0 # KD_feat_out 2

# python yolov8_KD/KD_training.py --model yolov8s.pt --teacher yolov8m.pt --project coco_kd  --data coco.yaml --epochs 300 --batch 32 --device 1 --worker 4 # coco_kd 1
# python yolov8_KD/KD_training.py --model coco_kd/train/weights/last.pt --teacher yolov8m.pt --resume True --device 0 # coco_kd 1

# test
# python yolov8_KD/KD_training_ver4.py --model yolov8s.yaml --teacher yolov8m.pt --project coco_128 --type_kd_loss mse --data coco128.yaml --epochs 50 --batch 4 --device 1

# prune
# python benchmarks/prunability/yolov8_pruning.py --model "asset/trained_model/coco_mini_10k/v8s_tea_v8m_coco10k_featbase_DSSIM(normalize)_[15,18,21].pt" --data coco_minitrain_10k.yaml --batch 16 --workers 2 --iterative-steps 16 --target-prune-rate 0.5 --epochs 40 --project yolov8s_coco10k_prune --device 1

# KD_fgd
# python yolov8_KD/KD_training_fgd.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 1 # KD_feat 6
# python yolov8_KD/KD_training_fgd.py --model KD_feature/train6/weights/last.pt --teacher yolov8m.pt --resume True --device 1 # KD_feat 6

# KD_v8s feat_adapt m fgfi
# python yolov8_KD/KD_training_ver7.py --model yolov8s.yaml --teacher yolov8m_coco10k.pt --data coco_minitrain_10k.yaml --type_kd_loss fgfi --epochs 500 --batch 16 --device 1 # KD_feat 9
# python yolov8_KD/KD_training_ver7.py --model KD_feature/train9/weights/last.pt --teacher yolov8m_coco10k.pt --type_kd_loss fgfi --resume True --device 1 # KD_feat 9
