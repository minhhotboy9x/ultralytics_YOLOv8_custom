import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK

# model = YOLO('yolov8s.pt') 
# model = YOLO('yolov8n.yaml')
# model = YOLO('yolov8s.yaml')
model = YOLO('/home/minhnq/ultralytics/asset/trained_model/UA-DETRAC/v8s_UA_DETRAC.pt') 

if __name__ == '__main__':
    # model.train(data = 'UA-DETRAC.yaml', epochs = 500, batch=32, device = 0) # run 9 (ReLu)
    # model.train(data = 'UA-DETRAC.yaml', resume=True, batch=16, device = 0, workers=4)
    metrics = model.val(data = 'UA-DETRAC.yaml', device = 0) # run 0
    # print(model.model.model)
    print(metrics)
# train 10k thuong

# KD_v8s feat_adapt m dssim
# python yolov8_KD/KD_training_ver7.py --model yolov8s.yaml --teacher yolov8m_coco10k.pt --data coco_minitrain_10k.yaml --type_kd_loss dssim --epochs 500 --batch 16 --device 0 # KD_feat 10
# python yolov8_KD/KD_training_ver7.py --model KD_feature/train10/weights/last.pt --teacher yolov8m_coco10k.pt --typưe_kd_loss dssim --resume True --device 1 # KD_feat 10

# KD
# python yolov8_KD/KD_training_ver3.py --model yolov8s.yaml --teacher yolov8m.pt --type_kd_loss dssim --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 0 # KD_feat_out 2
# python yolov8_KD/KD_training_ver3.py --model KD_feature_out/train2/weights/last.pt --teacher yolov8m.pt --type_kd_loss dssim --resume True --device 0 # KD_feat_out 2

# python yolov8_KD/KD_training.py --model yolov8s.pt --teacher yolov8m.pt --project coco_kd  --data coco.yaml --epochs 300 --batch 32 --device 1 --worker 4 # coco_kd 1
# benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/UA-DETRAC/v8s_UA_DETRAC.pt --data UA-DETRAC.yaml --epochs 50 --batch 16 --device 0
# test
# python yolov8_KD/KD_training_ver7.py --model yolov8s.yaml --teacher yolov8m.pt --project coco_128 --type_kd_loss LD --data coco128.yaml --epochs 50 --batch 4 --device 0
# python yolov8_KD/KD_training_ver7.py --model asset/trained_model/coco_mini_10k/v8s+sm_tea_v8m_coco10k_feat_FGFI_[22,23,24]_alpha=0.1.pt --teacher yolov8m.pt --project coco_128 --type_kd_loss fgd --data coco128.yaml --epochs 50 --batch 4 --device 1

# prune
# python benchmarks/prunability/yolov8_pruning.py --model "asset/trained_model/coco_mini_10k/v8s_tea_v8m_coco10k_featbase_DSSIM(normalize)_[15,18,21].pt" --data coco_minitrain_10k.yaml --batch 16 --workers 2 --iterative-steps 16 --target-prune-rate 0.5 --epochs 40 --project yolov8s_coco10k_prune --device 1

# KD_fgd
# python yolov8_KD/KD_training_fgd.py --model yolov8s.yaml --teacher yolov8m.pt --data coco_minitrain_10k.yaml --epochs 500 --batch 16 --device 1 # KD_feat 6
# python yolov8_KD/KD_training_fgd.py --model KD_feature/train6/weights/last.pt --teacher yolov8m.pt --resume True --device 1 # KD_feat 6

# KD_v8m smaller FGFI loss
# python yolov8_KD/KD_training_ver7.py --model yolov8m.yaml --teacher yolov8m_coco10k.pt --data coco_minitrain_10k.yaml --type_kd_loss FGFI --epochs 500 --batch 16 --device 1 # KD_feat 15
# python yolov8_KD/KD_training_ver7.py --model KD_feature/train15/weights/last.pt --teacher yolov8m_coco10k.pt --type_kd_loss FGFI --resume True --device 1 # KD_feat 15

# KD_v8s LD loss
# python yolov8_KD/KD_training_ver7.py --model yolov8s.yaml --teacher yolov8m_coco10k.pt --data coco_minitrain_10k.yaml --type_kd_loss LD --epochs 500 --batch 16 --device 0 # KD_feat 17
# python yolov8_KD/KD_training_ver7.py --model KD_feature/train17/weights/last.pt --teacher yolov8m_coco10k.pt --type_kd_loss LD --resume True --device 0 # KD_feat 17

# pruning 
# python benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/UA-DETRAC/v8s_UA_DETRAC.pt --data UA-DETRAC.yaml --epochs 50 --batch 16 --device 0