import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
import onnx
from thop import profile
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
import onnxruntime
import time

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
# model = YOLO('yolov8s.yaml') 
# model = YOLO('yolov8m.pt')
# model = YOLO('/home/minhnq/ultralytics/pruning/train5/step_4_finetune/weights/best.pt')
# model = YOLO('asset/trained_model/UA-DETRAC/v8n_UA_DETRAC.pt')
# model = YOLO('yolov8m.yaml')
# model = YOLO('runs/detect/train8/weights/best.pt') 
model = YOLO("asset/trained_model/UA-DETRAC/v8n_UA_DETRAC_T-head.pt")
model.export()


if __name__ == '__main__':
    # model.train(data = 'coco_minitrain_10k.yaml', epochs = 500, batch=16, device = 0, project='quantize_training') # quantize_training 0
    # model.train(data = 'coco128.yaml', epochs = 5, device = 0) 
    # print(model.model)
    
    # model.train(data = 'UA-DETRAC.yaml', epochs = 500, batch=16, device = 0) # run 15 detect
    # model.info()
    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=4, device='cpu') 
    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=16) 
    # model.train(data = 'coco128.yaml', epochs=5, project='coco_128')

    # input_tensor = torch.randn(100, 3, 640, 640)
    # t_time = 0
    # for i in range(100):
    # # Suy luận trên dữ liệu
    #     start_time = time.time()
    #     output = model.model(input_tensor[i:i+1])
    #     end_time = time.time()
    #     t_time += end_time - start_time
    # print(t_time/100 * 1000)
    pass
    # print(model.model.model)
    # print(metrics)
    # flops, params = profile(model.model, inputs=torch.randn(1, 1, 3, 640, 640))
    # print(f"FLOPs: {flops}, Params: {params}")
    # print(model)
    # model.info(verbose=True)


# train 10k thuong
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
# python yolov8_KD/KD_training_ver7.py --model yolov8s.yaml --teacher yolov8m_coco10k.pt --data coco_minitrain_10k.yaml --type_kd_loss LD --epochs 500 --batch 16 --device 0 # KD_feat 16
# python yolov8_KD/KD_training_ver7.py --model KD_feature/train17/weights/last.pt --teacher yolov8m_coco10k.pt --type_kd_loss LD --resume True --device 0 # KD_feat 16

# pruning 
# python benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/UA-DETRAC/v8s_UA_DETRAC.pt --data UA-DETRAC.yaml --iterative-steps 5 --epochs 50 --target-prune-rate 0.3 --batch 16 --device 0

# test
# python benchmarks/prunability/yolov8_pruning.py --model yolov8s.pt --data coco128.yaml --iterative-steps 20 --epochs 2 --target-prune-rate 0.8 --batch 32 --max-map-drop 1.0 --device 0


# test quantized model
# python yolov8_QT/qat.py --model yolov8n.yaml --data coco128.yaml --epochs 3 --batch 4 --device 0
    
# quantize 
# python yolov8_QT/ptq.py --model asset/trained_model/UA-DETRAC/v8n_UA_DETRAC.pt --data UA-DETRAC.yaml