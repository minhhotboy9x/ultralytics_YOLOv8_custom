import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
import onnx
from thop import profile
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck, C2fGhost, GhostBottleneck, CBAM, C2f_v2
import onnxruntime
import time

    
model = YOLO('coco/train4/weights/last.pt') 
# model = YOLO('yolov8n.yaml')
# model = YOLO('yolov8n_relu.pt') 
# model = YOLO('pruning/train19/step_46_finetune4/weights/best.pt') 
# model = YOLO('asset/trained_model/VOC/v8s_relu_fullghost_VOC.pt') 
# model = YOLO("KD_feature/train16/weights/best.pt")
# model = YOLO("asset/trained_model/VOC/v8s_relu_localprune_bftrain_VOC.pt")
# model = torch.jit.load("asset/trained_model/UA-DETRAC2/v8n_relu.torchscript", map_location=torch.device('cpu'))
# cv1 = CBAM(3)
if __name__ == '__main__':
    # model.train(data = 'coco.yaml', epochs = 500, batch=32, device = 0, project='coco')
    model.train(data = 'coco.yaml', resume=True, device = 1)
    # model.train(data = 'coco128.yaml', epochs = 5, project='coco_128', device = 1) 
    
    # print(model.model)
    # run 15 detect v8n
    # run 16 detect v8s
    # model.train(data = 'VOC.yaml', epochs = 300, batch=32, device = 1) 
    # model.train(data = 'coco_vehicle.yaml', epochs = 300, batch=32, device = 0) 
    # model.train(data = 'VOC.yaml', resume=True, device=1) 

    # model.fuse()
    # model2.info()
    # model.export(format='onnx')
    # metrics = model.val(data = 'DETRAC_fix_roi.yaml', batch=64, device=2, split='test') 
    # metrics = model.val(data = 'coco_vehicle.yaml', batch=64, device=2, split='test') 
    # metrics = model.val(data = 'VOC.yaml', batch=32, device=2, split='test') 
    # metrics = model.val(data = 'coco_minitrain_10k.yaml', batch=32, device='3') 

    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=32, device=0, split='val')
    # print(metrics) 
    # replace_c2f_with_c2f_v2(model.model)
    # print(model.model)
    # model.train(data = 'UA-DETRAC.yaml', epochs = 300, lr0 = 1e-4, batch=32, device = 1) 
    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=64, device=0, split='test') 
    # print(metrics)
    # model.train(data = 'coco128.yaml', epochs=5, project='coco_128')
    # model.model.eval()
    # model2.model.eval()
    # input_tensor = torch.randn(1, 3, 640, 640)
    # model(input_tensor, verbose=False, device=0)
    # model2(input_tensor, verbose=False, device=0)
    # t_time = 0
    # for i in range(1000):
    # # Suy luận trên dữ liệu
    #     start_time = time.time()
    #     output = model2(input_tensor, verbose=False, device=0)
    #     end_time = time.time()
    #     t_time += end_time - start_time
    # print(t_time)

    # t_time = 0
    # for i in range(1000):
    # # Suy luận trên dữ liệu
    #     start_time = time.time()
    #     output = model(input_tensor, verbose=False, device=2)
    #     end_time = time.time()
    #     t_time += end_time - start_time
    # print(t_time)

    # print(model.model.model)
    # print(metrics)
    # flops, params = profile(model.model, inputs=torch.randn(1, 1, 3, 640, 640))
    # print(f"FLOPs: {flops}, Params: {params}")
    # print(model)
    # model.info(verbose=True)


# pruning 
# python benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/VOC/v8s_relu_VOC.pt --data VOC.yaml --iterative-steps 2 --epochs 200 --target-prune-rate 0.2 --sparse-training True --batch 16 --device 3
# python benchmarks/prunability/yolov8_pruning.py --model yolov8s.yaml --data VOC.yaml --iterative-steps 1 --epochs 300 --target-prune-rate 0.3 --lr0 0.01 --batch 16 --device 3
# python benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/UA-DETRAC2/v8s_relu_c2fv2_DETRAC2.pt --data UA-DETRAC.yaml --iterative-steps 3 --epochs 200 --target-prune-rate 0.3 --batch 32 --device 1
# python benchmarks/prunability/yolov8_pruning_taylor.py --model asset/trained_model/VOC/v8s_relu_VOC.pt --data VOC.yaml --iterative-steps 2 --epochs 200 --target-prune-rate 0.2 --batch 32 --device 3


# test
# python benchmarks/prunability/yolov8_pruning_taylor.py --model yolov8s.pt --data coco128.yaml --iterative-steps 20 --epochs 2 --target-prune-rate 0.8 --batch 4 --max-map-drop 1.0 --device 1


# test quantized model
# python yolov8_QT/qat.py --model yolov8n.yaml --data coco128.yaml --epochs 3 --batch 4 --device 3
    
# quantize 
# python yolov8_QT/ptq.py --model asset/trained_model/UA-DETRAC/v8n_UA_DETRAC.pt --data UA-DETRAC.yaml
    
# ptq
# python yolov8_QT/ptq.py --model asset/trained_model/UA-DETRAC2/v8s_relu_DETRAC2.pt --data UA-DETRAC.yaml --device 1

# KD
# python yolov8_KD/KD_training_ver7.py --model asset/trained_model/coco_vehicle/v8s_relu_ghostneck_cocovehicle.pt --teacher asset/trained_model/coco_vehicle/v8s_relu_cocovehicle.pt --data coco_vehicle.yaml --batch 4 --lr0 1e-3 --epochs 200 --device 1