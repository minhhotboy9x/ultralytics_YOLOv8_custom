import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
import onnx
from thop import profile
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck, C2fGhost, GhostBottleneck, CBAM
import onnxruntime
import time

# class C2f_v2(nn.Module):
#     # CSP Bottleneck with 2 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv0 = Conv(c1, self.c, 1, 1)
#         self.cv1 = Conv(c1, self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         # y = list(self.cv1(x).chunk(2, 1))
#         y = [self.cv0(x), self.cv1(x)]
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
    
# model = YOLO('yolov8n.pt') 
# model = YOLO('yolov8s.pt')
# model = YOLO('asset/trained_model/UA-DETRAC/pruning/l2/v8s_DETRAC_prune_0.1_local_l2.pt')
# model = YOLO('asset/trained_model/UA-DETRAC/v8s_DETRAC.torchscript')
# model = YOLO('asset/trained_model/UA-DETRAC/pruning/l2/v8s_DETRAC_prune_0.1_local_l2.pt')
# model = YOLO('yolov8n.yaml')
# model = YOLO('runs/detect/train8/weights/best.pt') 
# model = YOLO("asset/trained_model/UA-DETRAC_torchscript/v8n_UA_DETRAC_default_ptq.torchscript")
# model = torch.jit.load("asset/trained_model/UA-DETRAC_torchscript/v8n_UA_DETRAC_default_ptq.torchscript", map_location=torch.device('cpu'))
cv1 = CBAM(3)
if __name__ == '__main__':
    
    # model.train(data = 'coco.yaml', epochs = 500, batch=16, device = 0, project='coco')
    # model.train(data = 'coco128.yaml', epochs = 5, project='coco_128', device = 1) 
    # print(model.model)
    # run 15 detect v8n
    # run 16 detect v8s
    # model.train(data = 'UA-DETRAC.yaml', epochs = 500, lr0 = 0.001, batch=16, device = 1) 
    # model.info()
    # model.export()
    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=4, device='cpu') 
    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=4, device='cpu') 

    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=16, device=1) 
    # print(metrics)
    # model.train(data = 'coco128.yaml', epochs=5, project='coco_128')

    input_tensor = torch.randn(100, 3, 640, 640)
    print(cv1(input_tensor).shape)
    # t_time = 0
    # for i in range(100):
    # # Suy luận trên dữ liệu
    #     start_time = time.time()
    #     output = model2(input_tensor[i:i+1], verbose=False)
    #     end_time = time.time()
    #     t_time += end_time - start_time
    # print(t_time/100 * 1000)

    # print(model.model.model)
    # print(metrics)
    # flops, params = profile(model.model, inputs=torch.randn(1, 1, 3, 640, 640))
    # print(f"FLOPs: {flops}, Params: {params}")
    # print(model)
    # model.info(verbose=True)


# pruning 
# python benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/UA-DETRAC/v8s_DETRAC.pt --data UA-DETRAC.yaml --iterative-steps 2 --epochs 300 --target-prune-rate 0.1 --batch 16 --device 0
# python benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/UA-DETRAC/v8s_DETRAC.pt --data UA-DETRAC.yaml --iterative-steps 2 --epochs 300 --target-prune-rate 0.2 --batch 16 --device 1

# test
# python benchmarks/prunability/yolov8_pruning.py --model yolov8s.pt --data coco128.yaml --iterative-steps 20 --epochs 2 --target-prune-rate 0.8 --batch 32 --max-map-drop 1.0 --device 0


# test quantized model
# python yolov8_QT/qat.py --model yolov8n.yaml --data coco128.yaml --epochs 3 --batch 4 --device 3
    
# quantize 
# python yolov8_QT/ptq.py --model asset/trained_model/UA-DETRAC/v8n_UA_DETRAC.pt --data UA-DETRAC.yaml
    
# ptq
# python yolov8_QT/ptq.py --model asset/trained_model/UA-DETRAC/v8s_DETRAC.pt --data UA-DETRAC.yaml --device 1
