import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
import onnx
from thop import profile
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck, C2fGhost, GhostBottleneck, CBAM, C2f_v2
import onnxruntime
import time

def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)
    
# model = YOLO('coco/train/weights/best.pt') 
# model = YOLO('yolov8n_relu.pt') 
# model = YOLO('runs/detect/train21/weights/best.pt') 
# model = YOLO('asset/trained_model/UA-DETRAC2/v8s_relu_DETRAC2.pt')
# model = YOLO('pruning/train7/step_0_finetune/weights/best.pt')
# model = YOLO('runs/detect/train18/weights/last.pt')
# model = YOLO('asset/trained_model/UA-DETRAC/pruning/l2/v8s_silu_DETRAC_prune_0.1_global_l2.pt')
# model = YOLO('yolov8s.yaml')
# model = YOLO('pruning/train6/step_0_finetune/weights/best.pt') 
model = YOLO('coco/train2/weights/last.pt') 
# model = YOLO("asset/trained_model/UA-DETRAC_torchscript/v8n_UA_DETRAC_default_ptq.torchscript")
# model = torch.jit.load("asset/trained_model/UA-DETRAC/v8s_relu_DETRAC.torchscript", map_location=torch.device('cpu'))
# cv1 = CBAM(3)
if __name__ == '__main__':
    # model.train(data = 'coco.yaml', epochs = 500, batch=32, device = 1, project='coco')
    model.train(data = 'coco.yaml', resume=True, device =1)
    # model.train(data = 'coco128.yaml', epochs = 5, project='coco_128', device = 1) 
    
    # print(model)
    # run 15 detect v8n
    # run 16 detect v8s
    # model.train(data = 'UA-DETRAC.yaml', epochs = 300, lr0 = 1e-5, batch=32, device = 1) 
    # model.train(data = 'UA-DETRAC.yaml', resume=True, device=1) 
    # model.info()
    # model.export()
    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=4, device='cpu') 
    # metrics = model.val(data = 'coco_minitrain_10k.yaml', batch=32, device='3') 

    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=32, device=0, split='val')
    # print(metrics) 
    # replace_c2f_with_c2f_v2(model.model)
    # print(model.model)
    # model.train(data = 'UA-DETRAC.yaml', epochs = 300, lr0 = 1e-4, batch=32, device = 1) 
    # metrics = model.val(data = 'UA-DETRAC.yaml', batch=32, device=2, split='test') 
    # print(metrics)
    # model.train(data = 'coco128.yaml', epochs=5, project='coco_128')
    # model.model.eval()
    # model2.model.eval()
    # input_tensor = torch.randn(100, 3, 640, 640)
    # t_time = 0
    # for i in range(100):
    # # Suy luận trên dữ liệu
    #     start_time = time.time()
    #     output = model2(input_tensor[i:i+1], verbose=False, device='cpu')
    #     end_time = time.time()
    #     t_time += end_time - start_time
    # print(t_time/100 * 1000)

    # t_time = 0
    # for i in range(100):
    # # Suy luận trên dữ liệu
    #     start_time = time.time()
    #     output = model(input_tensor[i:i+1], verbose=False, device='cpu')
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
# python benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/UA-DETRAC2/v8s_relu_DETRAC2.pt --data UA-DETRAC.yaml --iterative-steps 3 --epochs 200 --target-prune-rate 0.5 --batch 32 --device 0
# python benchmarks/prunability/yolov8_pruning.py --model asset/trained_model/UA-DETRAC2/v8s_relu_DETRAC2.pt --data UA-DETRAC.yaml --iterative-steps 3 --epochs 100 --target-prune-rate 0.5 --batch 32 --device 1

# test
# python benchmarks/prunability/yolov8_pruning.py --model yolov8s.pt --data coco128.yaml --iterative-steps 20 --epochs 2 --target-prune-rate 0.8 --batch 4 --max-map-drop 1.0 --sparse-training --device 1


# test quantized model
# python yolov8_QT/qat.py --model yolov8n.yaml --data coco128.yaml --epochs 3 --batch 4 --device 3
    
# quantize 
# python yolov8_QT/ptq.py --model asset/trained_model/UA-DETRAC/v8n_UA_DETRAC.pt --data UA-DETRAC.yaml
    
# ptq
# python yolov8_QT/ptq.py --model asset/trained_model/UA-DETRAC2/v8s_relu_DETRAC2.pt --data UA-DETRAC.yaml --device 1
