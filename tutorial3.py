import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.cfg import get_cfg
import onnx
from torch.onnx import export
import onnxruntime

if __name__ == "__main__":
    model_yolo = YOLO('asset/trained_model/UA-DETRAC/v8m_UA_DETRAC.pt')
    model = torch.jit.load('asset/trained_model/UA-DETRAC/v8n_UA_DETRAC_default_ptq.torchscript')
    # print(model)
    input = torch.randn(1, 3, 640, 640)
    export(model, input, "yolov8s.onnx", opset_version=17)

    print(model(input).shape)    
    print(model_yolo.model(input)[0].shape)    

    input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
    ort_session = onnxruntime.InferenceSession("yolov8s.onnx")
    onnx_model = onnx.load("yolov8s.onnx")

    # List all input names
    input_names = [input.name for input in onnx_model.graph.input]
    print("Input names:", input_names)

    output = ort_session.run(None, {'x.3': input_data})
    print(output[0].shape)
    # yolo_obj = YOLO(r'quantizing\train3\weights\quantized_model_jit.torchscript', task='detect')
    # yolo_obj.val(data = 'coco128.yaml', device='cpu')