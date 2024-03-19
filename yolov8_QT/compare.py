import torch
import torch.nn as nn
import os, sys
import torch.onnx
import argparse

ultralytics_dir = os.path.abspath("./")
# Thêm đường dẫn của folder cha vào sys.path
sys.path.append(ultralytics_dir)

from ultralytics.yolo.utils import yaml_load, RANK, TQDM_BAR_FORMAT
from ultralytics import YOLO

from ultralytics.nn.modules import Conv, Bottleneck
from ultralytics.yolo.cfg import get_cfg, DEFAULT_CFG
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.data import build_dataloader
from tqdm import tqdm
from utils import replace_conv_with_qconv_v2_ptq
from itertools import islice

def preprocess_batch(batch, device): #preprocess_batch trong từng ảnh
    """Preprocesses a batch of images by scaling and converting to float."""
    batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
    return batch

def main(args):
    args = vars(args)
    model = YOLO(args.model, task='detect')
    qmodel = YOLO(args.model, task='detect')
    replace_conv_with_qconv_v2_ptq(qmodel)
    qmodel.load_state_dict(torch.load(args.qmodel))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.pt', help='Pretrained pruning target model file')
    parser.add_argument('--qmodel', default='yolov8n.pth', help='Quantized torch module model')
    parser.add_argument('--batch', default=8, type=int, help='batch_size')
    parser.add_argument('--data', default='coco128.yaml', help='dataset')
    parser.add_argument('--device', default=0, help='cpu or gpu')
    parser.add_argument('--imgsz', type=int, default=640, help='Size of input images')
    
    args = parser.parse_args()
    
    main(args)