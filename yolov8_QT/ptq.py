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

def preprocess_batch(batch, device): #preprocess_batch trong từng ảnh
    """Preprocesses a batch of images by scaling and converting to float."""
    batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
    return batch

def calibration(model, args):
    data = check_det_dataset(args.data)
    trainset = data['train']
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gs = max(int(de_parallel(model).stride.max() if model else 0), 32)
    train_loader = build_dataloader(args, args.batch, img_path=trainset, stride=gs, rank=RANK, mode='train',
                             rect=False, data_info=data)[0]
    nb = len(train_loader)
    print('--------Calibration start--------')
    pbar = tqdm(enumerate(train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
    for i, batch in pbar:
        batch = preprocess_batch(batch, device)
        model(batch['img'])
    print('--------Calibration done---------')

def main(args):
    args = get_cfg(DEFAULT_CFG, vars(args))
    model = YOLO(args.model, task='detect')
    detect_model = model.model
    # print('before replace:', detect_model.model[0].conv.weight[0, :, :, :])
    # print(model)
    replace_conv_with_qconv_v2_ptq(detect_model)
    torch.quantization.prepare(detect_model, inplace=True)
    calibration(detect_model, args)
    detect_model = detect_model.cpu()
    torch.quantization.convert(detect_model, inplace=True)
    metrics = model.val(data=args.data, batch=args.batch, device='cpu')
    print('-----------------------')
    model.info()
    print(metrics)
    model.export()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.pt', help='Pretrained pruning target model file')
    parser.add_argument('--batch', default=4, type=int, help='batch_size')
    parser.add_argument('--data', default='coco128.yaml', help='dataset')
    parser.add_argument('--device', default='cpu', help='cpu or gpu')
    parser.add_argument('--imgsz', type=int, default=640, help='Size of input images')
    
    args = parser.parse_args()
    
    main(args)