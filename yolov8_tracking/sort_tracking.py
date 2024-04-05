import torch
import os, sys
ultralytics_dir = os.path.abspath("../")
sys.path.append(ultralytics_dir)
from PIL import Image, ImageDraw
from ultralytics import YOLO
# from sort import *
# image = Image.open('MVI_40244_img00675.jpg')
# print(image.size)
SIZE = (960, 540)
# tracker = Sort()


image = Image.open('MVI_40793_img01356.jpg')

# Tạo đối tượng ImageDraw để vẽ lên ảnh
draw = ImageDraw.Draw(image)


# model = YOLO('/home/minhnq/ultralytics/pruning/train5/step_4_finetune/weights/best.onnx', task='detect')
model = YOLO('/home/minhnq/ultralytics/runs/detect/train17/weights/best.pt', task='detect')
# model.export(format = 'onnx')

results = model.predict('MVI_40793_img01356.jpg', device=2)
# Mở ảnh sử dụng thư viện Pillow


for result in results:
    print('----------')
    res = result.boxes
    
    for cls, box_xyxy, box_xywh in zip(res.cls.tolist(), res.xyxy.tolist(), res.xywh.tolist()):
        # x = SIZE[0]*box[0]
        # w = SIZE[0]*box[2]
        # y = SIZE[1]*box[1]
        # h = SIZE[1]*box[3]
        # print(box_xyxy)
        box_xy = box_xyxy[:2]
        box_wh = box_xywh[2:]
        print(cls, box_xy, box_wh)
        draw.rectangle([tuple(box_xy[0:2]), (box_xy[0]+box_wh[0], box_xy[1]+box_wh[1])], outline="red")

image.save('res.jpg')
# print(result)