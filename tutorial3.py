from ultralytics import YOLO

# model = YOLO("runs/detect/train3/weights/last.pt")  # or a segmentation model .i.e yolov8n-seg.pt
# model = YOLO("yolov8m.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model = YOLO("yolov8s.pt")  # or a segmentation model .i.e yolov8n-seg.pt

#inputs = 'datasets/AIC-HCMC-2020/images/val'  # list of numpy arrays
# results = model.val('ultralytics/datasets/AIC-HCMC-2020.yaml')  # generator of Results objects
# python yolov8_KD/KD_training_ver2.py --model KD/train2/weights/last.pt --teacher yolov8m.pt --resume True
# python yolov8_KD/KD_training_ver2.py --model yolov8s.pt --teacher yolov8m.pt --data coco.yaml --batch 8 --epochs 200

if __name__ == "__main__":
    # model.train(data='coco_minitrain.yaml', batch=4, epochs=200, imgsz=640, device=1, workers=2)
    # model.train(data='AIC-HCMC-2020.yaml', batch=4, epochs=200, imgsz=640, device=0, workers=2) #train 4 8m
    # model.train(data='AIC-HCMC-2020.yaml', batch=4, epochs=200, imgsz=640, device=1, workers=2) #train 5 8s
    model.val(data='coco_minitrain.yaml')
    # model.train(resume=True)