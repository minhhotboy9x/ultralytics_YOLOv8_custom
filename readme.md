# YOLOv8 compressed model

## Pruning (inherited from [Pruning](benchmarks/prunability/readme.md))
### Prune model from scratch: 

```
python benchmarks/prunability/yolov8_pruning_bftrain.py --model yolov8s.yaml --data VOC.yaml --iterative-steps 1 --epochs 300 --target-prune-rate 0.4 --lr0 0.01 --batch 16 --device 0
```

### Prune model from pretrained model: 
```
python benchmarks/prunability/yolov8_pruning.py --model yolov8s.pt --data coco.yaml --iterative-steps 1 --epochs 300 --target-prune-rate 0.2 --lr0 0.01 --batch 32 --device 0
```

## Quantization (ONNX INT8 running on CPU)

### PTQ
```
python yolov8_QT/ptq.py --model yolov8s.pt --data coco.yaml --batch 8
```

### QAT
```
python yolov8_QT/qat.py --model yolov8s.pt --data coco.yaml --epochs 5 --batch 8
```

## Knowledge distillation (In progress, you can custom more in [yolov8_kd](yolov8_KD))
