import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.cfg import get_cfg

def init(self: YOLO,  **kwargs):
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
    if not overrides.get('resume'):  # manually set model only if not resuming
        self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
        self.model = self.trainer.model

def train_v2(self: YOLO,  **kwargs):
    """
    Disabled loading new model when pruning flag is set. originated from ultralytics/yolo/engine/model.py
    """

    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)

    if not overrides.get('resume'):  # manually set model only if not resuming
        self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
        self.model = self.trainer.model

    # self.trainer.hub_session = self.session  # attach optional HUB session

    # get teacher for KD by adding teacher attribute for trainer of model
    if kwargs.get('teacher'):
        # kwargs2 = kwargs.copy()
        # kwargs2['model'] = kwargs['teacher']
        teacher_model = YOLO(kwargs['teacher'])
        # teacher_model.init = init.__get__(teacher_model)
        # teacher_model.init(**kwargs2)
        self.trainer.teacher = teacher_model.model


# model = YOLO("runs/detect/train3/weights/last.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model1 = YOLO("yolov8m.pt")  # or a segmentation model .i.e yolov8n-seg.pt
# model2 = YOLO("yolov8m.pt")  # or a segmentation model .i.e yolov8n-seg.pt

#inputs = 'datasets/AIC-HCMC-2020/images/val'  # list of numpy arrays
# results = model.val('ultralytics/datasets/AIC-HCMC-2020.yaml')  # generator of Results objects
# python yolov8_KD/KD_training_ver2.py --model KD/train2/weights/last.pt --teacher yolov8m.pt --resume True
# python yolov8_KD/KD_training_ver2.py --model yolov8s.pt --teacher yolov8m.pt --data coco.yaml --batch 8 --epochs 200


if __name__ == "__main__":
    a = torch.tensor([-11.0])
    b = torch.tensor([-5.8]).sigmoid()
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    print(bce_loss(a, b))
    # model.val(data='coco_minitrain.yaml')
    # model.train(resume=True)
    # dump_image = torch.randn((1, 3, 640, 640))
    # model1.train_v2 = train_v2.__get__(model1)
    # model2.train_v2 = train_v2.__get__(model2)

    # args = get_cfg(DEFAULT_CFG)
    # args.data = 'coco128.yaml'
    # args.teacher = 'yolov8s.pt'
    # model1.train_v2(**vars(args))
    # model2.train_v2(**vars(args))