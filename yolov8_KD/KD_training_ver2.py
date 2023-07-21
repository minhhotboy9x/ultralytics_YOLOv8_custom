import subprocess
import time
import argparse
import math
import os, sys

ultralytics_dir = os.path.abspath("./")
# Thêm đường dẫn của folder cha vào sys.path
sys.path.append(ultralytics_dir)

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import initialize_weights, de_parallel
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.v8.detect.train import DetectionModel
from ultralytics.yolo.cfg import get_cfg
from tqdm import tqdm

def cal_kd_loss(self: BaseTrainer, student_preds, teacher_preds, T=3):
    m = self.model.model[-1]
    nc = m.nc  # number of classes
    no = m.no  # number of outputs per anchor
    reg_max = m.reg_max
 
    stu_pred_distri, stu_pred_scores = torch.cat([xi.view(student_preds[0].shape[0], no, -1) for xi in student_preds], 2).split(
            (reg_max * 4, nc), 1)

    stu_pred_scores = stu_pred_scores.permute(0, 2, 1).contiguous() # batch, anchors, channels
    stu_pred_distri = stu_pred_distri.permute(0, 2, 1).contiguous() # batch, anchors, channels
    b, a, c = stu_pred_distri.shape  # batch, anchors, channels

    tea_pred_distri, tea_pred_scores = torch.cat([xi.view(teacher_preds[0].shape[0], no, -1) for xi in student_preds], 2).split(
            (reg_max * 4, nc), 1)

    tea_pred_scores = tea_pred_scores.permute(0, 2, 1).contiguous() # batch, anchors, channels
    tea_pred_distri = tea_pred_distri.permute(0, 2, 1).contiguous() # batch, anchors, channels
    #-------------------------------KD loss----------------------------------------
    kl_loss = nn.KLDivLoss(reduction="mean")

    stu_pred_scores = (stu_pred_scores/T).sigmoid()
    stu_pred_distri = (stu_pred_distri/T).view(b, a, 4, c // 4).softmax(3)

    tea_pred_scores = (tea_pred_scores/T).sigmoid()
    tea_pred_distri = (tea_pred_distri/T).view(b, a, 4, c // 4).softmax(3)

    cl_loss = kl_loss(stu_pred_scores, tea_pred_scores)
    box_loss = kl_loss(stu_pred_distri, tea_pred_distri)

    return 0.8*box_loss + 0.2*cl_loss

def _do_train_v2(self: BaseTrainer, world_size=1):
    """Train completed, evaluate and plot if specified by arguments."""
    if world_size > 1:
        self._setup_ddp(world_size)

    self._setup_train(world_size)
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    nb = len(self.train_loader)  # number of batches
    nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
    last_opt_step = -1
    # self.run_callbacks('on_train_start')
    LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f'Starting training for {self.epochs} epochs...')
    if self.args.close_mosaic:
        base_idx = (self.epochs - self.args.close_mosaic) * nb
        self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])


    # create knowledge distillation function
    if hasattr(self, 'teacher'):
        dump_image = torch.randn((1, 3, self.args.imgsz, self.args.imgsz), device=self.device)

        preds = self.model(dump_image)  # forward
        teacher_preds = self.teacher(dump_image) 
        self.cal_kd_loss = cal_kd_loss.__get__(self)
        # print(self.cal_kd_loss(preds, teacher_preds)) # test kl loss
        

    for epoch in range(self.start_epoch, self.epochs):
        self.epoch = epoch
        self.run_callbacks('on_train_epoch_start')
        self.model.train()
        if hasattr(self, 'teacher'):
            self.teacher.eval() 

        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        # Update dataloader attributes (optional)
        if epoch == (self.epochs - self.args.close_mosaic):
            LOGGER.info('Closing dataloader mosaic')
            if hasattr(self.train_loader.dataset, 'mosaic'):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, 'close_mosaic'):
                self.train_loader.dataset.close_mosaic(hyp=self.args)
        if RANK in (-1, 0):
            LOGGER.info(self.progress_string())
            pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
        self.tloss = None
        self.optimizer.zero_grad()
        for i, batch in pbar:
            self.run_callbacks('on_train_batch_start')
            # Warmup
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

            # Forward
            with torch.cuda.amp.autocast(self.amp):
                batch = self.preprocess_batch(batch)
                self.kd_loss = 0.0 # make a kd_loss

                if hasattr(self, 'teacher'):
                    preds =  self.model(batch['img'])
                    teacher_preds =  self.teacher(batch['img'])
                    self.kd_loss = self.cal_kd_loss(preds, teacher_preds)
                else:
                    preds = self.model(batch['img']) # predict

                self.loss, self.loss_items = self.criterion(preds, batch) # cal loss

                # print(f'--------------{self.kd_loss}---------------')

                if RANK != -1:
                    self.loss *= world_size
                self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                    else self.loss_items

            # Backward
            self.scaler.scale(self.loss + 0.2 * self.kd_loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= self.accumulate:
                self.optimizer_step()
                last_opt_step = ni

            # Log
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
            losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
            if RANK in (-1, 0):
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                    (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                self.run_callbacks('on_batch_end')
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)

            self.run_callbacks('on_train_batch_end')

        self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

        self.scheduler.step()
        self.run_callbacks('on_train_epoch_end')

        if RANK in (-1, 0):

            # Validation
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

            if self.args.val or final_epoch:
                self.metrics, self.fitness = self.validate()
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop = self.stopper(epoch + 1, self.fitness)

            # Save model
            if self.args.save or (epoch + 1 == self.epochs):
                self.save_model()
                self.run_callbacks('on_model_save')

        tnow = time.time()
        self.epoch_time = tnow - self.epoch_time_start
        self.epoch_time_start = tnow
        self.run_callbacks('on_fit_epoch_end')
        torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

        # Early Stopping
        if RANK != -1:  # if DDP training
            broadcast_list = [self.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                self.stop = broadcast_list[0]
        if self.stop:
            break  # must break all DDP ranks

    if RANK in (-1, 0):
        # Do final val with best.pt
        LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                    f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
        self.final_eval()
        if self.args.plots:
            self.plot_metrics()
        self.run_callbacks('on_train_end')
    torch.cuda.empty_cache()
    self.run_callbacks('teardown')

def trainer_train_v2(self: BaseTrainer):
    """Allow device='', device=None on Multi-GPU systems to default to device=0."""
    if isinstance(self.args.device, int) or self.args.device:  # i.e. device=0 or device=[0,1,2,3]
        world_size = torch.cuda.device_count()
    elif torch.cuda.is_available():  # i.e. device=None or device=''
        world_size = 1  # default to device 0
    else:  # i.e. device='cpu' or 'mps'
        world_size = 0

    # Run subprocess if DDP training, else train normally
    if world_size > 1 and 'LOCAL_RANK' not in os.environ:
        # Argument checks
        if self.args.rect:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting rect=False")
            self.args.rect = False
        # Command
        cmd, file = generate_ddp_command(world_size, self)
        try:
            LOGGER.info(f'Running DDP command {cmd}')
            subprocess.run(cmd, check=True)
        except Exception as e:
            raise e
        finally:
            ddp_cleanup(self, str(file))
    else:
        self._do_train_v2(world_size)

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

    self.trainer.hub_session = self.session  # attach optional HUB session

    # get teacher for KD by adding teacher attribute for trainer of model
    if kwargs.get('teacher'):
        self.trainer.teacher = YOLO(kwargs['teacher']).model.to(self.trainer.device)
        

    self.trainer.train_v2 = trainer_train_v2.__get__(self.trainer)
    self.trainer._do_train_v2 = _do_train_v2.__get__(self.trainer)
    self.trainer.train_v2()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)


def main(args):

    args = get_cfg(DEFAULT_CFG, vars(args))
    model = YOLO(args.model)
    model.train_v2 = train_v2.__get__(model)
    model.train_v2(**vars(args))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8m.yaml', help='Pretrained pruning target model file')
    parser.add_argument('--teacher', help='teacher model') # kd output of model
    
    parser.add_argument('--batch', default=4, type=int, help='batch_size')
    parser.add_argument('--data', default='coco128.yaml', help='dataset')
    parser.add_argument('--device', default=0, help='cpu or gpu')
    parser.add_argument('--project', default='KD', help='project name')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Size of input images')
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads for data loading (per RANK if DDP)")
    parser.add_argument('--resume', type=bool, default=False, help="continue training (if KD, must provide teacher)")

    args = parser.parse_args()


    main(args)
    

    


    
    

