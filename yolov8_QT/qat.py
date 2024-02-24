import subprocess
import time
import argparse
import math
import os, sys
from pathlib import Path
from typing import List, Union

ultralytics_dir = os.path.abspath("./")
# Thêm đường dẫn của folder cha vào sys.path
sys.path.append(ultralytics_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics import YOLO, __version__
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.engine.model import TASK_MAP
from torch.optim import lr_scheduler
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.engine.trainer import check_amp
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.nn.modules_quantized import Q_Conv
from ultralytics.yolo.cfg import get_cfg
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from utils import attempt_load_one_weight_v2, replace_conv_with_qconv_v2_qat

def _setup_train_v2(self: BaseTrainer, world_size):
    """
    Builds dataloaders and optimizer on correct rank process.
    """
    # QAT setup
    if '.pt' in self.args.model:
        replace_conv_with_qconv_v2_qat(self.model)
    else:
        self.model.eval()
        for m in self.model.modules():
            if type(m) is Q_Conv:
                torch.ao.quantization.fuse_modules(m, [["conv", "bn", "act"]], True)
    torch.quantization.prepare_qat(self.model.train(), inplace=True)

    # Model
    self.run_callbacks('on_pretrain_routine_start')
    ckpt = self.setup_model()
    self.model = self.model.to(self.device)
    self.set_model_attributes()
    # Check AMP
    self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
    if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
        callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
        self.amp = torch.tensor(check_amp(self.model), device=self.device)
        callbacks.default_callbacks = callbacks_backup  # restore callbacks
    if RANK > -1:  # DDP
        dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
    self.amp = bool(self.amp)  # as boolean
    self.scaler = amp.GradScaler(enabled=self.amp)
    if world_size > 1:
        self.model = DDP(self.model, device_ids=[RANK])
    # Check imgsz
    gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
    self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
    # Batch size
    if self.batch_size == -1:
        if RANK == -1:  # single-GPU only, estimate best batch size
            self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
        else:
            SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                        'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

    # Optimizer
    self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
    weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
    self.optimizer = self.build_optimizer(model=self.model,
                                            name=self.args.optimizer,
                                            lr=self.args.lr0,
                                            momentum=self.args.momentum,
                                            decay=weight_decay)
    # Scheduler
    if self.args.cos_lr:
        self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
    else:
        self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
    self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
    self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

    # Dataloaders
    batch_size = self.batch_size // world_size if world_size > 1 else self.batch_size
    self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
    if RANK in (-1, 0):
        self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
        self.validator = self.get_validator()
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
        self.ema = ModelEMA(self.model)
        if self.args.plots and not self.args.v5loader:
            self.plot_training_labels()
    self.resume_training(ckpt)
    self.scheduler.last_epoch = self.start_epoch - 1  # do not move
    self.run_callbacks('on_pretrain_routine_end')

def save_model_v2(self: BaseTrainer, quantized_model):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(quantized_model),
        # 'ema': deepcopy(self.ema.ema),
        # 'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    traced_model = torch.jit.trace(deepcopy(quantized_model), torch.rand(1, 3, self.args.imgsz, self.args.imgsz))
    path_to_quantized = self.best.parent / "quantized_model_jit.torchscript"
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
        torch.jit.save(traced_model, path_to_quantized)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt

def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/yolo/engine/trainer.py
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Disabled half precision saving. originated from ultralytics/yolo/utils/torch_utils.py
    """
    x = torch.load(f, map_location=torch.device('cpu'))
    args = {**DEFAULT_CFG_DICT, **x['train_args']}  # combine model args with default args, preferring model args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    # for p in x['model'].parameters():
    #     p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def _do_train_v2(self, world_size=1):
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

    for epoch in range(self.start_epoch, self.epochs):
        self.epoch = epoch
        self.run_callbacks('on_train_epoch_start')
        self.model.train()
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
            # with torch.cuda.amp.autocast(self.amp):
                # print(f'----------------{batch["img"].shape}-------------------')
            batch = self.preprocess_batch(batch)
            preds = self.model(batch['img']) # predict
            self.loss, self.loss_items = self.criterion(preds, batch) # cal loss

            if RANK != -1:
                self.loss *= world_size
            self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                else self.loss_items

            # Backward
            self.scaler.scale(self.loss).backward()

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
            quantized_model = torch.quantization.convert(deepcopy(self.model).cpu())
            if self.args.val or final_epoch:
                self.metrics, self.fitness = self.validate_v2(quantized_model)

            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop = self.stopper(epoch + 1, self.fitness)

            # Save model
            if self.args.save or (epoch + 1 == self.epochs):
                self.save_model_v2(quantized_model)
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
        # self.final_eval_v2()
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

def val_v2(self: YOLO, data=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides['rect'] = True  # rect batches as default
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)
        validator = TASK_MAP[self.task][2](args=args, _callbacks=self.callbacks)
        metrics = validator(model=self.model) # result need to get
        self.metrics = validator.metrics
        return metrics

def validate_v2(self: BaseTrainer, model):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        tmp_model = YOLO(self.args.model)
        tmp_model.val_v2 = val_v2.__get__(tmp_model)
        tmp_model.model = deepcopy(model)
        tmp_model.model.args = vars(tmp_model.model.args)
        metrics = tmp_model.val_v2(data = tmp_model.model.args['data'], device = 'cpu')
        fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

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

    self.trainer.train_v2 = trainer_train_v2.__get__(self.trainer)
    self.trainer._do_train_v2 = _do_train_v2.__get__(self.trainer)
    self.trainer.validate_v2 = validate_v2.__get__(self.trainer)
    self.trainer._setup_train = _setup_train_v2.__get__(self.trainer)
    self.trainer.save_model_v2 = save_model_v2.__get__(self.trainer)
    self.trainer.final_eval_v2 = final_eval_v2.__get__(self.trainer)
    self.trainer.train_v2()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight_v2(str(self.trainer.best)) # không load lại best model
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)

def main(args):
    args = get_cfg(DEFAULT_CFG, vars(args))
    model = YOLO(args.model)
    model.train_v2 = train_v2.__get__(model)
    model.train_v2(**vars(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.pt', help='Pretrained pruning target model file')
    parser.add_argument('--batch', default=4, type=int, help='batch_size')
    parser.add_argument('--data', default='coco128.yaml', help='dataset')
    parser.add_argument('--device', default=0, help='cpu or gpu')
    parser.add_argument('--project', default='quantizing', help='project name')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Size of input images')
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads for data loading (per RANK if DDP)")
    parser.add_argument('--resume', type=bool, default=False, help="continue training (if KD, must provide teacher)")
    parser.add_argument('--lr0', type=float, default=0.0001, help="initial learning rate (i.e. SGD=1E-2, Adam=1E-3)")  
    
    args = parser.parse_args()


    main(args)