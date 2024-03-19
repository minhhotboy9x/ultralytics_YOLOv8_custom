# This code is adapted from Issue [#147](https://github.com/VainF/Torch-Pruning/issues/147), implemented by @Hyunseok-Kim0.
import argparse
import math
import os, sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union

# Xác định đường dẫn của folder cha
ultralytics_dir = os.path.abspath("../ultralytics")
# Thêm đường dẫn của folder cha vào sys.path
sys.path.append(ultralytics_dir)


import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck, C2f_v2
from ultralytics.nn.modules_deformable import TDetect
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, TQDM_BAR_FORMAT, colorstr
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import initialize_weights, de_parallel

import torch_pruning as tp


def save_pruning_performance_graph(x, y1, y2, y3, dir=""):
    """
    Draw performance change graph
    Parameters
    ----------
    x : List
        Parameter numbers of all pruning steps
    y1 : List
        mAPs after fine-tuning of all pruning steps
    y2 : List
        MACs of all pruning steps
    y3 : List
        mAPs after pruning (not fine-tuned) of all pruning steps

    Returns
    -------

    """
    try:
        plt.style.use("ggplot")
    except:
        pass

    x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
    y2_ratio = y2 / y2[0]

    # create the figure and the axis object
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP')
    ax.plot(x, y1, label='recovered mAP')
    ax.scatter(x, y1)
    ax.plot(x, y3, color='tab:gray', label='pruned mAP')
    ax.scatter(x, y3, color='tab:gray')

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('MACs')
    ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
    ax2.scatter(x, y2_ratio, color='tab:orange')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(105, -5)
    ax.set_ylim(0, max(y1) + 0.05)
    ax2.set_ylim(0.05, 1.05)

    # calculate the highest and lowest points for each set of data
    max_y1_idx = np.argmax(y1)
    min_y1_idx = np.argmin(y1)
    max_y2_idx = np.argmax(y2)
    min_y2_idx = np.argmin(y2)
    max_y1 = y1[max_y1_idx]
    min_y1 = y1[min_y1_idx]
    max_y2 = y2_ratio[max_y2_idx]
    min_y2 = y2_ratio[min_y2_idx]

    # add text for the highest and lowest values near the points
    ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
    ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
    ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
    ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

    plt.title('Comparison of mAP and MACs with Pruning Ratio')
    plt.savefig(os.path.join(dir, f'pruning_perf_change.png'))


def save_pruning_size_graph(x, y1, y2, dir=""): # sparities_list, no_params_list, flops_list
    try:
        plt.style.use("ggplot")
    except:
        pass

    x, y1, y2= np.array(x), np.array(y1), np.array(y2)
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Channel sparsity')
    ax.set_ylabel('#Params (M)')
    ax.plot(x, y1, label='Remaining #params')
    ax.scatter(x, y1)

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('FLOPs (G)')
    ax2.plot(x, y2, color='tab:blue', label='Remaining FLOPs')
    ax2.scatter(x, y2, color='tab:blue')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(1, 0)
    ax.set_ylim(0, max(y1) + 1.5)
    ax2.set_ylim(0, max(y2) + 1.5)

    plt.title('Comparison of FLOPs and #Params with Pruning Ratio')
    plt.savefig(os.path.join(dir, f'pruning_size_change.png'))

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
            if self.pruner:
                self.pruner.update_regularizor()

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
                    if self.pruner:
                        self.pruner.regularize(self.model)
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

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

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


def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
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
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")

def train_v2(self: YOLO, pruning=False, pruner=None, **kwargs):
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

    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model
        self.trainer.pruner = pruner
        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)
        self.trainer._do_train = _do_train_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)


def prune(args):
    # load trained yolov8 model
    model = YOLO(args.model) # YOLO object
    model.__setattr__("train_v2", train_v2.__get__(model)) #add train_v2 function to model
    pruning_cfg = yaml_load(check_yaml(args.cfg))
    batch_size = pruning_cfg['batch'] = args.batch

    # modify gpu
    pruning_cfg['device'] = args.device

    #modify workers
    pruning_cfg['workers'] = args.workers

    # save results in folder
    pruning_cfg["project"] = args.project
    prefix_folder = f'train{args.no_runs}'

    # use coco128 dataset for 10 epochs fine-tuning each pruning iteration step
    # this part is only for sample code, number of epochs should be included in config file
    pruning_cfg['data'] = args.data
    pruning_cfg['epochs'] = args.epochs
    pruning_cfg["imgsz"] = args.imgsz
    pruning_cfg['lr0'] = args.lr0

    model.model.train()

    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
    flops_list, no_params_list, sparities_list = [], [], []
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    # do validation before pruning model
    pruning_cfg['name'] = os.path.join(prefix_folder, f"baseline_val")
    # pruning_cfg['batch'] = 1
    validation_model = deepcopy(model) #YOLO object
    metric = validation_model.val(**pruning_cfg)
    init_map = metric.box.map

    macs_list.append(base_macs)
    nparams_list.append(100)
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    flops_list.append(base_macs/1e9 *2)
    no_params_list.append(base_nparams/1e6)
    sparities_list.append(1.0)

    print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")

    # prune same ratio of filter based on initial size
    # áp dụng bài toán lãi suất kép để tính ch_sparsity cho mỗi iterative_steps
    ch_sparsity = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)
    '''
        giải thích:
            - Giả sử số lượng chanel ban đầu là Ao, số lượng chanel sau prune là An = A0 * (1-p) 
              với p = args.target_prune_rate, n = args.iterative_steps
            -> cần tìm c = ch_sparsity cho mỗi lần lặp pruning
            - Lặp:
                + A1 = A0(1 - c)
                + A2 = A1(1 - c) = A0(1-c)^2
                ...
                + An = A0(1 - c)^n 

            mà An = A0 * (1 - p)
            => (1 - c)^n = (1 - p)
            => c = 1 - (1-p) ^ (1/n) (dpcm)
    '''
    print('-----------pruning ratio each step:', ch_sparsity)
    for i in range(args.iterative_steps):

        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []
        for m in model.model.modules():
            # if isinstance(m, (Detect,)):
            #     for modulelist in m.cv2:
            #         ignored_layers.append(modulelist[-1])
            #     for modulelist in m.cv3:
            #         ignored_layers.append(modulelist[-1])
            if isinstance(m, (Detect,)):
                ignored_layers.append(m)
                break
        
        example_inputs = example_inputs.to(model.device)
        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            global_pruning = False, # additional test
            importance=tp.importance.MagnitudeImportance(p=2),  # L2 norm pruning,
            iterative_steps=1,
            ch_sparsity=ch_sparsity,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )
        pruner.step() # remove some weights with lowest importance

        # pre fine-tuning validation
        pruning_cfg['name'] = os.path.join(prefix_folder, f"step_{i}_pre_val")
        # pruning_cfg['batch'] = 1
        validation_model.model = deepcopy(model.model)
        metric = validation_model.val(**pruning_cfg)
        pruned_map = metric.box.map
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs)
        current_speed_up = float(macs_list[0]) / pruned_macs
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
              f"mAP={pruned_map}, speed up={current_speed_up}")

        # fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = os.path.join(prefix_folder, f"step_{i}_finetune")
        pruning_cfg['batch'] = batch_size  # restore batch size
        # pass pruner if sparse training
        model.train_v2(pruning=True, pruner = pruner if args.sparse_training else None, **pruning_cfg)

        # post fine-tuning validation
        pruning_cfg['name'] = os.path.join(prefix_folder, f"step_{i}_post_val")
        # pruning_cfg['batch'] = 1
        validation_model = YOLO(model.trainer.best)
        metric = validation_model.val(**pruning_cfg)
        current_map = metric.box.map
        print(f"After fine tuning mAP={current_map}")

        macs_list.append(pruned_macs)
        nparams_list.append(pruned_nparams / base_nparams * 100)
        pruned_map_list.append(pruned_map)
        map_list.append(current_map)
        flops_list.append(pruned_macs/1e9*2)
        no_params_list.append(pruned_nparams/1e6)
        sparities_list.append(sparities_list[-1] * (1-ch_sparsity))
        # remove pruner after single iteration
        del pruner

        save_pruning_performance_graph(nparams_list, map_list, macs_list, pruned_map_list, 
                                       dir = os.path.join(pruning_cfg["project"], prefix_folder) )
        
        save_pruning_size_graph(sparities_list, no_params_list, flops_list,
                                dir = os.path.join(pruning_cfg["project"], prefix_folder))
        if init_map - current_map > args.max_map_drop:
            print("Pruning early stop")
            break

    model.export(format='onnx')
    model.export(format='torchscript')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8m.pt', help='Pretrained pruning target model file')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--iterative-steps', default=16, type=int, help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.5, type=float, help='Target pruning rate')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')
    parser.add_argument('--sparse-training', default=False, type=bool, help='Sparse training')

    parser.add_argument('--batch', default=4, type=int, help='batch_size')
    parser.add_argument('--data', default='coco128.yaml', help='dataset')
    parser.add_argument('--device', default=0, help='cpu or gpu')
    parser.add_argument('--project', default='pruning', help='project name')
    parser.add_argument('--epochs', type=int, default=10, help='epochs each iterative-steps')
    parser.add_argument('--imgsz', type=int, default=640, help='Size of input images')
    parser.add_argument('--lr0', type=float, default=0.0001, help='Inintial learning rate')
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads for data loading (per RANK if DDP)")

    args = parser.parse_args()

    os.makedirs(args.project, exist_ok=True)
    k = os.listdir(args.project)
    args.no_runs = len(k)

    prune(args)
