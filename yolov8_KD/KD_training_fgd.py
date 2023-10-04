import subprocess
import time
import argparse
import math
import os, sys

ultralytics_dir = os.path.abspath("./")
# Thêm đường dẫn của folder cha vào sys.path
sys.path.append(ultralytics_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO, __version__
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from pytorch_msssim import ssim
from ultralytics.yolo.cfg import get_cfg
from tqdm import tqdm
from ultralytics.yolo.utils.callbacks.tensorboard import on_batch_end2, on_fit_epoch_end2

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.alpha_ = 1.6e-3
        self.beta_ = 8e-4
        self.gamma_ = 8e-3
        

    def binary_scale_masks(self, batch, teacher_features):
        Ms = [torch.zeros(i.shape).to(i.device) for i in teacher_features]
        Ss = [torch.ones(i.shape).to(i.device) for i in teacher_features]
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)

        '''
        import matplotlib.pyplot as plt
        imgs = batch['img'].clone()
        imgs = [row for row in imgs]
        
        for i in range(len(imgs)):
            imgs[i] = imgs[i].permute(1, 2, 0).cpu().numpy()
        '''

        for target in targets:
            id = int(target[0])
            x_center, y_center, width, height = [int(i*640) for i in target[2:]]
            color = (0, 255, 0) 
            thickness = 2 

            x_left, y_top = (x_center - width//2, y_center - height//2)
            x_right, y_down = (x_center + width//2, y_center + height//2)

            '''
            imgs[id][y_top:y_top+thickness, x_left:x_right+1, :] = color  # Vẽ top edge
            imgs[id][y_top:y_down+1, x_left:x_left+thickness, :] = color  # Vẽ left edge
            imgs[id][y_down-thickness:y_down, x_left:x_right+1, :] = color  # Vẽ bottom edge
            imgs[id][y_top:y_down+1, x_right-thickness:x_right, :] = color  # Vẽ left edge
            '''

            for i in range(len(Ms)):
                x_center, y_center, width, height = [int(id*Ms[i].shape[2]) for id in target[2:]]
                x_left, y_top = (x_center - width//2, y_center - height//2)
                x_right, y_down = (x_center + width//2, y_center + height//2)
                width += x_right - x_left + 1
                height += y_down - y_top + 1
                Ms[i][id, :, y_top:y_down+1, x_left:x_right+1] = 1.0
                # Ss[i][id, :, y_top:y_down+1, x_left:x_right+1] = \
                #     torch.min(Ss[i][id, :, y_top:y_down+1, x_left:x_right+1], torch.tensor(1/(width*height))[None, None, None, None])
                
        for i in range(len(Ss)):
            Ss[i] = Ss[i]/(1 - Ms[i]).sum(dim=(2, 3))[:, :, None, None]
        
        for target in targets:
            id = int(target[0])
            for i in range(len(Ss)):
                x_center, y_center, width, height = [int(id*Ms[i].shape[2]) for id in target[2:]]
                x_left, y_top = (x_center - width//2, y_center - height//2)
                x_right, y_down = (x_center + width//2, y_center + height//2)
                width += x_right - x_left + 1
                height += y_down - y_top + 1
                Ss[i][id, :, y_top:y_down+1, x_left:x_right+1] = 1.0
                Ss[i][id, :, y_top:y_down+1, x_left:x_right+1] = \
                    torch.min(Ss[i][id, :, y_top:y_down+1, x_left:x_right+1], 
                              torch.tensor(1/(width*height)).to(teacher_features[i].device)[None, None, None, None])
                
        '''
        for i in range(len(imgs)):
            plt.subplot(3, 3, 1)
            plt.imshow(Ms[0][i, 0, :, :].cpu().numpy(), cmap='gray')
            plt.subplot(3, 3, 2)  
            plt.imshow(Ms[1][i, 1, :, :].cpu().numpy(), cmap='gray')
            plt.subplot(3, 3, 3)  
            plt.imshow(Ms[2][i, 2, :, :].cpu().numpy(), cmap='gray')
            plt.subplot(3, 3, 4)  
            plt.imshow(imgs[i])
            
            plt.subplot(3, 3, 5)
            plt.imshow(Ss[0][i, 0, :, :].cpu().numpy(), cmap='hot')
            
            torch.set_printoptions(threshold=10000)
            plt.text(50, 125, Ss[2][i, 0, :, :].cpu().numpy(), color='red', fontsize=6, ha='center', va='center')

            plt.subplot(3, 3, 6)  
            plt.imshow(Ss[1][i, 1, :, :].cpu().numpy(), cmap='hot')

            plt.subplot(3, 3, 7)  
            plt.imshow(Ss[2][i, 2, :, :].cpu().numpy(), cmap='hot')

            plt.savefig('image.png')
            plt.show()
            plt.close()
        '''
        return Ms, Ss

    def attention_masks(self, features, T=1):
        Ass = []
        Acs = [] 
        for feature in features:
            b, c, h, w = feature.shape
            Gs = 1/c * feature.abs().sum(dim=1) # b, h, w
            Gc = 1/(h*w) * feature.abs().sum(dim=(2, 3))[:, :, None, None].repeat(1, 1, h, w) # b, c, h, w

            m_1d = nn.Softmax(dim=1)
            m_2d = nn.Softmax2d()

            As = h*w * m_1d((Gs/T).view(b, -1)).view(b, 1, h, w).repeat(1, c, 1, 1)
            Ac = c * m_2d(Gc/T)   

            Ass.append(As)
            Acs.append(Ac)
        
        return Ass, Acs

    def forward(self, y_pred, y_true, batch): 
        Ms, Ss = self.binary_scale_masks(batch, y_true)
        A_tea_ss, A_tea_cs = self.attention_masks(y_true)
        A_stu_ss, A_stu_cs = self.attention_masks(y_pred)
        l1 = nn.L1Loss()
        L_at = 0
        L_fea = 0
        for i in range(len(A_tea_cs)):
            L_at += self.gamma_*(l1(A_stu_ss[i], A_tea_ss[i]) + l1(A_stu_cs[i], A_tea_cs[i]))
        
        L_at /= len(A_stu_cs)

        for i in range(len(A_tea_cs)):
            b, c, h, w = y_true[i].shape 
            # print(f'--{A_tea_ss[i].device}----{A_tea_cs[i].device}----{Ms[i].device}----{Ss[i].device}----{y_true[i].device}----{y_pred[i].device}-')
            L_fea += self.alpha_ * (Ms[i] * Ss[i] * A_tea_ss[i] * A_tea_cs[i] * torch.pow(y_true[i]-y_pred[i], 2)).sum() / b \
                + self.beta_ * ((1-Ms[i]) * Ss[i] * A_tea_ss[i] * A_tea_cs[i] * torch.pow(y_true[i]-y_pred[i], 2)).sum() / b

        L_fea /= len(A_stu_cs)
        # print(f'---------------{L_fea}--{L_at}------------------')
        return L_fea + L_at

class GlobalLoss(nn.Module):
    def __init__(self):
        super(GlobalLoss, self).__init__()
        self.lambda_ = 8e-6
        
    def forward(self, y_pred, y_true, gc_blocks):
        mse = nn.MSELoss()
        loss = 0
        for i in range(len(y_pred)):
            r_pred = gc_blocks[i](y_pred[i])
            r_true = gc_blocks[i](y_true[i])
            loss += self.lambda_ * mse(r_pred, r_true)
        return loss

class GcBlock(nn.Module):
    def __init__(self, c = 3):
        super(GcBlock, self).__init__()
        # Định nghĩa các lớp convolution và pooling
        self.Wk = nn.Conv2d(in_channels=c, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.softmax2d = nn.Softmax2d()
        self.Wv1 = nn.Conv2d(in_channels=c, out_channels=c//2, kernel_size=1, stride=1)
        self.layer_norm = nn.LayerNorm([c//2, 1, 1]) 
        self.relu = nn.ReLU(inplace=True)
        self.Wv2 = nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=1, stride=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.Wk(x) # b, 1, H, W
        x1 = x1.view(b, 1, h*w) # b, 1, w*h
        x1 = x1.unsqueeze(1).permute(0, 3, 1, 2) # b, h*w, 1, 1
        x1 = self.softmax2d(x1)
        x1 = x1.view(b, h*w, 1) # b, h*w, 1
        x_ = x.view(b, c, h*w) # b, c, h*w
        # print(x_.shape, x1.shape)
        y1 = torch.matmul(x_, x1).view(b, c, 1, 1) #b, c, 1, 1
    
        x2 = self.Wv1(y1) # b, c//2, 1, 1
        # print(f'---------------{x2.shape}--{y1.shape}-------{x1.shape}----------')
        x2 = self.layer_norm(x2) # b, c//2, 1, 1 
        x2 = self.relu(x2) # b, c//2, 1, 1
        x2 = self.Wv2(x2) # b, c, 1, 1

        output = x + x2 # b, c, w, h
        return output

def cal_kd_loss(student_preds, teacher_preds, batch, gc_blocks):
    focal_cri = FocalLoss()
    global_cri = GlobalLoss()
    l_focal = focal_cri(student_preds, teacher_preds, batch)
    l_global = global_cri(student_preds, teacher_preds, gc_blocks)
    # print(f'-----------{l_global}-----------')
    return l_focal + l_global

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


        # create imitation mask for knowledge distillation
        mask_id = self.args.kd_layer
        if hasattr(self, 'teacher'):
            self.type_kd_loss = 'fgd_loss' # -----------------get type kd loss-----------------
            self.teacher.train() 
            dump_image = torch.zeros((1, 3, self.args.imgsz, self.args.imgsz), device=self.device)

            _, features = self.model(dump_image, mask_id = mask_id)  # forward
            _, teacher_feature= self.teacher(dump_image, mask_id = mask_id) 
            
            stu_feature_adapts = [] # contain imitation
            gc_blocks = [] # gc_block for global distill

            for i in range(len(features)):
                _, student_channel, student_out_size, _ = features[i].shape
                _, teacher_channel, teacher_out_size, _ = teacher_feature[i].shape
                stu_feature_adapts.append(nn.Sequential(nn.Conv2d(student_channel, teacher_channel, 1,
                                                        padding=0, stride=1)).to(self.device))
                gc_blocks.append(GcBlock(c=teacher_channel).to(self.device))
            

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
            self.t_kdloss = 0 # mean of kd loss

            #----------------mean feat-----------------
            self.stu_mean = []
            self.tea_mean = []
            for id in mask_id:
                self.stu_mean.append({id: 0})
                self.tea_mean.append({id: 0})
            #------------------------------------------------

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
                        preds, features =  self.model(batch['img'], mask_id = mask_id)
                        teacher_preds, teacher_features =  self.teacher(batch['img'], mask_id = mask_id)
                    else:
                        preds = self.model(batch['img']) # predict

                    self.loss, self.loss_items = self.criterion(preds, batch) # cal loss

                    if hasattr(self, 'teacher'):
                        stu_feature_maps = []
                        for i in range(len(features)):
                            # print(f'-------------{stu_feature_adapts[i]}')
                            stu_feature_maps.append(stu_feature_adapts[i](features[i]))

                        fgd_loss = FocalLoss()
                        fgd_loss.forward(stu_feature_maps, [f.detach() for f in teacher_features], batch) # debug

                        self.kd_loss = cal_kd_loss(stu_feature_maps, 
                                                [f.detach() for f in teacher_features], batch, gc_blocks)
                        
                        #--------------------------------------------------------
                        with torch.no_grad():
                            for j in range(len(mask_id)):
                                self.stu_mean[j][mask_id[j]] = (self.stu_mean[j][mask_id[j]] * i + torch.mean(stu_feature_maps[j])) / (i+1)
                                self.tea_mean[j][mask_id[j]] = (self.tea_mean[j][mask_id[j]] * i + torch.mean(teacher_features[j])) / (i+1)
                        
                        #-----------------------------------------------------------
                    
                    if RANK != -1:
                        self.loss *= world_size
                        if hasattr(self, 'teacher'):
                            self.kd_loss *= world_size

                    #-------------------------------------------------------
                    self.t_kdloss = (self.t_kdloss * i + self.kd_loss) / (i + 1)
                    
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.alpha_kd = 2.0 # default 2.0
                self.scaler.scale(self.loss + self.alpha_kd * self.kd_loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    # + '\n' + '%15.5g' * (1 + len(mask_id)*2)
                    # , self.kd_loss
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len) + '%11.4g') % 
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1], self.t_kdloss))
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

def progress_string_v2(self: BaseTrainer):
    """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
    return ('\n' + '%11s' *
            (4 + len(self.loss_names))  + '%11s') % \
            ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size', 'KD_loss')

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
    self.trainer.progress_string = progress_string_v2.__get__(self.trainer)
    self.trainer.train_v2()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)

def main(args):
    args = get_cfg(DEFAULT_CFG, vars(args))
    model = YOLO(args.model)
    model.add_callback('on_batch_end', on_batch_end2)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end2)
    model.train_v2 = train_v2.__get__(model)
    model.train_v2(**vars(args))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8m.yaml', help='Pretrained pruning target model file')
    parser.add_argument('--teacher', help='teacher model')
    parser.add_argument('--kd_layer', nargs='*', type=int, default=[15, 18, 21], help="id of layers for KD")
    # parser.add_argument('--type_kd_loss', default='mse', help='type of kd loss for feature maps')

    parser.add_argument('--batch', default=4, type=int, help='batch_size')
    parser.add_argument('--data', default='coco128.yaml', help='dataset')
    parser.add_argument('--device', default=0, help='cpu or gpu')
    parser.add_argument('--project', default='KD_feature', help='project name')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Size of input images')
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads for data loading (per RANK if DDP)")
    parser.add_argument('--resume', type=bool, default=False, help="continue training (if KD, must provide teacher)")
    
    args = parser.parse_args()

    main(args)
    

    


    
    

