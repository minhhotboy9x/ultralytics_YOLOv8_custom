import os, sys
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO, __version__
from copy import deepcopy
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from torch.optim import lr_scheduler
from ultralytics.yolo.utils.metrics import bbox_iou
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from  ultralytics.nn.tasks import DetectionModel
from  ultralytics.nn.modules import Detect

ultralytics_dir = os.path.abspath("./")

class ExponentialDecayVariable:
    def __init__(self, initial_value=1, decay_rate = 0.1, min_value=0.0):
        self.value = initial_value
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.step_count = 0

    def step(self):
        self.value = np.exp(-self.decay_rate * self.step_count)
        self.step_count += 1
        self.value = max(self.value, self.min_value)

def add_params_kd(self: BaseTrainer, module_list: list):
    params = []
    for module in module_list:
        params += list(module.parameters())
    self.optimizer.add_param_group({'params': params})

class GetKeyRegion: # from teacher
    def __init__(self, model, topk=13):  # model must be de-paralleled
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no  # number of outputs per anchor
        self.reg_max = m.reg_max
        self.device = device
        
        # print(f'-------{self.reg_max}-------------') reg_max = 16
        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device) # (0->15)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device) # (batch_size, maximum count in counts, 5) 
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype)) # (b, a, 4)
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, alpha=0.5):
        # self.debug_bboxes(batch)
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() # batch, anchors, class
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() # batch, anchors, reg_max * 4

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_bboxes /= stride_tensor
        '''
            target_bboxes: (16, 8400, 4), (b, num_anchor, 4 edge)
            target_scores: (16, 8400, 80), (b, num_anchor, classes)
            pred_bboxes: (16, 8400, 4), (b, num_anchor, 4 edge)
            anchor_points: (8400, 2), (num_anchor, xy)
            pred_scores: (16, 8400, 80), (b, num_anchor, classes)
            stride_tensor: (8400, 1), (num_anchors, 1) # stride from feature maps to original images
            fg_mask: (16, 8400), (b, num_anchor) # mask to get prediction for each gt box (bool)
            self.stride = model.stride: (3)
            target_gt_idx (Tensor): shape(bs, num_total_anchors) # index of target gt box
        '''
        
        candidate_iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False).squeeze(axis = -1) # (16, 8400), (b, h*w)
        # mask_iou = candidate_iou * fg_mask # iou of chosen bboxes
        id_target_cls = torch.max(target_scores, dim=-1)[1] # (b, h*w)
        pred_target_scores = pred_scores[torch.arange(pred_scores.size(0)).unsqueeze(1), torch.arange(pred_scores.size(1)).unsqueeze(0), id_target_cls]
        # logit
        # print((torch.max(target_scores, dim=-1)[0] - candidate_iou )[0, 3000: 3300])

        return candidate_iou, pred_target_scores, target_gt_idx, fg_mask, id_target_cls, pred_scores, pred_distri # (16, 8400)
    
def get_detect_head(model: DetectionModel):
    detector = deepcopy(model.model[-1])
    
    def forward(self: Detect, x):
        shape = x[0].shape  # BCHW
        box_branch = []
        for i in range(self.nl):
            # Forward qua hai convolution layers đầu tiên trong self.cv2 và cv3
            box_branch.append(nn.Sequential(*self.cv2[i][:2])(x[i]))
        return box_branch
    
    detector.forward = forward.__get__(detector)
    return detector