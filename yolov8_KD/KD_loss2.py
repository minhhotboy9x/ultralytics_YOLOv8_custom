import os, sys
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO, __version__
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, TQDM_BAR_FORMAT, 
                            DEFAULT_CFG_KEYS, DEFAULT_CFG, callbacks, clean_url, colorstr, emojis, yaml_save)
from pytorch_msssim import ssim
from utils import GetKeyRegion
from KD_loss import KLDLoss as KLD
ultralytics_dir = os.path.abspath("./")

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # eps = 1e-9
        loss = - y_true * (y_pred / y_true).log()
        return loss.sum()

class RMPG(nn.Module):
    def __init__(self, teacher, student_preds, teacher_preds) -> None:
        super(RMPG, self).__init__()
        self.device = next(teacher.parameters()).device
        self.teacher = teacher
        self.getkey = GetKeyRegion(teacher)
        self.tea_preds = teacher_preds
        self.stu_preds = student_preds

    def head_loss(self, stu_ious, tea_ious, stu_target_scores, tea_target_scores, tea_gt_ids, T=1.0):
        b, a = stu_ious.shape
        N_tea_scores = [] * b
        N_tea_ious = [] * b
        N_stu_scores = [] * b
        N_stu_ious = [] * b
        t_loss = 0
        kl_criteria = KLDLoss()
        for gt_box, tea_score, stu_score, tea_iou, stu_iou in zip(tea_gt_ids, tea_target_scores,
                                         stu_target_scores, tea_ious, stu_ious):
            N_tea_scores.append({}) # [batch: {gt_box:[]}]
            N_tea_ious.append({})
            N_stu_scores.append({})
            N_stu_ious.append({})
            n_box = gt_box.max().item()
            tmp_loss = 0
            for i in range(a):
                id_gt_box = gt_box[i].item()
                if id_gt_box == 0:
                    continue
                if id_gt_box not in N_tea_scores[-1]:
                    N_tea_scores[-1][id_gt_box ] = torch.tensor([], device=self.device)
                    N_stu_scores[-1][id_gt_box ] = torch.tensor([], device=self.device)
                    N_tea_ious[-1][id_gt_box ] = torch.tensor([], device=self.device)
                    N_stu_ious[-1][id_gt_box ] = torch.tensor([], device=self.device)
                N_tea_scores[-1][id_gt_box ] = torch.cat((N_tea_scores[-1][id_gt_box], tea_score[i].unsqueeze(0)), dim=0)
                N_stu_scores[-1][id_gt_box ] = torch.cat((N_stu_scores[-1][id_gt_box], stu_score[i].unsqueeze(0)), dim=0)
                N_tea_ious[-1][id_gt_box ] = torch.cat((N_tea_ious[-1][id_gt_box], tea_iou[i].unsqueeze(0)), dim=0)
                N_stu_ious[-1][id_gt_box ] = torch.cat((N_stu_ious[-1][id_gt_box], stu_iou[i].unsqueeze(0)), dim=0)
            for i in range(0, n_box+1):
                if N_stu_scores[-1].get(i) is not None:
                    # print(i, N_stu_scores[-1].get(i))
                    # print('---', N_stu_scores[-1].get(i).softmax(dim=-1))
                    tmp_loss += kl_criteria(N_stu_scores[-1][i].softmax(dim=-1), N_tea_scores[-1][i].softmax(dim=-1)) 
                    tmp_loss += kl_criteria(N_stu_ious[-1][i].softmax(dim=-1), N_tea_ious[-1][i].softmax(dim=-1))
            t_loss += tmp_loss/(n_box+1)
        t_loss /= b        
        # print(f'----{type(t_loss)} {t_loss}-----------')
        return t_loss

    def forward(self, y_pred, y_true, batch):
        L_imi = 0
        L_head = 0
        stu_ious, _, _, _, _, stu_pred_scores, _ = self.getkey(self.stu_preds, batch) 
        tea_ious, tea_target_scores, tea_gt_ids, Ms, tea_id_target_cls, tea_pred_scores, _ = self.getkey(self.tea_preds, batch)
        # stu_pred_scores, tea_pred_scores are logit
        stu_target_scores = stu_pred_scores[
                            torch.arange(stu_pred_scores.size(0)).unsqueeze(1), 
                            torch.arange(stu_pred_scores.size(1)).unsqueeze(0), 
                            tea_id_target_cls]
        # print(stu_target_scores.sigmoid()[0, 3000:3100])
        # stu_mask_quality, stu_gt_ids = torch.zeros_like(tea_mask_quality), torch.zeros_like(tea_gt_ids)
        # tea_mask_quality, tea_gt_ids = torch.zeros_like(stu_mask_quality), torch.zeros_like(stu_gt_ids)
        
        # get only cls output
        stu_pred_maps = torch.split(stu_pred_scores.sigmoid(), [xi.shape[2]**2 for xi in y_pred], dim=1) # (b, anchors, cls)
        tea_pred_maps = torch.split(tea_pred_scores.sigmoid(), [xi.shape[2]**2 for xi in y_pred], dim=1)
        stu_pred_maps = [stu_pred_maps[i].view(y_pred[i].shape[0], y_pred[i].shape[2], y_pred[i].shape[3], -1) for i in range(len(y_pred))]
        tea_pred_maps = [tea_pred_maps[i].view(y_pred[i].shape[0], y_pred[i].shape[2], y_pred[i].shape[3], -1) for i in range(len(y_pred))]
        
        L_head = self.head_loss(stu_ious, tea_ious, stu_target_scores.sigmoid(), tea_target_scores.sigmoid(), tea_gt_ids)

        p_difs = []
        for tea_pred_map, stu_pred_map in zip(tea_pred_maps, stu_pred_maps):
            dif = ((tea_pred_map - stu_pred_map)**2).mean(dim=-1)
            p_difs.append(dif)

        f_difs = [] 
        for tea_feat, stu_feat in zip(y_true, y_pred):
            dif = ((tea_feat - stu_feat)**2).mean(dim=1)
            f_difs.append(dif)
        for p_dif, f_dif in zip(p_difs, f_difs):
            tmp_norm = torch.norm(p_dif*f_dif, dim = (1, 2))**2 / (p_dif.shape[1]*p_dif.shape[2])
            L_imi += tmp_norm.mean()
        L_imi /= len(y_pred)
        # print(f'-----------{L_imi}-----{L_head}------')
        return 1.5*L_imi + 4.0*L_head

class LDLoss(nn.Module):
    def __init__(self, teacher, student_preds, teacher_preds) -> None:
        super(LDLoss, self).__init__()
        self.device = next(teacher.parameters()).device
        self.teacher = teacher
        self.getkey = GetKeyRegion(teacher, topk=10)
        self.tea_preds = teacher_preds
        self.stu_preds = student_preds

    def head_loss(self, stu_distri, tea_distri, stu_pred_scores, tea_pred_scores, Ms, T=1.0):
        ba, c = tea_distri[Ms].shape  # chosen batch and anchors, channels
        tea_distri_chosen = (tea_distri[Ms]/T).view(ba, 4, c // 4).softmax(2)
        stu_distri_chosen = (stu_distri[Ms]/T).view(ba, 4, c // 4).softmax(2)
        kl_loss = KLD()
        box_loss = kl_loss(stu_distri_chosen, tea_distri_chosen)
        return box_loss


    def forward(self, y_pred, y_true, batch):
        tea_candidate_iou, _, tea_target_gt_idx, Ms, _, tea_pred_scores, tea_distri = self.getkey(self.tea_preds, batch)
        stu_candidate_iou, _, stu_target_gt_idx, _, _, stu_pred_scores, stu_distri = self.getkey(self.stu_preds, batch)
        
        # get the beneficial anchors
        Ms = (tea_candidate_iou >= stu_candidate_iou) * (tea_target_gt_idx == stu_target_gt_idx) * Ms

        loss = self.head_loss(stu_distri, tea_distri, stu_pred_scores, tea_pred_scores, Ms, T=1.0)
        # print(f'kd loss {loss}')
        return loss
    