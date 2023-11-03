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
ultralytics_dir = os.path.abspath("./")

def head_loss(self: BaseTrainer, student_preds, teacher_preds, T=3.0):
    m = self.model.model[-1]
    nc = m.nc  # number of classes
    no = m.no  # number of outputs per anchor
    reg_max = m.reg_max

    stu_pred_distri, stu_pred_scores = torch.cat([xi.view(student_preds[0].shape[0], no, -1) for xi in student_preds], 2).split(
            (reg_max * 4, nc), 1)

    stu_pred_scores = stu_pred_scores.permute(0, 2, 1).contiguous() # batch, anchors, class
    stu_pred_distri = stu_pred_distri.permute(0, 2, 1).contiguous() # batch, anchors, reg_max * 4
    b, a, c = stu_pred_distri.shape  # batch, anchors, channels

    tea_pred_distri, tea_pred_scores = torch.cat([xi.view(teacher_preds[0].shape[0], no, -1) for xi in teacher_preds], 2).split(
            (reg_max * 4, nc), 1)

    tea_pred_scores = tea_pred_scores.permute(0, 2, 1).contiguous() # batch, anchors, class
    tea_pred_distri = tea_pred_distri.permute(0, 2, 1).contiguous() # batch, anchors, reg_max * 4
    #-------------------------------KD loss----------------------------------------
    kl_loss = KLDLoss()
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    stu_pred_scores = (stu_pred_scores/T)
    stu_pred_distri = (stu_pred_distri/T).view(b, a, 4, c // 4).softmax(3)

    tea_pred_scores = (tea_pred_scores/T).sigmoid()
    tea_pred_distri = (tea_pred_distri/T).view(b, a, 4, c // 4).softmax(3)

    cl_loss = bce_loss(stu_pred_scores, tea_pred_scores)
    box_loss = kl_loss(stu_pred_distri, tea_pred_distri)

    # print(f'----------------size----------- {tea_pred_scores.shape} {tea_pred_distri.shape}')  # torch.Size([32, 8400, 80]) torch.Size([32, 8400, 4, 16])

    # print(f'\n----------------kd_loss----------- {cl_loss} {box_loss}')

    # print(f'\n---------------{stu_pred_scores[13, 4]}, \n { tea_pred_scores[13, 4]}')
    # for i in range(8400):
    #     tmp = kl_loss(stu_pred_scores[13, i], tea_pred_scores[13, i])
    #     if torch.isnan(tmp):
    #         print(f'\n----------------kd_loss----------- {i}: {kl_loss(stu_pred_scores[13, i], tea_pred_scores[13, i])} ')
    #         break

    return 0.8*box_loss + 0.2*cl_loss

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # epsilon = 1e-7
        loss = - torch.where(y_true != 0, y_true * (y_pred / y_true).log(), torch.tensor(0.0))
        num_dims = y_pred.dim()
        # keep_dims = tuple(range(1, num_dims))
        return loss.sum(dim = num_dims-1).mean()

class MinMaxRescalingLayer(nn.Module):
    def __init__(self):
        super(MinMaxRescalingLayer, self).__init__()

    def forward(self, x, y):
        min_val = torch.min(x.min(-1)[0].min(-1)[0], y.min(-1)[0].min(-1)[0])
        max_val = torch.max(x.max(-1)[0].max(-1)[0], y.max(-1)[0].min(-1)[0])
        
        # Kiểm tra và xử lý trường hợp mẫu số bằng 0
        denominator_zero_mask = (max_val - min_val) == 0
        max_val = torch.where(denominator_zero_mask, max_val + 1e-6, max_val)
        
        rescaled_x = (x - min_val[:,:,None,None]) / (max_val[:,:,None,None] - min_val[:,:,None,None])
        rescaled_y = (y - min_val[:,:,None,None]) / (max_val[:,:,None,None] - min_val[:,:,None,None])
        return rescaled_x, rescaled_y

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        mse = nn.MSELoss(reduction='mean')
        loss = 0
        for i in range(len(y_pred)):
            loss += mse(y_pred[i], y_true[i])
        loss /= len(y_pred)
        return loss

class DSSIMLoss(nn.Module):
    def __init__(self, device = 'cpu'):
        super(DSSIMLoss, self).__init__()
        self.device = device
        self.scaler = MinMaxRescalingLayer()

    def forward(self, y_pred, y_true): 
        t_loss = 0
        for i in range(len(y_pred)):
            y_pred_scaled, y_true_scaled = self.scaler(y_pred[i], y_true[i])
            loss = ssim(y_pred_scaled, y_true_scaled, data_range=1.0)
            loss = (1-loss)/2
            t_loss += loss
        t_loss /= len(y_pred)
        # print(f'------------{type(t_loss)}--------------')
        return t_loss

class DSSIM(nn.Module):
    def __init__(self, device = 'cpu'):
        super(DSSIM, self).__init__()
        self.device = device
        self.scaler = MinMaxRescalingLayer()

    def forward(self, y_pred, y_true): 
        y_pred_scaled, y_true_scaled = self.scaler(y_pred, y_true)
        loss = ssim(y_pred_scaled, y_true_scaled, data_range=1.0)
        loss = (1-loss)/2
        # print(f'------------{type(loss)}--------------')
        return loss

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
            '''
            x_center, y_center, width, height = [int(i*640) for i in target[2:]]
            color = (0, 255, 0) 
            thickness = 2 
            x_left, y_top = (x_center - width//2, y_center - height//2)
            x_right, y_down = (x_center + width//2, y_center + height//2)
            imgs[id][y_top:y_top+thickness, x_left:x_right+1, :] = color  # Vẽ top edge
            imgs[id][y_top:y_down+1, x_left:x_left+thickness, :] = color  # Vẽ left edge
            imgs[id][y_down-thickness:y_down, x_left:x_right+1, :] = color  # Vẽ bottom edge
            imgs[id][y_top:y_down+1, x_right-thickness:x_right, :] = color  # Vẽ left edge
            '''

            for i in range(len(Ms)):
                x_center, y_center, width, height = [int(id*Ms[i].shape[2]) for id in target[2:]]
                x_left, y_top = (x_center - width//2, y_center - height//2)
                x_right, y_down = (x_center + width//2, y_center + height//2)
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
                width = x_right - x_left + 1
                height = y_down - y_top + 1
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

    def attention_masks(self, features, T=0.5):
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
        x1 = x1.view(b, 1, h*w) # b, 1, h*w
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
    
class FGFI(nn.Module):
    def __init__(self):
        super(FGFI, self).__init__()
    
    def binary_scale_masks(self, batch, teacher_features):
        Ms = [torch.zeros(i.shape).to(i.device) for i in teacher_features]
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        for target in targets:
            id = int(target[0])
            for i in range(len(Ms)):
                x_center, y_center, width, height = [int(id*Ms[i].shape[2]) for id in target[2:]]
                x_left, y_top = (x_center - width//2, y_center - height//2)
                x_right, y_down = (x_center + width//2, y_center + height//2)
                x_left, y_top = max(x_left, 0), max(y_top, 0)
                x_right, y_down = min(x_right, Ms[i].shape[2]-1), min(y_down, Ms[i].shape[2]-1)
                Ms[i][id, :, y_top:y_down+1, x_left:x_right+1] = 1.0
        return Ms
    
    def forward(self, y_pred, y_true, batch): 
        Ms = self.binary_scale_masks(batch, y_true)
        mse = nn.MSELoss(reduction='sum')
        L_imi = 0
        for i in range(len(y_pred)):
            N_pos_points = torch.sum(Ms[i])
            tmp_stu_feat = Ms[i] * y_pred[i]
            tmp_tea_feat = Ms[i] * y_true[i]
            L_imi = L_imi + mse(tmp_stu_feat, tmp_tea_feat) / N_pos_points
        L_imi /= len(Ms)
        # print(f'----------kd-----{L_imi.requires_grad}----------------')
        return L_imi

class BoxGauss(nn.Module):
    def __init__(self):
        super(BoxGauss, self).__init__()
    
    def create_mask(self, x_left, x_right, y_top, y_down, x_center, y_center, std=2):
        width = x_right - x_left + 1
        height = y_down - y_top + 1
        y_cord, x_cord = torch.meshgrid(torch.linspace(y_top, y_down, height), torch.linspace(x_left, x_right, width))
        gaussian_matrix = torch.exp( -( (x_cord-x_center)**2 / (std**2 * (width/2)**2) + (y_cord-y_center)**2 / (std**2 * (height/2)**2)))
        # print(f'------------{width}-------------')
        return gaussian_matrix

    def gauss_masks(self, batch, teacher_features):
        device = teacher_features[0].device
        Ms = [torch.zeros(i.shape).to(device) for i in teacher_features]
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        for target in targets:
            id = int(target[0])
            for i in range(len(Ms)):
                x_center, y_center, width, height = [int(id*Ms[i].shape[2]) for id in target[2:]]
                x_left, y_top = (x_center - width//2, y_center - height//2)
                x_right, y_down = (x_center + width//2, y_center + height//2)
                x_left, y_top = max(x_left, 0), max(y_top, 0)
                x_right, y_down = min(x_right, Ms[i].shape[2]-1), min(y_down, Ms[i].shape[2]-1)
                # print(f'----------origin shape {x_right - x_left + 1}----------------')
                Ms[i][id, :, y_top:y_down+1, x_left:x_right+1] = \
                    torch.max(Ms[i][id, :, y_top:y_down+1, x_left:x_right+1], self.create_mask(x_left, x_right, y_top, y_down, x_center, y_center).to(device))
        return Ms
    
    def forward(self, y_pred, y_true, batch): 
        Ms = self.gauss_masks(batch, y_true)
        mse = nn.MSELoss(reduction='sum')
        L_imi = 0
        for i in range(len(y_pred)):
            N_pos_points = torch.sum(Ms[i])
            tmp_stu_feat = Ms[i] * y_pred[i]
            tmp_tea_feat = Ms[i] * y_true[i]
            L_imi = L_imi + mse(tmp_stu_feat, tmp_tea_feat) / N_pos_points
        L_imi /= len(Ms)
        # print(f'----------kd-----{L_imi.requires_grad}----------------')
        return L_imi
    

class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap

