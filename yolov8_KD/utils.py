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
from torch.optim import lr_scheduler
ultralytics_dir = os.path.abspath("./")


def add_params_kd(self: BaseTrainer, module_list: list):
    params = []
    for module in module_list:
        params += list(module.parameters())
    self.optimizer.add_param_group({'params': params})