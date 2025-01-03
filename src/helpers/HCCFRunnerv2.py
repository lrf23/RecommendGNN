# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner



class HCCFRunnerv2(BaseRunner):
    def _build_scheduler(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.3* (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
        
    def _build_optimizer(self, model):
        return torch.optim.Adam(model.customize_parameters(), lr=0.001, weight_decay=0.001)
    
    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
            
        num_training_steps = len(dataset) * epoch//self.batch_size
        num_warmup_steps = int(0.15 * num_training_steps)
        scheduler = self._build_scheduler(model.optimizer, num_warmup_steps, num_training_steps)
        lr_list=[]
        dataset.actions_before_epoch()  # must sample before multi thread start
        model.train()
        loss_lst = list()
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        for batch in tqdm(dl):
            batch = utils.batch_to_gpu(batch, model.device)
            out_dict = model(batch)
            loss= model.loss(out_dict,batch)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            scheduler.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
            lr_list.append(model.optimizer.param_groups[0]['lr'])
        #print('lr:',lr_list)
        return np.mean(loss_lst).item()

    def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
    	and the ground-truth item poses the first.
    	Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
    			 predictions like: [[1,3,4], [2,5,6]]
    	"""
        dataset.model.eval()
        predictions = list()
        dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=0)
        for batch in tqdm(dl):
            batch=utils.batch_to_gpu(batch, dataset.model.device)
            prediction=dataset.model.predict(batch)
            prediction=prediction.cpu().data.numpy()
            #print(prediction.shape)
            predictions.extend(prediction)
        predictions = np.array(predictions)
   
        return predictions


    