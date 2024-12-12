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
from models.general import HCCF


class HCCFRunner(BaseRunner):
    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
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
            loss_lst.append(loss.detach().cpu().data.numpy())
        
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


    