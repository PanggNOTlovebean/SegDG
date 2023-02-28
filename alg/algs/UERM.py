# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea, calc_loss
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from datautil.util import random_pairs_of_minibatches
import numpy as np
from network.seg_network import UNet,U_Net

class UERM(Algorithm):

    def __init__(self, args):

        super(UERM, self).__init__(args)
        self.unet = U_Net(1)
        self.args = args

    def update(self, minibatches, opt, sch):
        
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().float() for data in minibatches])
        predictions, _ = self.unet(all_x)
        loss =  calc_loss(predictions, all_y.cuda().float().unsqueeze(1))

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss.item()}

    def predict(self, x):
        predictions, z = self.unet(x)
        return predictions
