# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea, calc_loss
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from datautil.util import random_pairs_of_minibatches
import numpy as np
from nets.unet import Unet
from nets.fcn import *

class ERM(Algorithm):

    def __init__(self, args):

        super(ERM, self).__init__(args)
        if args.net == 'vgg':
            self.net = fcn16_vgg16(n_classes = 1, batch_size = args.batch_size, pretrained=True, fixed_feature=False)
        else:
            self.net = fcn16_resnet50(n_classes = 1, batch_size = args.batch_size, pretrained=True, fixed_feature=False)
        self.args = args

    def update(self, minibatches, opt, sch, epoch):
        
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().float() for data in minibatches])
        all_x = all_x.repeat(1, 3, 1, 1)
        predictions, _ = self.net(all_x)
        loss =  calc_loss(predictions, all_y.cuda().float())

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss.item()}

    def predict(self, x):
        x = x.repeat(1, 3, 1, 1)
        predictions, z = self.net(x)
        return predictions
