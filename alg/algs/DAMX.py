# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import dice_loss,BCEFocalLoss,calc_loss
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from datautil.util import random_pairs_of_minibatches
import numpy as np
# from network.seg_network import UNet,U_Net
from nets.unet import Unet
from nets.fcn import *
class DAMX(Algorithm):

    def __init__(self, args):

        super(DAMX, self).__init__(args)
        # self.unet = Unet(num_classes = 1, pretrained = True, backbone = args.net)
        if args.net == 'vgg':
            self.net = fcn16_vgg16(n_classes = 1, batch_size = args.batch_size, pretrained=True, fixed_feature=False)
            # self.feature_extractor = fcn16_resnet50(n_classes = 1, batch_size = 4, pretrained=True, fixed_feature=False)
            self.discriminator = Adver_network.Discriminator(
                512 * 14 * 14, args.dis_hidden, args.domain_num - len(args.test_envs))
        else:
            self.net = fcn16_resnet50(n_classes = 1, batch_size = args.batch_size, pretrained=True, fixed_feature=False)
            self.discriminator = Adver_network.Discriminator(
                1024 * 14 * 14, args.dis_hidden, args.domain_num - len(args.test_envs))
             
        self.args = args
        # self.focalloss = BCEFocalLoss(alpha = 0.8)

    def update(self, minibatches, opt, sch, epoch):
        t_loss = 0
        c_loss = 0
        d_loss = 0
        opt.zero_grad()
        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(self.args, minibatches):
            
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
            x = (lam * xi + (1 - lam) * xj).cuda().float() 
            d_i = torch.vstack([torch.eye(self.args.domain_num - len(self.args.test_envs))[d.long()] for d in di ]).cuda()
            d_j = torch.vstack([torch.eye(self.args.domain_num - len(self.args.test_envs))[d.long()] for d in dj ]).cuda()

            x = x.repeat(1, 3, 1, 1)
            predictions, z = self.net(x) 
            disc_input = z.reshape(z.shape[0], -1)
            disc_input = Adver_network.ReverseLayerF.apply(disc_input, self.args.alpha)
            disc_out = self.discriminator(disc_input)     
            
            disc_loss = lam * F.cross_entropy(disc_out, d_i)
            disc_loss = (1 - lam) * F.cross_entropy(disc_out, d_j)
            
            classifier_loss = lam * calc_loss(predictions, yi.cuda().float().unsqueeze(1), bce_weight= self.args.bce_weight)
            classifier_loss += (1 - lam) * calc_loss(predictions, yj.cuda().float().unsqueeze(1), bce_weight= self.args.bce_weight)
            
            loss = disc_loss  + classifier_loss 
            c_loss += classifier_loss.item() 
            d_loss += disc_loss.item() 
            t_loss += loss.item() 
            loss.backward()
            
        opt.step()
        if sch:
            sch.step()
        return {'total': t_loss / len(minibatches) , 'class': c_loss / len(minibatches), 'dis': d_loss / len(minibatches)}

    def predict(self, x):
        x = x.repeat(1, 3, 1, 1)
        predictions, z = self.net(x)
        return predictions
