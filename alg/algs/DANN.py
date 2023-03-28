# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from nets.unet import Unet
from nets.fcn import *
from alg.modelopera import get_fea, calc_loss
class DANN(Algorithm):

    def __init__(self, args):

        super(DANN, self).__init__(args)
        self.net = fcn16_resnet50(n_classes = 1, batch_size = args.batch_size, pretrained=True, fixed_feature=False)
        self.discriminator = Adver_network.Discriminator(
                1024 * 14 * 14, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.args = args

    def update(self, minibatches, opt, sch, epoch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        x = all_x.repeat(1, 3, 1, 1)
        
        all_preds, all_z = self.net(x)
        disc_input = all_z.reshape(all_z.shape[0], -1)
        disc_input = Adver_network.ReverseLayerF.apply(disc_input, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((data[0].shape[0], ), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])

        disc_loss = F.cross_entropy(disc_out, disc_labels)
        classifier_loss = calc_loss(all_preds, all_y.cuda().float(), bce_weight= self.args.bce_weight)
        loss = classifier_loss + disc_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x):
        x = x.repeat(1, 3, 1, 1)
        predictions, z = self.net(x)
        return predictions, z
