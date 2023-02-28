# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm

from datautil.util import random_pairs_of_minibatches
import numpy as np

class DAMXX(Algorithm):

    def __init__(self, args):

        super(DAMXX, self).__init__(args)

        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            self.featurizer.in_features, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.args = args

    def update(self, minibatches, opt, sch):
        classifier_loss = 0
        disc_loss = 0
        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(self.args, minibatches):
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
            x = (lam * xi + (1 - lam) * xj).cuda().float() 
            d_i = torch.vstack([torch.eye(self.args.domain_num - len(self.args.test_envs))[d.long()] for d in di ]).cuda()
            d_j = torch.vstack([torch.eye(self.args.domain_num - len(self.args.test_envs))[d.long()] for d in dj ]).cuda()
            z = self.featurizer(x)
            disc_input = z
            disc_input = Adver_network.ReverseLayerF.apply(disc_input, self.args.alpha)
            disc_out = self.discriminator(disc_input)
            disc_loss += lam * F.cross_entropy(disc_out, d_i)
            disc_loss += (1 - lam) * F.cross_entropy(disc_out, d_j)
            predictions = self.classifier(z)
            classifier_loss += lam * F.cross_entropy(predictions, yi.cuda().long())
            classifier_loss += (1 - lam) * F.cross_entropy(predictions, yj.cuda().long())

        classifier_loss /= len(minibatches)
        disc_loss /= len(minibatches)
        loss = classifier_loss + self.args.alpha * disc_loss
    
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
