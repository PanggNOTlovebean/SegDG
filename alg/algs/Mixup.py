# coding=utf-8
import numpy as np
import torch.nn.functional as F

from alg.modelopera import get_fea, calc_loss
from datautil.util import random_pairs_of_minibatches
from alg.algs.ERM import ERM


class Mixup(ERM):
    def __init__(self, args):
        super(Mixup, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch, epoch):
        objective = 0

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(self.args, minibatches):
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)

            x = (lam * xi + (1 - lam) * xj).cuda().float()

            predictions = self.predict(x)

            objective += lam * calc_loss(predictions, yi.cuda().float().unsqueeze(1)) 
            objective += (1 - lam) * calc_loss(predictions, yj.cuda().float().unsqueeze(1)) 

        objective /= len(minibatches)

        opt.zero_grad()
        objective.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': objective.item()}
