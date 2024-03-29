# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ
from datautil.getdataloader import get_img_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="DAMX")
    parser.add_argument('--alpha', type=float,
                        default=0.1, help='DANN dis alpha')
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--bce_weight', type=float,
                        default=0.5, help='bce_weight')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--output', type=str,
                        default="output/test", help='result output path')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=4, help='batch_size')
    parser.add_argument('--beta', type=float,
                        default=1, help='DIFEX beta')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=1, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='medical')
    parser.add_argument('--data_dir', type=str, default='data', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--disttype', type=str, default='2-norm',
                        choices=['1-norm', '2-norm', 'cos', 'norm-2-norm', 'norm-1-norm'])
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=2, help="max iterations")
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')

    parser.add_argument('--N_WORKERS', type=int, default=8)
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
 help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[3], help='target domains')
    parser.add_argument('--domain_num', type=int, 
                        default=6, help='domain number')

    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)
    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)
    
    s = print_args(args, [])    
    print('=======hyper-parameter used========')
    print(s)
    metrics = ['acc', 'iou', 'miou', 'dice', 'precision', 'recall']
    record = {}
    acc_record = {}
    iou_record = {}
    miou_record = {}
    dice_record = {}
    precision_record = {}
    recall_record = {}
    
    
    best_val_record  = []
    
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_dice, best_valid_acc= 0, 0
    valid_acc, valid_dice, valid_iou, valid_precision, valid_recall, valid_miou = 0, 0, 0, 0, 0, 0
    target_acc, target_dice, target_iou, target_precision, target_recall, target_miou = 0, 0, 0, 0, 0, 0
    print('===========start training===========')
    
    sss = time.time()
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data) for data in next(train_minibatches_iterator)]
            step_vals = algorithm.update(minibatches_device, opt, sch, epoch)

        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in loss_list:
                s += (item+'_loss:%.4f,' % step_vals[item])
            print(s[:-1])
            s = ''
            for item in acc_type_list:
                result = [modelopera.accuracy(algorithm, eval_loaders[i]) for i in eval_name_dict[item]]
                result = torch.tensor(result)
                result = torch.mean(result, axis = 0)
                mean_acc, mean_iou, mean_miou, mean_dice, mean_precision, mean_recall = list(result)
                
                acc_record[item] = mean_acc
                iou_record[item] = mean_iou
                miou_record[item] = mean_miou
                dice_record[item] = mean_dice
                precision_record[item] = mean_precision
                recall_record[item] = mean_recall
                s += (item+'_acc:%.4f,' % acc_record[item])
                s += (item+'_iou:%.4f,' % iou_record[item])
                s += (item+'_miou:%.4f,' % miou_record[item])
                s += (item+'_dice:%.4f,' % dice_record[item])
                s += (item+'_precision:%.4f,' % precision_record[item])
                s += (item+'_recall:%.4f,' % recall_record[item])
            print(s[:-1])
            if dice_record['valid'] > best_valid_dice:
                best_valid_dice = dice_record['valid']
                
                valid_acc = acc_record['target']
                valid_dice = dice_record['valid']
                valid_iou = iou_record['valid']
                valid_miou = miou_record['valid']
                valid_precision = precision_record['valid']
                valid_recall = recall_record['valid']
                
                target_acc = acc_record['target']
                target_dice = dice_record['target']
                target_iou = iou_record['target']
                target_miou = miou_record['target']
                target_precision = precision_record['target']
                target_recall = recall_record['target']
                save_checkpoint(f'model_val.pkl', algorithm, args)
                # algorithm = algorithm.cuda()
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time()-sss))
            # algorithm_dict = algorithm.state_dict()

    # save_checkpoint('model.pkl', algorithm, args)

    print('valid dice: %.4f' % best_valid_dice)
    print('DG result acc: %.4f' % target_acc)
    print('DG result dice: %.4f' % target_dice)
    print('DG result iou: %.4f' % target_iou)
    print('DG result miou: %.4f' % target_miou)
    print('DG result precision: %.4f' % target_precision)
    print('DG result reacll: %.4f' % target_recall)

    with open(os.path.join(args.output, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        
        f.write('algorithm %s\n' % args.algorithm)
        f.write('test_envs: %.4f\n' % args.test_envs[0])
        f.write('alpha: %.4f\n' % args.alpha)
        f.write('mixupalpha: %.4f\n' % args.mixupalpha)
        f.write('bce_weight: %.4f\n' % args.bce_weight)
        f.write('lr: %.4f\n' % args.lr)

        f.write('DG result valid acc: %.4f\n' % valid_acc)
        f.write('DG result valid dice: %.4f\n' % valid_dice)
        f.write('DG result valid iou: %.4f\n' % valid_iou)
        f.write('DG result valid miou: %.4f\n' % valid_miou)
        f.write('DG result valid precision: %.4f\n' % valid_precision)
        f.write('DG result valid recall: %.4f\n' % valid_recall)
        
        f.write('DG result acc: %.4f\n' % target_acc)
        f.write('DG result dice: %.4f\n' % target_dice)
        f.write('DG result iou: %.4f\n' % target_iou)
        f.write('DG result miou: %.4f\n' % target_miou)
        f.write('DG result precision: %.4f\n' % target_precision)
        f.write('DG result recall: %.4f\n' % target_recall)

