CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 120 --net resnet50 --alpha 0.1 --task img_dg --output output_0304/DAMXX/output0 --test_envs 0 --dataset medical --algorithm DAMXX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 0 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 120 --net resnet50 --alpha 0.1 --task img_dg --output output_0304/DAMXX/output1 --test_envs 0 --dataset medical --algorithm DAMXX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 0 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 120 --net resnet50 --alpha 0.1 --task img_dg --output output_0304/DAMXX/output2 --test_envs 0 --dataset medical --algorithm DAMXX --lr 0.001 --mixupalpha 0.2 --batch_size 12 --bce_weight 1 --seed 0 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 120 --net resnet50 --alpha 0.1 --task img_dg --output output_0304/DAMXX/output3 --test_envs 0 --dataset medical --algorithm DAMXX --lr 0.001 --mixupalpha 0.2 --batch_size 12 --bce_weight 0.5 --seed 0