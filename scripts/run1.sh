CUDA_VISIBLE_DEVICES=1 python train.py --data_dir data --max_epoch 120 --net resnet50 --alpha 0.1 --task img_dg --output output_0304/DAMX/output4 --test_envs 1 --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 0 &&
CUDA_VISIBLE_DEVICES=1 python train.py --data_dir data --max_epoch 120 --net resnet50 --alpha 0.1 --task img_dg --output output_0304/DAMX/output5 --test_envs 1 --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 0 &&
CUDA_VISIBLE_DEVICES=1 python train.py --data_dir data --max_epoch 120 --net resnet50 --alpha 0.1 --task img_dg --output output_0304/DAMX/output6 --test_envs 1 --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.2 --batch_size 12 --bce_weight 1 --seed 0 &&
CUDA_VISIBLE_DEVICES=1 python train.py --data_dir data --max_epoch 120 --net resnet50 --alpha 0.1 --task img_dg --output output_0304/DAMX/output7 --test_envs 1 --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.2 --batch_size 12 --bce_weight 0.5 --seed 0