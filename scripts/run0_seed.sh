test_envs=0
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net vgg --alpha 0.1 --task img_dg --output output_bestV/DAMX_seed/output1 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 1 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net vgg --alpha 0.1 --task img_dg --output output_bestV/DAMX_seed/output2 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 2 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net vgg --alpha 0.1 --task img_dg --output output_bestV/DAMX_seed/output3 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 3 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net vgg --alpha 0.1 --task img_dg --output output_bestV/DAMX_seed/output4 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 4 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net vgg --alpha 0.1 --task img_dg --output output_bestV/DAMX_seed/output5 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 5 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net vgg --alpha 0.1 --task img_dg --output output_bestV/DAMX_seed/output6 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 6