test_envs=0
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet50 --alpha 0.1 --task img_dg --output output_best/DAMX/output0 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.0001 --mixupalpha 0.1 --batch_size 12 --bce_weight 1 --seed 42 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet50 --alpha 0.1 --task img_dg --output output_best/DAMX/output1 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.0001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 42 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet50 --alpha 0.1 --task img_dg --output output_best/DAMX/output2 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.0001 --mixupalpha 0.2 --batch_size 12 --bce_weight 1 --seed 42 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet50 --alpha 0.1 --task img_dg --output output_best/DAMX/output3 --test_envs $test_envs --dataset medical --algorithm DAMX --lr 0.0001 --mixupalpha 0.2 --batch_size 12 --bce_weight 0.5 --seed 42