test_envs=5
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_ERM2/output1 --test_envs $test_envs --dataset medical --algorithm ERM --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 10 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_ERM2/output2 --test_envs $test_envs --dataset medical --algorithm ERM --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 20 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_ERM2/output3 --test_envs $test_envs --dataset medical --algorithm ERM --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 30 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_ERM2/output4 --test_envs $test_envs --dataset medical --algorithm ERM --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 40 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_ERM2/output5 --test_envs $test_envs --dataset medical --algorithm ERM --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 50 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_ERM2/output6 --test_envs $test_envs --dataset medical --algorithm ERM --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 60
