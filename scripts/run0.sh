CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_FCN_resnet/Mixup/output0 --test_envs 0 --dataset medical --algorithm Mixup --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 0 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_FCN_resnet/Mixup/output1 --test_envs 1 --dataset medical --algorithm Mixup --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 0 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_FCN_resnet/Mixup/output2 --test_envs 2 --dataset medical --algorithm Mixup --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 0 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_FCN_resnet/Mixup/output3 --test_envs 3 --dataset medical --algorithm Mixup --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 0 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_FCN_resnet/Mixup/output4 --test_envs 4 --dataset medical --algorithm Mixup --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 0 &&
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir data --max_epoch 100 --net resnet --alpha 0.1 --task img_dg --output output_FCN_resnet/Mixup/output5 --test_envs 5 --dataset medical --algorithm Mixup --lr 0.001 --mixupalpha 0.1 --batch_size 12 --bce_weight 0.5 --seed 0
