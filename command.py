dataset='medical'
# algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx') 
algorithm=('DAMX') 
test_envs=0
gpu_id=0
data_dir='data'
max_epoch=120
net='vgg'
task='img_dg'
alpha=0.1
mixupalpha=0.2
lr=0.001
batch_size=12
count = 0
cuda = 0
out_dir = 'output'
seed = 0
for test_envs in [2, 3]:
    for mixupalpha in [0.1]:
        for bce_weight in [0.5, 1]:
            output = f'{out_dir}/{algorithm}VGG/output{count}'
            print(f'CUDA_VISIBLE_DEVICES={cuda} python train.py --data_dir {data_dir} --max_epoch {max_epoch} --net {net} --alpha {alpha} --task {task} --output {output} --test_envs {test_envs} --dataset {dataset} --algorithm {algorithm} --lr {lr} --mixupalpha {mixupalpha} --batch_size {batch_size} --bce_weight {bce_weight} --seed {seed} &&')
            count += 1
        print('==============================')
    cuda = 1 - cuda
    

# out_dir = '/data/output/DeepDGSEED'
# alpha = 0.1
# mixupalpha = 0.1
# lr = 0.001
# for seed in range(0,10):
#     if count > 4:
#         cuda = 1
#     else:
#         cuda = 0
#     output = f'{out_dir}/{algorithm}/output{count}'
    
#     print(f'CUDA_VISIBLE_DEVICES={cuda} python train.py --data_dir {data_dir} --max_epoch {max_epoch} --net {net} --alpha {alpha} --task {task} --output {output} --test_envs $test_envs --dataset {dataset} --algorithm {algorithm} --mixupalpha {mixupalpha} --batch_size {batch_size} --seed {seed} &&')
#     count += 1
    
          
        

# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_dir /data/PACS/ --max_epoch 120 --net resnet18 --alpha 0.5 --task img_dg --output /data/DeepDG/DAMX/output13 --test_envs 0 --dataset PACS --algorithm DAMX --mixupalpha 0.2 &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_dir /data/PACS/ --max_epoch 120 --net resnet18 --alpha 1 --task img_dg --output /data/DeepDG/DAMX/output14 --test_envs 0 --dataset PACS --algorithm DAMX --mixupalpha 0.1 &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_dir /data/PACS/ --max_epoch 120 --net resnet18 --alpha 1 --task img_dg --output /data/DeepDG/DAMX/output15 --test_envs 0 --dataset PACS --algorithm DAMX --mixupalpha 0.2 &


