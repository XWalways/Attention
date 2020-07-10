python imagenet.py -a sge_resnet101 --data /your/imagenet data/path/ --epochs 100 --schedule 30 60 90 --gamma 0.1 -c checkpoints/sge_resnet101 --gpu-id 0,1,2,3,4,5,6,7

#python -m torch.distributed.launch --nproc_per_node=8 imagenet_fast.py -a sge_resnet50 --data /your/imagenet data/path/ --epochs 100 --schedule 30 60 90 --wd 1e-4 --gamma 0.1 -c checkpoints/sge_resnet50 --train-batch 32 --opt-level O0 --wd-all --label-smoothing 0. --warmup 0
