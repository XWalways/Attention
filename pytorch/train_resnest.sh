python train_dist.py --dataset imagenet --model resnest50 --lr-scheduler cos --epochs 270 --checkname resnest50_rt_bots \
                     --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2  --last-gamma --no-bn-wd --warmup-epochs 5 --dropblock-prob 0.1 --rectify

