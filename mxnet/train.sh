python train_val.py --rec-train /path/to/train.rec --rec-train-idx /path/to/train.idx \
                    --rec-val /path/to/val.rec --rec-val-idx /path/to/val.idx \
                    --use-rec --batch-size 128 --num-gpus 8 -j 60 --num-epochs 120 --lr-mode cosine --warmup-epochs 5 \
                    --hybrid --no-wd --model se_resnet50 --save-dir params_se_resnet50 --log-dir logs_se_resnet50 --logging-file train_imagenet_se_resnet50.log
