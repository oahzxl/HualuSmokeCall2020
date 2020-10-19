CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnest --lr 1e-4 --batch-size 8 | tee -a log/tmp/resnest1.txt

CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 1e-4 --sche cos --t0 2 --tm 2 --batch-size 8 | tee -a log/tmp/cbam1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 3e-4 --sche cos --t0 2 --tm 2 --batch-size 8 | tee -a log/tmp/cbam2.txt

CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 3e-4 --sche reduce --batch-size 8 | tee -a log/tmp/reduce1.txt
