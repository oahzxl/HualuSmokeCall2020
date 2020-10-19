CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 2e-5 --batch-size 8 | tee -a log/tmp/log_wsl_3.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 1e-5 --batch-size 8 | tee -a log/tmp/log_wsl_4.txt

CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnet18 --lr 3e-5 --batch-size 8 | tee -a log/tmp/log_res.txt
