CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model wsdan --lr 2e-4 --batch-size 8 | tee -a log/tmp/wsdan1.txt
CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model wsdan --lr 4e-4 --batch-size 8 | tee -a log/tmp/wsdan2.txt
