CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model hrnet --lr 6e-3 --batch-size 8 | tee -a log/tmp/hrnet1.txt
CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model hrnet --lr 1e-2 --batch-size 8 | tee -a log/tmp/hrnet2.txt

CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model res_cbam --lr 3e-5 --batch-size 8 | tee -a log/tmp/res_cbam1.txt
CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model res_cbam --lr 6e-5 --batch-size 8 | tee -a log/tmp/res_cbam2.txt

CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model wsdan --lr 6e-4 --batch-size 8 | tee -a log/tmp/ws+res3.txt
CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model wsdan --lr 1e-3 --batch-size 8 | tee -a log/tmp/ws+res3.txt

CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --h 600 --w 450 --model resnest --lr 1e-4 --batch-size 8 | tee -a log/tmp/resnest1.txt
CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --h 600 --w 450 --model resnest --lr 5e-5 --batch-size 8 | tee -a log/tmp/resnest2.txt
