CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 1e-4 --batch-size 8 | tee -a log/tmp/casa.txt

CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --h 450 --w 340 --lr 1e-4 --batch-size 8 | tee -a log/tmp/size1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --h 450 --w 450 --lr 1e-4 --batch-size 8 | tee -a log/tmp/size2.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --h 300 --w 300 --lr 1e-4 --batch-size 8 | tee -a log/tmp/size3.txt
