CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model res_cbam --lr 2e-4 --batch-size 8 | tee -a log/tmp/cbam_dif_2.txt
CUDA_VISIBLE_DEVICES=2 python3 main.py --mode train --model res_cbam --lr 4e-4 --batch-size 8 | tee -a log/tmp/cbam_dif_4.txt
