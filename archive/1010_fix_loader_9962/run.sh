CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 1e-5 --batch-size 8 | tee -a log/tmp/cbam_base_1e-5.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 1e-5 --batch-size 8 | tee -a log/tmp/cbam_resize_1e-5.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 2e-5 --batch-size 8 | tee -a log/tmp/cbam_resize_2e-5.txt

#CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_18 --lr 2e-5 --batch-size 8 | tee -a log/tmp/18_1.txt
#CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_18 --lr 4e-5 --batch-size 8 | tee -a log/tmp/18_2.txt
#CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_18 --lr 8e-5 --batch-size 8 | tee -a log/tmp/18_3.txt


# CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 2e-5 --batch-size 8 | tee -a log/tmp/cbam.txt
