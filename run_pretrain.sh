export CUDA_VISIBLE_DEVICES=0,1,2,3


accelerate launch --main_process_port 7316 pretrain/kd_pretrain.py

