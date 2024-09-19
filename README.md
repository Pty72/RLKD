# RLKD

After setting up the environment, first configure the distributed training environment using the **accelerate** library. Use the following command to specify the number of GPUs and DeepSpeed settings.

```
accelerate config
```

+ Distillation for Pre-Training Task

  ```
  bash run_pretrain.sh
  ```
  You can view and change the adjustable parameter settings in pretrain/kd_pretrain.py.

+ Distillation for Downstream Tasks

  ```
  bash run_sft.sh
  ```
  You can view and change the adjustable parameter settings in sft/kd_sft.py.

