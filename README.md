# [COLING2025]Enhancing Knowledge Distillation of Large Language Models through Efficient Multi-Modal Distribution Alignment

## Training

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


## Citations

Thank you for citing our work.
```
@misc{peng2024enhancingknowledgedistillationlarge,
      title={Enhancing Knowledge Distillation of Large Language Models through Efficient Multi-Modal Distribution Alignment}, 
      author={Tianyu Peng and Jiajun Zhang},
      year={2024},
      eprint={2409.12545},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.12545}, 
}
```
