import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_scheduler
import random
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
import sentencepiece as spm
import torchsort
from tqdm import tqdm
import os
import numpy as np
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
import transformers
from typing import Optional, Dict, Sequence
from sft_dataset import DataCollatorForSupervisedDataset, SupervisedDataset, DollyDataset
from llama.modeling_llama import LlamaForCausalLM, LlamaConfig
import argparse

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



def seed_it(seed):
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = True  
    torch.manual_seed(seed)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def calculate_perplexity(logits, labels):
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
    perplexity = torch.exp(torch.mean(ce_loss)).item()
    return perplexity

data_id = 0
def init_netinput(batch_size, dataset, tokenizer, max_length):
    global data_id
    batch = []

    for i in range(batch_size):
        input_ids = tokenizer.encode(dataset[data_id]['text'], truncation=True, max_length=max_length, add_special_tokens=True)
        data_id += 1
        input_ids.append(2)

        while len(input_ids) < max_length:
            input_ids.extend(tokenizer.encode(dataset[data_id]['text'], truncation=True, max_length=max_length, add_special_tokens=True))
            data_id += 1
            input_ids.append(2)
        input_ids = input_ids[:max_length]
        batch.append(input_ids)

    return torch.tensor(batch)


def kl_divergence(p, q):
    """
    Compute KL divergence between two distributions p and q.
    """
    return torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=-1)

def kl_loss(teacher_distribution, student_distribution):
    """
    Compute knowledge distillation loss using KL divergence.
    """

    # Calculate KL divergence
    kl_loss = kl_divergence(teacher_distribution, student_distribution)

    # Average over the batch and sequence dimensions
    kl_loss = torch.mean(kl_loss, dim=-1)

    kl_loss = torch.mean(kl_loss, dim=-1)

    return kl_loss


def rkl_loss(teacher_distribution, student_distribution):
    """
    Compute reward knowledge distillation loss using KL divergence.
    """

    return kl_loss(student_distribution, teacher_distribution)

def spearman_loss(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)

    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

def topk_spearman_loss(student_logits, teacher_logits, k=15, **kw):
    _, tea_topk_indices = torch.topk(teacher_logits, k, dim=-1)
    _, stu_topk_indices = torch.topk(student_logits, k, dim=-1)

    # We have approximated the ranking loss calculation for batch of data using 'torch.unique' to a certain extent, which has effectively improved computational efficiency at the cost of some minor errors.
    union_indices = torch.unique(torch.cat((tea_topk_indices, stu_topk_indices), dim=-1), dim=-1)

    tea_topk_indices,_ = torch.sort(union_indices, dim=-1)
    length = tea_topk_indices.size()[-1]

    tea_topk_values = torch.gather(teacher_logits, -1, tea_topk_indices)
    stu_topk_values = torch.gather(student_logits, -1, tea_topk_indices)

    stu_topk_values = torchsort.soft_rank(stu_topk_values.view(-1, length), **kw)

    tea_topk_values = torchsort.soft_rank(tea_topk_values.view(-1, length), **kw)


    stu_topk_values = stu_topk_values - stu_topk_values.mean(dim=1, keepdim=True)
    stu_topk_values = stu_topk_values / stu_topk_values.norm(dim=1, keepdim=True)
    tea_topk_values = tea_topk_values - tea_topk_values.mean(dim=1, keepdim=True)
    tea_topk_values = tea_topk_values / tea_topk_values.norm(dim=1, keepdim=True)

    dot_product = (stu_topk_values * tea_topk_values).sum(dim=1)

    return torch.mean(dot_product)



def tvd_loss(teacher_logits, student_logits):
    absolute_differences = torch.sum(torch.abs(teacher_logits - student_logits), dim=-1)

    return 0.5 * absolute_differences
    
def CEloss(net_output, target):
    # net_logits = F.log_softmax(net_output, dim=-1)
    loss_fct = CrossEntropyLoss()
    a, b, vocab_size = net_output.size()
    loss = loss_fct(net_output.contiguous().view(-1, vocab_size), target.contiguous().view(-1))
    
    return loss

def topk_consistency(teacher_logits, student_logits, k):
    # 获取teacher和student的topk预测
    _, teacher_topk_idx = torch.topk(teacher_logits, k, dim=-1)
    _, student_topk_idx = torch.topk(student_logits, k, dim=-1)

    # 判断student的topk预测是否在teacher的topk预测内
    consistency = torch.zeros_like(teacher_topk_idx, dtype=torch.float)
    for i in range(k):
        consistency += (teacher_topk_idx == student_topk_idx[:, i].view(-1, 1)).float()

    # 计算正确率
    accuracy = consistency.sum(dim=1) / k
    
    return accuracy



def js_divergence(p, q):
    """
    Compute JS divergence between two distributions p and q.
    """
    # Calculate the mean distribution
    m = 0.5 * (p + q)
    
    # Calculate KL divergences
    kl_p_m = kl_divergence(p, m)
    kl_q_m = kl_divergence(q, m)
    
    # Calculate JS divergence
    js_div = 0.5 * kl_p_m + 0.5 * kl_q_m
    
    return js_div

def js_loss(teacher_distribution, student_distribution):
    """
    Compute knowledge distillation loss using JS divergence.
    """
    # Calculate JS divergence
    js_loss = js_divergence(teacher_distribution, student_distribution)
    
    # Average over the batch and sequence dimensions
    js_loss = torch.mean(js_loss, dim=-1)
    js_loss = torch.mean(js_loss, dim=-1)
    
    return js_loss



def only_rank_loss(teacher_logits, student_logits, k=10):
    
    rank_loss = 1 - topk_spearman_loss(student_logits, teacher_logits, k)

    
    if torch.isnan(rank_loss):
        print('nan')
        teacher_logits = F.softmax(teacher_logits, dim=-1)
        student_logits = F.softmax(student_logits, dim=-1)
        final_loss = kl_loss(teacher_logits, student_logits)
    else:
        final_loss =  rank_loss
        # final_loss = 0.5 * k * rank_loss + (1 - k) * v_loss

    return final_loss



def distill_loss(teacher_logits, student_logits, distill_obj, use_ranking_loss, ranking_range, ranking_loss_magnification):
    k = ranking_range
    if use_ranking_loss:
        rank_loss = 1 - topk_spearman_loss(student_logits, teacher_logits, k)
    teacher_logits = F.softmax(teacher_logits, dim=-1)
    student_logits = F.softmax(student_logits, dim=-1)
    if distill_obj == 'kl':
        v_loss = kl_loss(teacher_logits, student_logits)
    elif distill_obj == 'rkl':
        v_loss = rkl_loss(teacher_logits, student_logits)
    elif distill_obj == 'js':
        v_loss = js_loss(teacher_logits, student_logits)
    elif distill_obj == 'tvd':
        v_loss = tvd_loss(teacher_logits, student_logits)

    if use_ranking_loss:
        # To avoid rare operator anomalies.2
        if torch.isnan(rank_loss):
            print('error with ranking loss')
            final_loss = v_loss
        else:
            final_loss =  rank_loss +  v_loss / ranking_loss_magnification

        return final_loss
    else:
        return v_loss

def CEloss(net_output, target):
    # net_logits = F.log_softmax(net_output, dim=-1)
    loss_fct = CrossEntropyLoss()
    a, b, vocab_size = net_output.size()
    loss = loss_fct(net_output.contiguous().view(-1, vocab_size), target.contiguous().view(-1))
    
    return loss




def distill():
    parser = argparse.ArgumentParser(description="RLKD Demo")

    parser.add_argument('--task', type=str, default='gsm8k', choices=['gsm8k', 'xsum', 'dolly'])
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--gradient_acc', type=int, default=32)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--teacher_path', type=str, default="/ssd/data/typeng/models/gsm8k-rft-llama7b2-u13b")
    parser.add_argument('--student_path', type=str, default="/ssd/data/typeng/models/tinyllama")
    parser.add_argument('--tokenizer_path', type=str, default="/ssd/data/typeng/models/gsm8k-rft-llama7b2-u13b")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--data_path', type=str, default='/share/typeng/projects/RLKD/sft/data/gsm8k/train_use.jsonl')
    parser.add_argument('--distill_obj', type=str, default='kl', choices=['kl', 'rkl', 'js', 'tvd'])
    parser.add_argument('--use_ranking_loss', type=bool, default=True)
    parser.add_argument('--ranking_range', type=int, default=5)
    parser.add_argument('--ranking_loss_magnification', type=float, default=2.0)
    parser.add_argument('--save_path', type=str, default='/ssd/data/typeng/checkpoints/rlkd/sft/test')
    args = parser.parse_args()

    num_epochs = args.epoch

    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_acc)
    device = accelerator.device
    seed = args.seed
    seed_it(seed)
    print('seed:', seed)
    batch_size=args.batch_size
    max_length=args.seq_len

    data_file = args.data_path

    teacher_model_path = args.teacher_path
    student_model_path = args.student_path

    new_training = True

    teacher_config = LlamaConfig.from_pretrained(teacher_model_path)
    teacher_model = LlamaForCausalLM.from_pretrained(teacher_model_path, torch_dtype=torch.bfloat16, config=teacher_config).to(device)


    # load student
    student_config = LlamaConfig.from_pretrained(student_model_path)
    student_model = LlamaForCausalLM.from_pretrained(student_model_path, config=student_config).to(device)

    if new_training:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right", use_fast=False, model_max_length=max_length)
        # if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=student_model,
        )
        # if "llama" in base_model:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right", use_fast=False, model_max_length=max_length)

    optimizer = optim.AdamW(student_model.parameters(), lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=100000,
    )
    # lr_scheduler = get_scheduler(
    #     name='constant',
    #     optimizer=optimizer,
    #     num_training_steps=num_step*num_epochs,
    # )
    student_model, optimizer, lr_scheduler = accelerator.prepare(
        student_model, optimizer, lr_scheduler
    )
    total_step = 0

    if args.task == 'gsm8k':
        dataset = SupervisedDataset(data_file, tokenizer)
    else:
        dataset = DollyDataset(data_file, tokenizer)
    dataset.shard(accelerator.process_index, args.num_gpus)
    
    num_step = len(dataset) // batch_size

    for epoch in range(num_epochs):
        # ----------------------------------------
        # Training
        # ----------------------------------------
        student_model.train()
        teacher_model.eval()

        data_id = 0
        with tqdm(total= num_step, disable=not accelerator.is_local_main_process) as _tqdm:
            for step in range(num_step):
                total_step += 1
                # data = dataset[step]
                with accelerator.accumulate(student_model):
                    len_querys = []
                    len_inputs = []
                    inputs = []
                    for i in range(batch_size):
                        data = dataset[data_id]
                        data_id += 1
                        # print(data['input_ids']+data['labels'])
                        query = tokenizer(data['input_ids'], return_tensors='pt', max_length=max_length, truncation=True)['input_ids']
                        input = tokenizer(data['input_ids'] + data['labels'], return_tensors='pt', max_length=max_length, truncation=True)['input_ids']

                        inputs.append(input.squeeze())

                        len_inputs.append(input.size()[-1])
                        len_querys.append(query.size()[-1])

                    # print(len_labels)
                    # print(len_querys)
                    input_ids = torch.nn.utils.rnn.pad_sequence(
                        inputs, batch_first=True, padding_value=tokenizer.pad_token_id
                    ).to(device)
                    
                    attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device)

                    student_logits = student_model(input_ids, attention_mask=attention_mask).logits
                    with torch.no_grad():
                        teacher_logits = teacher_model(input_ids, attention_mask=attention_mask).logits

                
                    _, seq_len, vocab_size = teacher_logits.size()

                    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
                    for i in range(batch_size):
                        mask[i, len_querys[i]:len_inputs[i]] = 1

                    teacher_logits_batch = teacher_logits[mask].view(-1, vocab_size)
                    student_logits_batch = student_logits[mask].view(-1, vocab_size)

                    loss = distill_loss(teacher_logits_batch, student_logits_batch, args.distill_obj, args.use_ranking_loss, args.ranking_range, args.ranking_loss_magnification)

                    # print(loss)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(student_model.parameters(), 1.0)
                    # loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    _tqdm.set_postfix(consistency='{:.2f}'.format(loss))
                    _tqdm.update(1)

        # ----------------------------------------
        # save model
        # ----------------------------------------
        save_dir = args.save_path + '_epoch' + str(epoch)        


        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(student_model)
        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save, state_dict=accelerator.get_state_dict(student_model))

distill()

