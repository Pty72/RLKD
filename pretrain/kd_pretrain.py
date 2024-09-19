import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_scheduler
import random
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
import torchsort
from tqdm import tqdm
import os
import numpy as np
from llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
import argparse

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

    return 0.5 * torch.mean(absolute_differences)


def CEloss(net_output, target):
    # net_logits = F.log_softmax(net_output, dim=-1)
    loss_fct = CrossEntropyLoss()
    a, b, vocab_size = net_output.size()
    loss = loss_fct(net_output.contiguous().view(-1, vocab_size), target.contiguous().view(-1))
    
    return loss

def topk_consistency(teacher_logits, student_logits, k):
    # 获取teacher和student的topk预测
    _, teacher_topk_idx = torch.topk(teacher_logits, k, dim=1)
    _, student_topk_idx = torch.topk(student_logits, k, dim=1)

    # 判断student的topk预测是否在teacher的topk预测内
    consistency = torch.zeros_like(teacher_topk_idx, dtype=torch.float)
    for i in range(k):
        consistency += (teacher_topk_idx == student_topk_idx[:, i].view(-1, 1)).float()

    # 计算正确率
    accuracy = consistency.sum(dim=1, keepdim=True) / k
    
    return accuracy.mean()

def same_topk(teacher_logits, student_logits, k):
    # 获取teacher和student的topk预测
    _, teacher_topk_idx = torch.topk(teacher_logits, k, dim=1)
    _, student_topk_idx = torch.topk(student_logits, k, dim=1)

    consistency_mask = torch.eq(teacher_topk_idx, student_topk_idx).all(dim=1)

    consistency_rate = consistency_mask.float().mean().item()

    return consistency_rate

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

def verify_ood(student_model, teacher_model, tokenizer, device, dataset=None, max_length=256):

    dataset = load_dataset("json", data_files=dataset)['train']
    student_model.eval()
    teacher_model.eval()

    total_perplexity = 0.0
    total_samples = 0

    top30_covery = 0
    top20_covery = 0
    top10_covery = 0
    top5_covery = 0
    top3_covery = 0

    consistency_3 = 0
    consistency_5 = 0
    consistency_1 = 0
    consistency_2 = 0
    consistency_4 = 0

    for example in tqdm(dataset):
        text = example['text']
        
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
        if len(input_ids[0]) < 4:
            continue

        with torch.no_grad():
            stu_outputs = student_model(input_ids)
            tea_outputs = teacher_model(input_ids)
        
        # ***************************
        # 1.PPL
        # ***************************
        logits = stu_outputs.logits[:, :-1, :]  
        labels = input_ids[:, 1:]

        perplexity = calculate_perplexity(logits, labels)

        total_perplexity += perplexity
        

        # ***************************
        # 2.TopK OR
        # ***************************
        stu_logits = stu_outputs.logits
        tea_logits = tea_outputs.logits
        top30_covery += topk_consistency(tea_logits.squeeze(0), stu_logits.squeeze(0), 30)
        top20_covery += topk_consistency(tea_logits.squeeze(0), stu_logits.squeeze(0), 20)
        top10_covery += topk_consistency(tea_logits.squeeze(0), stu_logits.squeeze(0), 10)
        top5_covery += topk_consistency(tea_logits.squeeze(0), stu_logits.squeeze(0), 5)
        top3_covery += topk_consistency(tea_logits.squeeze(0), stu_logits.squeeze(0), 3)
        # ***************************
        # 2.Topk CR
        # ***************************
        consistency_4 += same_topk(tea_logits.squeeze(0), stu_logits.squeeze(0), 4)
        consistency_3 += same_topk(tea_logits.squeeze(0), stu_logits.squeeze(0), 3)
        consistency_5 += same_topk(tea_logits.squeeze(0), stu_logits.squeeze(0), 5)
        consistency_2 += same_topk(tea_logits.squeeze(0), stu_logits.squeeze(0), 2)
        consistency_1 += same_topk(tea_logits.squeeze(0), stu_logits.squeeze(0), 1)

        total_samples += 1

    average_perplexity = total_perplexity / total_samples
    print(f"Average Perplexity: {average_perplexity}")

    average_30_covery = top30_covery / total_samples
    print(f"Top30 OR: {average_30_covery}")

    average_20_covery = top20_covery / total_samples
    print(f"Top20 OR: {average_20_covery}")

    average_10_covery = top10_covery / total_samples
    print(f"Top10 OR: {average_10_covery}")

    average_5_covery = top5_covery / total_samples
    print(f"Top5 OR: {average_5_covery}")

    average_3_covery = top3_covery / total_samples
    print(f"Top3 OR: {average_3_covery}")

    average_5_consistency = consistency_5 / total_samples
    print(f"Top5 CR: {average_5_consistency}")

    average_4_consistency = consistency_4 / total_samples
    print(f"Top4 CR: {average_4_consistency}")

    average_3_consistency = consistency_3 / total_samples
    print(f"Top3 CR: {average_3_consistency}")

    average_2_consistency = consistency_2 / total_samples
    print(f"Top2 CR: {average_2_consistency}")

    average_1_consistency = consistency_1 / total_samples
    print(f"Top1 CR: {average_1_consistency}")

    return average_perplexity


def distill():
    parser = argparse.ArgumentParser(description="RLKD Demo")

    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--train_step', type=int, default=2000)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--gradient_acc', type=int, default=32)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--teacher_path', type=str, default="/share/typeng/models/llama-2-7b")
    parser.add_argument('--student_path', type=str, default="/ssd/data/typeng/models/tinyllama")
    parser.add_argument('--tokenizer_path', type=str, default="/share/typeng/models/llama-2-7b")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--data_path', type=str, default='/ssd/data/typeng/datasets/SlimPajama/SlimPajama-6B/jsonl/mix_40w_0')
    parser.add_argument('--distill_obj', type=str, default='kl', choices=['kl', 'rkl', 'js', 'tvd'])
    parser.add_argument('--use_ranking_loss', type=bool, default=True)
    parser.add_argument('--ranking_range', type=int, default=15)
    parser.add_argument('--ranking_loss_magnification', type=float, default=2.0)
    parser.add_argument('--test_file', type=str, default='/ssd/data/typeng/datasets/SlimPajama/SlimPajama-6B/jsonl/test_5k.jsonl')
    parser.add_argument('--save_path', type=str, default='/ssd/data/typeng/checkpoints/rlkd/pretrain/test')
    args = parser.parse_args()

    num_epochs = args.epoch
    num_step = args.train_step * args.gradient_acc
    
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_acc)
    device = accelerator.device
    seed = args.seed
    seed_it(seed)
    print('seed:', seed)
    batch_size=args.batch_size
    max_length=args.seq_len
    
    
    teacher_model_path = args.teacher_path
    student_model_path = args.student_path

    teacher_config = LlamaConfig.from_pretrained(teacher_model_path)
    teacher_model = LlamaForCausalLM.from_pretrained(teacher_model_path, torch_dtype=torch.bfloat16, config=teacher_config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)


    # load student
    student_config = LlamaConfig.from_pretrained(student_model_path)
    student_model = LlamaForCausalLM.from_pretrained(student_model_path, torch_dtype=torch.bfloat16, config=student_config).to(device)


    optimizer = optim.AdamW(student_model.parameters(), lr=args.learning_rate)

    # lr_scheduler = get_scheduler(
    #     name='cosine',
    #     optimizer=optimizer,
    #     num_warmup_steps=1000,
    #     num_training_steps=num_step*num_epochs,
    # )
    
    lr_scheduler = get_scheduler(
        name='constant',
        optimizer=optimizer,
        num_training_steps=num_step*num_epochs,
    )

    student_model, optimizer, lr_scheduler = accelerator.prepare(
        student_model, optimizer, lr_scheduler
    )
    total_step = 0


    dataset = load_dataset('json', data_files=args.data_path, split='train')
    dataset = dataset.shard(num_shards=args.num_gpus, index=accelerator.process_index)
    print('Finished load data from {}'.format(args.data_path))

    for epoch in range(num_epochs):
        # ----------------------------------------
        # Training
        # ----------------------------------------
        student_model.train()
        teacher_model.eval()
        
        input = None
        with tqdm(total= num_step, disable=not accelerator.is_local_main_process) as _tqdm:
            for step in range(num_step):
                total_step += 1
                
                with accelerator.accumulate(student_model):
                    input = init_netinput(batch_size, dataset, tokenizer, max_length)

                    input = input.to(device)
                    
                    student_logits = student_model(input).logits
                    with torch.no_grad():
                        teacher_logits = teacher_model(input).logits
                    # consistency = topk_consistency(teacher_logits, student_logits, 20)
                    # loss = distill_loss(teacher_logits, student_logits, consistency)
                    loss = distill_loss(teacher_logits, student_logits, args.distill_obj, args.use_ranking_loss, args.ranking_range, args.ranking_loss_magnification)
                    # loss = CEloss(student_logits[:, :-1, :], input[:, 1:])
                    # get_next_token_id(teacher_logits, student_logits)

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
        # validation
        # ----------------------------------------
        global data_id
        data_id = 0

        # ----------------------------------------
        # save model
        # ----------------------------------------
        save_dir = args.save_path + '_epoch' + str(epoch)


        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(student_model)
        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save, state_dict=accelerator.get_state_dict(student_model))

        if accelerator.is_local_main_process:
            print("\n ========================= START EVALUATION =========================")
            print('epoch {}:'.format(epoch))
            verify_datasets = [args.test_file]
            for data in verify_datasets:
                print('\nStart verify in {}'.format(data))
                verify_ood(student_model, teacher_model, tokenizer, device, dataset=data, max_length=max_length)

distill()

