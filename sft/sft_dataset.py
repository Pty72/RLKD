
import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, AutoTokenizer

import json



IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]

        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        if 'instruction' in list_data_dict[0]:
            pass
        else:
            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])
            list_data_dict = [{'instruction':data['query'].split('\n')[0], 'input':get_input(data['query']), 'output':data['response']} for data in list_data_dict]
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

    def shard(self, i, total_indices):
        
        start_idx = i * (len(self.sources) // total_indices)
        end_idx = (i + 1) * (len(self.sources) // total_indices)
        print(start_idx, end_idx)
        self.sources = self.sources[start_idx:end_idx]
        self.targets = self.targets[start_idx:end_idx]

class DollyDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(DollyDataset, self).__init__()
        logging.warning("Loading data...")
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]

        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        # if 'instruction' in list_data_dict[0]:
        #     pass
        # else:
        # def get_input(query):
        #     if query.find('\n') == -1:
        #         return ''
        #     return '\n'.join(query.split('\n')[1:])
        
        list_data_dict = [{'instruction':data['instruction'], 'input':data['context'], 'output':data['response']} for data in list_data_dict]
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

    def shard(self, i, total_indices):
        
        start_idx = i * (len(self.sources) // total_indices)
        end_idx = (i + 1) * (len(self.sources) // total_indices)
        print(start_idx, end_idx)
        self.sources = self.sources[start_idx:end_idx]
        self.targets = self.targets[start_idx:end_idx]




    
tokenizer = AutoTokenizer.from_pretrained("/ssd/data/typeng/models/gsm8k-rft-llama7b2-u13b")
# if tokenizer.pad_token is None:
#     smart_tokenizer_and_embedding_resize(
#         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
#         tokenizer=tokenizer,
#         model=model,
#     )
print(tokenizer.pad_token_id)
dataset = DollyDataset('/share/typeng/projects/distillation/datasets/dolly_train.jsonl', tokenizer)

print(dataset[0])
# max_len = 0
# long_num = 0
# # data = dataset[0]
# # query = tokenizer(data['input_ids'], return_tensors='pt')['input_ids']
# for data in dataset:
#     label = tokenizer(data['input_ids'] + data['labels'])['input_ids']
#     max_len = max(max_len, len(label))
#     if len(label) > 4000:
#         long_num += 1
#         print(data['labels'])
# # input = torch.cat((query, label), dim=-1)
# print(max_len)
# print(long_num)
# print(query, label)
# batch_size = 2
# seq_len = 10
# vocab_size = 32001

# # 生成随机张量
# teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
# student_logits = torch.randn(batch_size, seq_len, vocab_size)

# _, seq_len, vocab_size = teacher_logits.size()

# student_logits_batch = None
# teacher_logits_batch = None
# vocab_size=teacher_logits.size()[-1]
# len_querys = [3, 1]
# len_inputs = [7, 10]

# # 构建掩码
# mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
# for i in range(batch_size):
#     mask[i, len_querys[i]:len_inputs[i]] = 1

# print(mask)
# teacher_logits_batch = teacher_logits[mask].view(-1, vocab_size)
# student_logits_batch = student_logits[mask].view(-1, vocab_size)

# print(teacher_logits_batch.shape)
# print(student_logits_batch.shape)