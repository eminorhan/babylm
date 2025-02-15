#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import argparse
import logging
import math
import os
import random
from itertools import chain

import datasets
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    default_data_collator,
)
from transformers.utils.versions import require_version


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def check_file_extensions(file_list):
    # Extract the extension from the first file
    extension = file_list[0].split('.')[-1]
    
    # Check that all files have one of the allowed extensions
    for file_name in file_list:
        assert file_name.endswith(('.csv', '.json', '.txt')), "Invalid file extension."
    
    # Check that all files have the same extension
    for file_name in file_list:
        assert file_name.split('.')[-1] == extension, "Files have different extensions."

    print("All files have the same extension: {}.".format(extension))


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune large language models on causal language modeling tasks")
    parser.add_argument('--train_files', nargs='+', help="list of files containing the training data")
    parser.add_argument('--val_files', nargs='+', help="list of files containing the validation data")
    parser.add_argument("--tokenizer_file", type=str, default=None, help="Pretrained tokenizer path")
    parser.add_argument("--block_size", type=int, default=None, help="The training dataset will be truncated to blocks of this size (after tokenization) for training.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--preprocessing_num_workers", type=int, default=16, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files.")

    args = parser.parse_args()

    # Sanity check for file extensions
    check_file_extensions(args.train_files)
    check_file_extensions(args.val_files)

    return args


def main():
    args = parse_args()
    print(args)

    # Initialize the accelerator
    accelerator = Accelerator()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, 'load_dataset' function guarantee that only one local process can concurrently download the dataset.
    data_files = {"train": args.train_files, "validation": args.val_files}
    dataset_args = {}
    extension = args.train_files[0].split(".")[-1]
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if args.tokenizer_file:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=args.tokenizer_file,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
            )
    else:
        raise ValueError("Please specify a valid tokenizer file path.")

    print('Tokenizer len:', len(tokenizer))

    # Preprocessing the datasets. First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). Picking 1024 instead. You can change that default value by passing --block_size xxx.")
            block_size = 1024
    else:
        block_size = args.block_size
        if args.block_size > tokenizer.model_max_length:
            logger.warning(f"The block_size passed ({args.block_size}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}.")
            block_size = tokenizer.model_max_length

    print('Block size:', block_size)
    print('Pad token id:', tokenizer.pad_token_id)

    # TODO: revert this back later on
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding='max_length', truncation=True, max_length=block_size)

    def tokenize_function_original(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function_original,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = tokenized_datasets["train"]    
    val_dataset = tokenized_datasets["validation"]

    len_train = len(train_dataset)
    len_val = len(val_dataset)
    print('train-val lengths:', len_train, len_val)

    train_lens = np.zeros(len_train)
    val_lens = np.zeros(len_val)

    for i in range(len_train):
        train_lens[i] = len(train_dataset[i]['input_ids'])
        if i % 10000 == 0:
            print('iter, tokenized story:', i, train_dataset[i])

    for i in range(len_val):
        val_lens[i] = len(val_dataset[i]['input_ids'])
        if i % 10000 == 0:
            print('iter, tokenized story:', i, val_dataset[i])

    print('Min, max, mean, median train lens:', np.min(train_lens), np.max(train_lens), np.mean(train_lens), np.median(train_lens))
    print('Min, max, mean, median val lens:', np.min(val_lens), np.max(val_lens), np.mean(val_lens), np.median(val_lens))

    print('max 100 train:', np.sort(train_lens)[-100:])
    print('max 100 val:', np.sort(val_lens)[-100:])

if __name__ == "__main__":
    main()