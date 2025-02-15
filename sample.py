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
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import argparse
import logging
import os
from itertools import chain

import datasets
import torch

import transformers
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator
)
from transformers.utils.versions import require_version

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DATASETS = {
    "babylm_10M": {
        "train": ["data/train_10M/childes.txt", "data/train_10M/bnc_spoken.txt", "data/train_10M/gutenberg.txt", "data/train_10M/open_subtitles.txt", "data/train_10M/simple_wiki.txt", "data/train_10M/switchboard.txt"],
        "validation": ["data/dev/childes.txt", "data/dev/bnc_spoken.txt", "data/dev/gutenberg.txt", "data/dev/open_subtitles.txt", "data/dev/simple_wiki.txt", "data/dev/switchboard.txt"]
        }, 
    "babylm_100M": {
        "train": ["data/train_100M/childes.txt", "data/train_100M/bnc_spoken.txt", "data/train_100M/gutenberg.txt", "data/train_100M/open_subtitles.txt", "data/train_100M/simple_wiki.txt", "data/train_100M/switchboard.txt"],
        "validation": ["data/dev/childes.txt", "data/dev/bnc_spoken.txt", "data/dev/gutenberg.txt", "data/dev/open_subtitles.txt", "data/dev/simple_wiki.txt", "data/dev/switchboard.txt"]
        }, 
    "wikipedia_10M_1": ["eminorhan/wikipedia", "10M_1"],
    "wikipedia_10M_2": ["eminorhan/wikipedia", "10M_2"],
    "wikipedia_10M_3": ["eminorhan/wikipedia", "10M_3"],
    "wikipedia_100M_1": ["eminorhan/wikipedia", "100M_1"],
    "wikipedia_100M_2": ["eminorhan/wikipedia", "100M_2"],
    "wikipedia_100M_3": ["eminorhan/wikipedia", "100M_3"],
    "gutenberg_10M_1": ["eminorhan/gutenberg_en_dec24", "10M_1"],
    "gutenberg_10M_2": ["eminorhan/gutenberg_en_dec24", "10M_2"],
    "gutenberg_10M_3": ["eminorhan/gutenberg_en_dec24", "10M_3"],
    "gutenberg_100M_1": ["eminorhan/gutenberg_en_dec24", "100M_1"],
    "gutenberg_100M_2": ["eminorhan/gutenberg_en_dec24", "100M_2"],
    "gutenberg_100M_3": ["eminorhan/gutenberg_en_dec24", "100M_3"],
    "tinystories_10M_1": ["eminorhan/tinystories", "10M_1"],
    "tinystories_10M_2": ["eminorhan/tinystories", "10M_2"],
    "tinystories_10M_3": ["eminorhan/tinystories", "10M_3"],
    "tinystories_100M_1": ["eminorhan/tinystories", "100M_1"],
    "tinystories_100M_2": ["eminorhan/tinystories", "100M_2"],
    "tinystories_100M_3": ["eminorhan/tinystories", "100M_3"],
    "pythonedu_10M_1": ["eminorhan/python-edu", "10M_1"],
    "pythonedu_10M_2": ["eminorhan/python-edu", "10M_2"],
    "pythonedu_10M_3": ["eminorhan/python-edu", "10M_3"],
    "pythonedu_100M_1": ["eminorhan/python-edu", "100M_1"],
    "pythonedu_100M_2": ["eminorhan/python-edu", "100M_2"],
    "pythonedu_100M_3": ["eminorhan/python-edu", "100M_3"],
}


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
    parser = argparse.ArgumentParser(description="Sample from a trained model")
    parser.add_argument("--dataset_name", type=str, help="dataset", choices=DATASETS.keys())
    parser.add_argument("--save_prefix", type=str, default='', help="Informative string for saving purposes.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="Batch size (per device).")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of generated samples per prompt.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None, help="The training dataset will be truncated to blocks of this size (after tokenization) for training.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets.")
    parser.add_argument("--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)

    # Initialize the accelerator
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.ERROR)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output dir creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # In distributed training, 'load_dataset' function guarantee that only one local process can concurrently download the dataset.
    if args.dataset_name.startswith("babylm"):
        data_files = DATASETS[args.dataset_name]

        # Sanity check for file extensions
        check_file_extensions(data_files["train"])
        check_file_extensions(data_files["validation"])

        dataset_args = {"keep_linebreaks": not args.no_keep_linebreaks}
        raw_datasets = load_dataset("text", data_files=data_files, **dataset_args)
    else:
        repo_info = DATASETS[args.dataset_name]
        raw_datasets = load_dataset(repo_info[0], repo_info[1])


    # LOAD PRETRAINED MODEL & TOKENIZER
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)  
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=True)
    else:
        raise ValueError("You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, save it, and load it from here, using --tokenizer_name.")

    if args.model_name_or_path:
        logger.info("Loading pretrained weights")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, trust_remote_code=True)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    print('Tokenizer len:', len(tokenizer))
    print('Pad token id:', tokenizer.pad_token_id)
    print('Model:', model)

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

    def preprocess_function_original(examples):
        # Concatenate all texts (aka sequence packing).
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()}
        result["labels"] = result["input_ids"].copy()
        return result
    
    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            preprocess_function_original,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping text.",
        )

    # dataset & dataloader creation
    train_dataset = lm_datasets["train"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_batch_size)
    print(f"Dataloader length: {len(train_dataloader)}")
    
    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(model)

    logger.info("***** Starting sampling *****")
    logger.info(f"Instantaneous batch size per device = {args.per_device_batch_size}")

    model.eval()

    # prepare dict for DPO dataset
    preference_dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            tokenized_input = batch['input_ids']
            prompt = tokenized_input[:, :tokenized_input.shape[1]//2]
            output_tok = model.generate(
                inputs=prompt.cuda(), 
                num_return_sequences=args.num_return_sequences, 
                do_sample=True, 
                max_length=tokenizer.model_max_length, 
                return_dict_in_generate=False, 
                output_scores=False
                )
            output = tokenizer.batch_decode(output_tok[:, (tokenized_input.shape[1]//2):], skip_special_tokens=True)
            original = tokenizer.decode(tokenized_input[0, (tokenized_input.shape[1]//2):], skip_special_tokens=True)

            for i in range(args.num_return_sequences):
                preference_dataset_dict["prompt"].append(prompt)
                preference_dataset_dict["chosen"].append(original)
                preference_dataset_dict["rejected"].append(output)

            if step % 100 == 0:
                print("step:", step, "of", len(train_dataloader))    

    # save results
    preference_dataset = Dataset.from_dict(preference_dataset_dict)
    preference_dataset.to_json(f"{args.save_prefix}.json")

if __name__ == "__main__":
    main()
