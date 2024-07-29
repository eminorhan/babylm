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
import json
import argparse
import logging
import os

import datasets
import torch

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils.versions import require_version

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained model")
    parser.add_argument("--test_file", type=str, default='', help="path to test stories")
    parser.add_argument("--save_prefix", type=str, default='', help="Informative string for saving purposes")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None, help="The training dataset will be truncated to blocks of this size (after tokenization) for training.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
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

    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(model)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    logger.info("***** Starting sampling *****")
    logger.info(f"Instantaneous batch size per device = {args.per_device_eval_batch_size}")

    # process test stories
    test_stories = [] 
    # Open the JSON file
    with open(args.test_file, 'r') as json_file:
        # Read each line separately and load JSON data
        for line in json_file:
            try:
                # Load JSON data from the line
                data = json.loads(line)

                # Assuming the JSON structure is a dictionary with a "story" field
                story = data.get("story", None)

                test_stories.append(story)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    model.eval()

    generations = []
    for i in range(100):
        with torch.no_grad():
            tokenized_input = tokenizer([test_stories[i]], return_tensors="pt")['input_ids']
            tokenized_input_trunc = tokenized_input[:, :tokenized_input.shape[1]//2]
            output_tok = model.generate(inputs=tokenized_input_trunc.cuda(), do_sample=True, max_length=tokenizer.model_max_length, return_dict_in_generate=False, output_scores=False)
            original = tokenizer.decode(tokenized_input[0], skip_special_tokens=True)
            prompt = tokenizer.decode(tokenized_input_trunc[0], skip_special_tokens=True)
            output = tokenizer.decode(output_tok[0], skip_special_tokens=True)
            print('origin:', original)
            print('prompt:', prompt)
            print('output:', output)
            print('\n')

            generations.append(output)

    # save results
    save_path = os.path.join(args.output_dir, args.save_prefix + '_results.npz')

if __name__ == "__main__":
    main()