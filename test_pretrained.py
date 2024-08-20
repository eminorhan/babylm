import argparse
import logging
import os
from itertools import chain

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers
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
from datetime import timedelta
from accelerate import InitProcessGroupKwargs

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
    parser = argparse.ArgumentParser(description="Evaluate recognition memory in large language models")
    parser.add_argument('--data_files', nargs='+', help="list of files containing the training data")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None, help="The training dataset will be truncated to blocks of this size (after tokenization) for training.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files.")
    parser.add_argument("--use_pretrained_weights", action="store_true", help="Whether to use pretrained weights.")

    args = parser.parse_args()

    # Sanity check for file extensions
    check_file_extensions(args.data_files)

    return args


def main():
    args = parse_args()
    print(args)

    # Initialize the accelerator
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # In distributed training, 'load_dataset' function guarantee that only one local process can concurrently download the dataset.
    data_files = {"train": args.data_files}
    dataset_args = {}
    extension = args.data_files[0].split(".")[-1]
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, token=True)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, token=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, model_max_length=1024, token=True)  # TODO: pass this more beautifully
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, model_max_length=1024, token=True)  # TODO: pass this more beautifully
        if args.model_name_or_path.startswith("meta-llama") or args.model_name_or_path.startswith("gpt2") or args.model_name_or_path.startswith("EleutherAI"):
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, save it, and load it from here, using --tokenizer_name.")

    if args.model_name_or_path and args.use_pretrained_weights:
        logger.info("Loading pretrained weights")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, token=True)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(new_num_tokens=len(tokenizer), pad_to_multiple_of=128)
    logger.info(f"Tokenizer len: {len(tokenizer)}")
    logger.info(f"Pad token id: {tokenizer.pad_token_id}")

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

    logger.info(f"Block size: {block_size}")

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

    dataset = lm_datasets["train"]
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    model, dataloader = accelerator.prepare(model, dataloader)

    logger.info("***** Running evaluation *****")
    logger.info(f"Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"Dataset size = {len(dataset)}")
    logger.info(f"Dataloader size = {len(dataloader)}")

    model.eval()

    # pass examples
    losses = []
    batches = []
    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            
            # log at regular intervals
            if idx % 10 == 0:
                logger.info(f"batch idx: {idx} of {len(dataloader)}; batch shape: {batch['input_ids'].shape}; loss shape: {loss.shape}")

            losses.append(loss.cpu())
            batches.append(batch['input_ids'].cpu())

    losses, indices = torch.sort(torch.tensor(losses))
    batches = torch.cat(batches)[indices, :]
    losses, batches = losses.bfloat16(), batches.int()
    logger.info(f"Evaluated the examples. Losses shape (dtype) = {losses.shape} ({losses.dtype}); Batches shape (type) = {batches.shape} ({batches.dtype})")
    logger.info(f"Mean-std = {torch.mean(losses)}-{torch.std(losses)}. Max-min = {torch.max(losses)}-{torch.min(losses)}")


if __name__ == "__main__":
    main()