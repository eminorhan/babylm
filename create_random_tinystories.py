import numpy as np
from datasets import load_dataset, Dataset


def load_preprocess_tinystories():

    dataset = load_dataset("roneneldan/TinyStories", data_files={"train": "TinyStoriesV2-GPT4-train.txt", "validation": "TinyStoriesV2-GPT4-valid.txt"}, sample_by="paragraph")

    split_txt_train, split_txt_validation = [], [] 

    for i in range(len(dataset["train"])):
        text = dataset["train"][i]["text"].split("<|endoftext|>")
        split_txt_train.extend([t.strip() for t in text if t])

    for i in range(len(dataset["validation"])):
        text = dataset["validation"][i]["text"].split("<|endoftext|>")
        split_txt_validation.extend([t.strip() for t in text if t])

    def gen_train():
        for text in split_txt_train:
            yield {"text": text}

    def gen_validation():
        for text in split_txt_validation:
            yield {"text": text}

    ds_train = Dataset.from_generator(gen_train)
    ds_validation = Dataset.from_generator(gen_validation)
        
    lens = np.zeros(len(ds_train), dtype=int)
    print('Number of data instances:', len(ds_train))

    for i in range(len(ds_train)):
        lens[i] = len(ds_train[i]['text'].split())
        if i % 100000 == 0:
            print(f"Story {i}; Length {lens[i]} words.")

    print(f"Min words in dataset: {np.min(lens)}")
    print(f"Max words in dataset: {np.max(lens)}")
    print(f"Mean words in dataset: {np.mean(lens)}")
    print(f"Median words in dataset: {np.median(lens)}")
    print(f"Total words in dataset: {np.sum(lens)}")

    # push all data to hub under "all"
    ds_train.push_to_hub("eminorhan/tinystories", "all", split="train", token=True)
    ds_validation.push_to_hub("eminorhan/tinystories", "all", split="validation", token=True)

    return ds_train, ds_validation, lens


def random_subset_tinystories(ds, ds_val, lens, subset, target_length=1e7):
    
    # shuffle indices, pick the first n articles whose word count sum exceeds target length
    indices = np.random.permutation(len(ds))
    lens_shuffled_cumsum = np.cumsum(lens[indices])
    n = np.searchsorted(lens_shuffled_cumsum, target_length, side="left")

    train_indices = indices[:n]

    # select subset with given indices
    ds_train = ds.select(train_indices)

    # push subset to hub
    ds_train.push_to_hub("eminorhan/tinystories", subset, split="train", token=True)
    ds_val.push_to_hub("eminorhan/tinystories", subset, split="validation", token=True)

    print(f"Total number of articles selected: {n-1}")
    print(f"Total number of words: {lens_shuffled_cumsum[n-1]}")


ds_train, ds_validation, lens = load_preprocess_tinystories()
random_subset_tinystories(ds_train, ds_validation, lens, "10M", target_length=1e7)
random_subset_tinystories(ds_train, ds_validation, lens, "100M", target_length=1e8)