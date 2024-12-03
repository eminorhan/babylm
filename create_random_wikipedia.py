import numpy as np
from datasets import load_dataset


def strip_after_phrase(text):
    # iterate over each phrase and split the text
    for phrase in ["References", "See also", "External links", "Further reading", "Citations", "Notes"]:
        if phrase in text["text"]:
            text["text"] = text["text"].split(phrase)[0].strip()
    return text


def load_preprocess_wikipedia():
    
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    ds = ds.map(strip_after_phrase)
    lens = np.zeros(len(ds), dtype=int)
    print('Number of data instances:', len(ds))

    for i in range(len(ds)):
        lens[i] = len(ds[i]['text'].split())
        if i % 100000 == 0:
            print(f"Article {i}; Length {lens[i]} words.")

    print(f"Min words in dataset: {np.min(lens)}")
    print(f"Max words in dataset: {np.max(lens)}")
    print(f"Mean words in dataset: {np.mean(lens)}")
    print(f"Median words in dataset: {np.median(lens)}")
    print(f"Total words in dataset: {np.sum(lens)}")

    # push all processed data to hub under "all"
    ds.push_to_hub("eminorhan/wikipedia", "all", split="train", token=True)

    return ds, lens


def random_subset_wikipedia(ds, lens, subset, target_length=1e7):
    
    # shuffle indices, pick the first n articles whose word count sum exceeds target length
    indices = np.random.permutation(len(ds))
    lens_shuffled_cumsum = np.cumsum(lens[indices])
    n = np.searchsorted(lens_shuffled_cumsum, target_length, side="left")

    train_indices = indices[:n]
    val_indices = indices[n:(n+1000)]  # 1000 validation examples

    # select subset with given indices
    train_dataset = ds.select(train_indices)
    val_dataset = ds.select(val_indices)

    # push subset to hub
    train_dataset.push_to_hub("eminorhan/wikipedia", subset, split="train", token=True)
    val_dataset.push_to_hub("eminorhan/wikipedia", subset, split="validation", token=True)

    print(f"Total number of articles selected: {n-1}")
    print(f"Total number of words: {lens_shuffled_cumsum[n-1]}")


if __name__ == '__main__':

    # full dataset
    ds, lens = load_preprocess_wikipedia()
    
    # 10M subsets
    random_subset_wikipedia(ds, lens, "10M_1", target_length=1e7)
    random_subset_wikipedia(ds, lens, "10M_2", target_length=1e7)
    random_subset_wikipedia(ds, lens, "10M_3", target_length=1e7)

    # 100M subsets
    random_subset_wikipedia(ds, lens, "100M_1", target_length=1e8)
    random_subset_wikipedia(ds, lens, "100M_2", target_length=1e8)
    random_subset_wikipedia(ds, lens, "100M_3", target_length=1e8)        