import numpy as np
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
lens = np.zeros(len(ds), dtype=int)
print('Number of data instances:', len(ds))

for i in range(len(ds)):
    lens[i] = len(ds[i]['text'].split())
    if i % 100000 == 0:
        print(f"Article {i}; Length {lens[i]} words.")

print(f"Min words in dataset: {np.min(lens)}")
print(f"Max words in dataset: {np.max(lens)}")
print(f"Mean words in dataset: {np.mean(lens)}")
print(f"Total words in dataset: {np.sum(lens)}")

def select_items_to_target(y, x, subset, target_length=10000000):
    
    # shuffle indices, pick the first n articles whose word count sum exceeds target length
    indices = np.random.permutation(len(x))
    x_shuffled_cumsum = np.cumsum(x[indices])
    n = np.searchsorted(x_shuffled_cumsum, target_length, side="left")

    train_indices = indices[:n]
    val_indices = indices[n:(n+1000)]

    # select subset with given indices
    train_dataset = y.select(train_indices)
    val_dataset = y.select(val_indices)

    # push to hub
    train_dataset.push_to_hub("eminorhan/random_wikipedia", subset, split="train", token=True)
    val_dataset.push_to_hub("eminorhan/random_wikipedia", subset, split="validation", token=True)

    print(f"Total number of articles selected: {n-1}")
    print(f"Total number of words: {x_shuffled_cumsum[n-1]}")

    return 

ds_10M = select_items_to_target(ds, lens, "10M", target_length=1e7)
ds_100M = select_items_to_target(ds, lens, "100M", target_length=1e8)