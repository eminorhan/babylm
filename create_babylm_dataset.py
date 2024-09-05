import random
import json
import numpy as np
from huggingface_hub import HfApi
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en")
n = len(ds['train'])
lens = np.zeros(n, dtype=int)
print('Number of data instances:', n)

for i in range(n):
    lens[i] = len(ds['train'][i]['text'].split())
    if i % 100000 == 0:
        print(f"Article {i}; Length {lens[i]} words.")

print(f"Min words in dataset: {np.min(lens)}")
print(f"Max words in dataset: {np.max(lens)}")
print(f"Mean words in dataset: {np.mean(lens)}")
print(f"Total words in dataset: {np.sum(lens)}")

def select_items_to_target(y, x, target_length=10000000):
    # Initialize list to hold selected items and their lengths
    train_articles = []
    val_articles = []
    current_length_sum = 0
    
    # Create a list of indices to sample from
    indices = list(range(len(y)))
    
    while current_length_sum < target_length:
        # Randomly pick an index
        index = random.choice(indices)
        
        # Check if adding this item will exceed the target length
        if current_length_sum + x[index] > target_length:
            break
        
        # Add the item and update the sum
        train_articles.append(y[index])
        current_length_sum += x[index]
        
        # Remove the index to avoid picking the same item again
        indices.remove(index)

    # val data
    for _ in range(1000):
        index = random.choice(indices)
        val_articles.append(y[index])

    print(f"Total number of articles selected: {len(train_articles)}")
    print(f"Total number of words: {current_length_sum}")

    return {'train': train_articles, 'validation': val_articles}

ds_10M = select_items_to_target(ds['train'], lens, target_length=10000000)
ds_100M = select_items_to_target(ds['train'], lens, target_length=100000000)

# Write the dictionaries to a JSON file
with open('random_wikipedia_10M.json', 'w') as file:
    json.dump(ds_10M, file, indent=4)  # indent=4 makes the JSON file more readable

with open('random_wikipedia_100M.json', 'w') as file:
    json.dump(ds_100M, file, indent=4)

api = HfApi()

# upload data files
api.upload_file(
    path_or_fileobj="random_wikipedia_10M.json",
    path_in_repo="10M/random_wikipedia_10M.json",
    repo_id="eminorhan/random_wikipedia",
    repo_type="dataset",
    token=True
)

api.upload_file(
    path_or_fileobj="random_wikipedia_100M.json",
    path_in_repo="100M/random_wikipedia_100M.json",
    repo_id="eminorhan/random_wikipedia",
    repo_type="dataset",
    token=True
)