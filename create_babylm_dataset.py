import random
import numpy as np
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en")
n = len(ds['train'])
lens = np.zeros(n, dtype=int)
print('Number of data instances:', n)

for i in range(n):
    lens[i] = len(ds['train'][i]['text'].split())
    if i % 10000 == 0:
        print(f"Article {i}; Length {lens[i]} words.")

print(f"Min words in dataset: {np.min(lens)}")
print(f"Max words in dataset: {np.max(lens)}")
print(f"Mean words in dataset: {np.mean(lens)}")
print(f"Total words in dataset: {np.sum(lens)}")

def select_items_to_target(y, x, target_length=10000000):
    # Initialize list to hold selected items and their lengths
    selected_items = []
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
        selected_items.append(y[index])
        current_length_sum += x[index]
        
        # Remove the index to avoid picking the same item again
        indices.remove(index)
        
    print(f"Total number of articles selected: {len(selected_items)}")

    return {'train': selected_items}

ds_10M = select_items_to_target(ds['train'], lens, target_length=10000000)
ds_100M = select_items_to_target(ds['train'], lens, target_length=100000000)

ds_10M.push_to_hub("eminorhan/random_wikipedia", "10M")
ds_100M.push_to_hub("eminorhan/random_wikipedia", "100M")