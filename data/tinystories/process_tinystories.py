import jsonlines

split = 'valid'
num_keep = 'all'

# Specify the input file path
input_file_path = f"TinyStoriesV2-GPT4-{split}.txt"
output_file_path = f"TinyStoriesV2-GPT4-{split}-{num_keep}.jsonl"

# Read the text data from the input file
with open(input_file_path, 'r') as file:
    text_data = file.read()

# Split the text data into chunks based on the specified delimiter
chunks = text_data.split('<|endoftext|>')

# Remove empty chunks if any
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(f"Total number of stories in {split} split: {len(chunks)}")

# Create a list to store dictionaries (each story as a dictionary)
stories = []
story_id = 0
for chunk in chunks:
    stories.append({"story": chunk, "story_id": story_id})
    story_id += 1
    if (num_keep != 'all') and (story_id > num_keep - 1):
         break

print(f"Total number of stories kept: {len(stories)}")

# write to jsonl file
with jsonlines.open(output_file_path, mode='w') as writer:
       writer.write_all(stories)

print(f"Data has been split and saved to {output_file_path}")
