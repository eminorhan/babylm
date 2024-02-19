from datasets import load_dataset

train_file = '/home/emin/Documents/babylm/babylm_10M/aochildes.txt'

data_files = {"train": train_file}
dataset_args = {}
extension = train_file.split(".")[-1]
if extension == "txt":
    extension = "text"
    dataset_args["keep_linebreaks"] = False
raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)