import os
import torch
from torch.utils.data import Dataset
from random import randrange
from pathlib import Path

class BabylmDataset(Dataset):
    """
    Example usage:
    tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    train_dataset = BabylmDataset(PATH / "data/babylm_10M", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
    full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev", SEQ_LENGTH, tokenizer=tokenizer, offset=0)
    eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
    eval_dataset = Subset(full_eval_dataset, eval_indices)
    """

    def __init__(self, data_dir: str, seq_length: int, tokenizer, offset: int=0, random_chunk: bool=False):
        self.seq_length = seq_length
        self.offset = offset
        self.tokenizer = tokenizer
        self.random_chunk = random_chunk

        tokenizer_name = tokenizer.__class__.__name__
        tokenized_file = Path(os.path.join(data_dir, f"tokenized_{tokenizer_name}_{tokenizer.vocab_size}.pt"))

        if tokenized_file.exists():
            print(f"Loading data from {tokenized_file}")
            self.data = torch.load(tokenized_file)
        else:
            data = []
            src_files = [str(f) for f in Path(data_dir).glob("**/*")
                         if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train", ".dev"]]

            for src_file in src_files:
                text = Path(src_file).read_text(encoding="utf-8")
                encoded = self.tokenizer.encode(text)
                print("🔥", src_file, "len:", len(encoded))
                data.extend(encoded)

            self.data = torch.tensor(data)

            # Save tokenized data
            print(f"Saving data to {tokenized_file}")
            torch.save(self.data, tokenized_file)

    def __len__(self):
        if self.random_chunk:
            return len(self.data) // self.seq_length - 1
        else:
            return (len(self.data) - self.offset) // self.seq_length

    def __getitem__(self, i):
        if self.random_chunk:
            offset = randrange(self.seq_length) # Sample random offset between 0 and seq_length-1
            return self.data[i*self.seq_length+offset:(i+1)*self.seq_length+offset]
        else:
            return self.data[i*self.seq_length+self.offset:(i+1)*self.seq_length+self.offset]
