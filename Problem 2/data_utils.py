import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

#Converts characters to indices

class Vocab:
    def __init__(self, names: List[str]):
        chars = sorted(set("".join(names) + "^$"))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_idx = self.stoi["^"]   # reuse start token as pad
        self.start_idx = self.stoi["^"]  # ^ : start token of sequence
        self.end_idx = self.stoi["$"]   # $ : end token of sequence


    def encode(self, text: str) -> List[int]:  #Convert string to list of indices
        return [self.stoi[c] for c in text]

    def decode(self, ids: List[int]) -> str:   #list of indices to string
        return "".join(self.itos[i] for i in ids)

    def __len__(self) -> int: #return vocab. size
        return len(self.stoi)


#Creates training samples for next character prediction
class PrefixDataset(Dataset):
    """
    Example:For name "Ram"
    sequence becomes: ^Ram$

    Training pairs:
      input: ^      → target: R
      input: ^R     → target: a
      input: ^Ra    → target: m
      input: ^Ram   → target: $
    """
    def __init__(self, names: List[str], vocab: Vocab):
        self.examples: List[Tuple[List[int], int]] = []
        for name in names:             # Generate all prefix target pairs
            seq = "^" + name + "$"
            ids = vocab.encode(seq)
            for t in range(len(ids) - 1):
                self.examples.append((ids[: t + 1], ids[t + 1]))

    def __len__(self):     #Return total number of training examples
        return len(self.examples)

    def __getitem__(self, idx):     #Return one training examples
        x, y = self.examples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_names(path: str) -> List[str]:  #Removes empty lines and trailing spaces
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def collate_batch(batch, pad_idx: int):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long) # Length of each sequence
    max_len = max(len(x) for x in xs)                              # Find maximum length in batch
    xpad = torch.full((len(xs), max_len), pad_idx, dtype=torch.long) # Create padded tensor [batch_size, max_len]
    for i, x in enumerate(xs):
        xpad[i, :len(x)] = x
    y = torch.stack(ys)           # Stack targets
    return xpad, lengths, y

# Splits dataset into:Train set,Validation set, Test set

def make_loaders(dataset, batch_size: int, pad_idx: int, val_split: float = 0.1, test_split: float = 0.1, seed: int = 42):
    total = len(dataset)
    n_test = int(test_split * total)
    n_val = int(val_split * total)
    n_train = total - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    collate = lambda batch: collate_batch(batch, pad_idx)

    #Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader
