import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

class MlDataLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    