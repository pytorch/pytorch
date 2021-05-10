import random

import numpy as np
import torch
from torch.utils.data import Dataset


class DummyData(Dataset):

    def __init__(
        self,
        max_val: int,
        input_samples: int,
        input_dim: int,
        sparsity_percentage: int
    ):
        self.max_val = max_val
        self.input_samples = input_samples
        self.input_dim = input_dim
        self.sparsity_percentage = sparsity_percentage

        def generate_input():
            precentage_of_elements = (100 - self.sparsity_percentage) / float(100)
            index_count = int(self.max_val * precentage_of_elements)
            elements = list(range(self.max_val))
            random.shuffle(elements)
            elements = elements[:index_count]
            data = [
                [
                    elements[random.randint(0, index_count - 1)]
                    for _ in range(self.input_dim)
                ]
                for _ in range(self.input_samples)
            ]
            return torch.from_numpy(np.array(data))

        self.input = generate_input()
        self.target = torch.randint(0, max_val, [input_samples])
        self.start = 0
        self.end = max_val

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index]
