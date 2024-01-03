import random

import numpy as np

import torch
from torch.utils.data import Dataset


class DummyData(Dataset):
    def __init__(
        self,
        max_val: int,
        sample_count: int,
        sample_length: int,
        sparsity_percentage: int,
    ):
        r"""
        A data class that generates random data.
        Args:
            max_val (int): the maximum value for an element
            sample_count (int): count of training samples
            sample_length (int): number of elements in a sample
            sparsity_percentage (int): the percentage of
                embeddings used by the input data in each iteration
        """
        self.max_val = max_val
        self.input_samples = sample_count
        self.input_dim = sample_length
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
        self.target = torch.randint(0, max_val, [sample_count])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index]
