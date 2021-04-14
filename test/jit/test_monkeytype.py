
import os
import sys
import torch

def test_sum_monkey(a, b):
    return a + b

def test_sub_monkey(a, b):
    return a - b

def test_mul_monkey(a, b):
    return a * b

def test_args_complex(real, img):
    return torch.complex(real, img)
