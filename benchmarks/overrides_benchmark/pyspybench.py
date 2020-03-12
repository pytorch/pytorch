import sys
import torch
from common import SubTensor, WithTorchFunction  # noqa: F401
Tensor = torch.Tensor

NUM_REPEATS = 1000000

if __name__ == "__main__":
    TensorClass = globals()[sys.argv[1]]

    t1 = TensorClass(1)
    t2 = TensorClass(2)

    for _ in range(NUM_REPEATS):
        torch.add(t1, t2)
