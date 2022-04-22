import torch
import argparse
from common import SubTensor, WithTorchFunction, SubWithTorchFunction  # noqa: F401

Tensor = torch.tensor

NUM_REPEATS = 1000000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the torch.add for a given class a given number of times."
    )
    parser.add_argument(
        "tensor_class", metavar="TensorClass", type=str, help="The class to benchmark."
    )
    parser.add_argument(
        "--nreps", "-n", type=int, default=NUM_REPEATS, help="The number of repeats."
    )
    args = parser.parse_args()

    TensorClass = globals()[args.tensor_class]
    NUM_REPEATS = args.nreps

    t1 = TensorClass([1.])
    t2 = TensorClass([2.])

    for _ in range(NUM_REPEATS):
        torch.add(t1, t2)
