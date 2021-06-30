# flake8: noqa
import torch

torch.set_rng_state([1, 2, 3])  # E: Argument 1 to "set_rng_state" has incompatible type "List[int]"; expected "Tensor"
