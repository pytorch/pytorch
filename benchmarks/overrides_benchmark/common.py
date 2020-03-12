import torch

NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000


class SubTensor(torch.Tensor):
    pass


class WithTorchFunction(torch.Tensor):
    def __torch_function__(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        return args[0] + args[1]
