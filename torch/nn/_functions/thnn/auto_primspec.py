import torch.toffee


def threshold_primspec(input, threshold=0, value=0, inplace=False):
    if inplace:
        return None
    if threshold == 0 and value == 0:
        return torch.toffee.op("Relu", input)

primspec_fns = {
    'Threshold': threshold_primspec,
}
