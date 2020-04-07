import math

import torch
import pytest


N = 2 ** 14


@pytest.mark.parametrize('scale', tuple(range(-18, 11)))
def test_rs(scale):

    base = math.pow(2, scale)
    original_value = (base + math.pow(2, scale + 1)) / 2.0 + .5 * base
    x = torch.tensor([original_value] * N).cuda()
    _, exponent = math.frexp(original_value)
    exponent -= 1
    rounded = torch.stochastic_rounding(x)

    mean = torch.mean(rounded).item()
    delta_fp16 = math.pow(2, -10 + exponent if exponent >= -14 else -24)
    threshold = 1e-6
    diff = math.fabs(original_value - mean)

    # The right condition of `diff < delta_fp16 / 2.0` is for larger `original_value`.
    # The larger `original_value` is, the larger `delta_fp16` is.  So, no matter how many elements
    # we prepare, it's difficult to guarantee that `mean` is close enough the original value.
    assert diff < threshold or diff < delta_fp16 / 2.0
