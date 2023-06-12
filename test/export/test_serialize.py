# Owner(s): ["module: dynamo"]
import unittest

import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerialize(TestCase):
    pass

if __name__ == '__main__':
    run_tests()
