# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase, TemporaryFileName

class TestZeroDimTensorTrace(TestCase):
    def test_bugrepro(self):
        
        import torch
        def f(x):
            return x[x > 0]
        jf = torch.jit.trace(f, torch.tensor(2., device="cpu"))

        

if __name__ == '__main__':
    run_tests()
