# Owner(s): ["oncall: jit"]

import torch
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase

torch._lazy.ts_backend.init()


class LazyGeneratorTest(TestCase):
    def test_generator(self):
        """
        Test that generators are being inserted into the TorchScript
        graph by setting different seeds before each call to
        generate_tensor but the resulting tensor is the same
        """
        def generate_tensor():
            t = torch.tensor(1.0)
            g = torch.Generator()
            g.manual_seed(2023)
            t.uniform_()
            return t

        torch.manual_seed(1)

        with torch.device("cpu"):
            cpu_t = generate_tensor()

        torch.manual_seed(2)

        with torch.device("lazy"):
            lazy_t = generate_tensor()

        torch._lazy.mark_step()

        assert torch.allclose(cpu_t, lazy_t.to("cpu"))


