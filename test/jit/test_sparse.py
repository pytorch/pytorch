# Owner(s): ["oncall: jit"]

import io
import torch
from torch.testing._internal.jit_utils import JitTestCase


class TestSparse(JitTestCase):
    def test_freeze_sparse_coo(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()

            def forward(self, x):
                return x + self.a

        x = torch.rand(3, 4).to_sparse()

        m = SparseTensorModule()
        unfrozen_result = m.forward(x)

        m.eval()
        frozen = torch.jit.freeze(torch.jit.script(m))

        frozen_result = frozen.forward(x)

        self.assertEqual(unfrozen_result, frozen_result)

    def test_serialize_sparse_coo(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()

            def forward(self, x):
                return x + self.a

        torch.jit.save(torch.jit.script(SparseTensorModule()), io.BytesIO())
