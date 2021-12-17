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
                self.b = torch.rand(3, 4).to_sparse()

            def forward(self, x):
                return x + self.a + self.b

        x = torch.rand(3, 4).to_sparse()

        m = SparseTensorModule()
        unfrozen_result = m.forward(x)

        m.eval()
        frozen = torch.jit.freeze(torch.jit.script(m))

        frozen_result = frozen.forward(x)

        self.assertEqual(unfrozen_result, frozen_result)

        buffer = io.BytesIO()
        torch.jit.save(frozen, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(unfrozen_result, loaded_result)

    def test_serialize_sparse_coo(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()
                self.b = torch.rand(3, 4).to_sparse()

            def forward(self, x):
                return x + self.a + self.b

        x = torch.rand(3, 4).to_sparse()
        m = SparseTensorModule()
        expected_result = m.forward(x)

        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(m), buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(expected_result, loaded_result)
