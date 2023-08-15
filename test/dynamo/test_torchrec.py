import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter

try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    HAS_TORCHREC = True
except ImportError:
    HAS_TORCHREC = False


@unittest.skipIf(not HAS_TORCHREC, "these tests require torchrec")
class TorchRecTests(torch._dynamo.test_case.TestCase):
    def test_simple(self):
        jag_tensor1 = KeyedJaggedTensor(
            values=torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            keys=["index_0", "index_1"],
            lengths=torch.tensor([0, 0, 1, 1, 1, 3]),
        ).sync()

        # TODO: probably don't need this
        torch._dynamo.mark_dynamic(jag_tensor1.values(), 0)
        torch._dynamo.mark_dynamic(jag_tensor1.lengths(), 0)

        # ordinarily, this would trigger one specialization
        self.assertEqual(jag_tensor1.length_per_key(), [1, 5])

        counter = CompileCounter()

        @torch._dynamo.optimize(counter, nopython=True)
        def f(jag_tensor):
            return jag_tensor["index_0"].values().sum()

        f(jag_tensor1)

        self.assertEqual(counter.frame_count, 1)

        jag_tensor2 = KeyedJaggedTensor(
            values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            keys=["index_0", "index_1"],
            lengths=torch.tensor([2, 0, 1, 1, 1, 3]),
        ).sync()

        f(jag_tensor2)

        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
