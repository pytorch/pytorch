import unittest
import torch
from torch._foreach import foreach_map

class TestForeachMapMatmul(unittest.TestCase):
    def test_foreach_map_with_torch_mm(self):
        a_list = [torch.randn(3, 4) for _ in range(3)]
        b_list = [torch.randn(4, 2) for _ in range(3)]

        result = foreach_map(torch.mm, a_list, b_list)
        expected = [torch.mm(a, b) for a, b in zip(a_list, b_list)]

        for r, e in zip(result, expected):
            self.assertTrue(torch.allclose(r, e), msg=f"Expected {e}, got {r}")
