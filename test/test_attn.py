# coding=utf-8

import torch

from torch.testing._internal.common_utils import TestCase, run_tests

def attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    attn = q.matmul(k.t()).tanh()
    return attn.matmul(v), attn

class TestAttn(TestCase):
    def test_comparison_to_python_implementation(self):
        q = torch.rand(2, 3)
        k = torch.rand(2, 3)
        v = torch.rand(2, 4)

        cpp = torch.attn(q, k, v)
        py_output, py_attn = attn(q, k, v)

        assert torch.allclose(cpp.output, py_output)
        assert torch.allclose(cpp.attn, py_attn)

    def test_q_is_not_matrix(self):
        q = torch.rand(3)
        k = torch.rand(2, 3)
        v = torch.rand(2, 4)
        with self.assertRaisesRegex(RuntimeError, 'q is not a matrix'):
            torch.attn(q, k, v)

    def test_k_is_not_matrix(self):
        q = torch.rand(2, 3)
        k = torch.rand(3)
        v = torch.rand(2, 4)
        with self.assertRaisesRegex(RuntimeError, 'k is not a matrix'):
            torch.attn(q, k, v)

    def test_v_is_not_matrix(self):
        q = torch.rand(2, 3)
        k = torch.rand(2, 3)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, 'v is not a matrix'):
            torch.attn(q, k, v)

    def test_rows_mismatch(self):
        q = torch.rand(2, 3)
        k = torch.rand(2, 3)
        v = torch.rand(4, 4)  # uh-oh, |v rows| does not match q and k
        with self.assertRaisesRegex(RuntimeError, 'q, k, and v must all have the same number of rows'):
            torch.attn(q, k, v)

    def test_q_and_k_columns_mismatch(self):
        q = torch.rand(2, 3)
        k = torch.rand(2, 4)  # uh-oh, |k cols| ≠ |q cols|
        v = torch.rand(2, 4)
        with self.assertRaisesRegex(RuntimeError, 'q and k must have the same number of columns'):
            torch.attn(q, k, v)


if __name__ == '__main__':
    run_tests()
