# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch._inductor.test_case import run_tests, TestCase


class TestEmbedding(TestCase):
    @config.patch({"assert_indirect_indexing": True})
    def test_embedding_negative_index_errors(self):
        def fn(weight, indices):
            return torch.embedding(weight, indices)

        weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        indices = torch.tensor([-1, 0])
        opt_fn = torch.compile(fn, backend="inductor")

        with self.assertRaisesRegex(IndexError, "index out of range"):
            fn(weight, indices)
        with self.assertRaisesRegex(RuntimeError, "index out of bounds"):
            opt_fn(weight, indices)

    def test_embedding_dense_backward_negative_index_no_wrap(self):
        def fn(grad_output, indices):
            return torch.ops.aten.embedding_dense_backward.default(
                grad_output, indices, 8, -1, False
            )

        grad_output = torch.ones(8, 1)
        indices = torch.arange(8) - 4
        opt_fn = torch.compile(fn, backend="inductor")

        expected = fn(grad_output, indices)
        actual = opt_fn(grad_output, indices)

        self.assertEqual(expected, actual)
        self.assertEqual(actual, torch.cat((torch.ones(4, 1), torch.zeros(4, 1))))

    def test_embedding_block_shape_assert_accepts_list_shape(self):
        self.assertEqual(
            get_broadcasted_shape(["XBLOCK", "1"], ("1", "RBLOCK")),
            ("XBLOCK", "RBLOCK"),
        )


if __name__ == "__main__":
    run_tests()
