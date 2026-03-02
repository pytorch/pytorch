# test/fake_tensor/test_fake_embedding.py

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.testing._internal.common_utils import TestCase, run_tests


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 3)

    def forward(self, x):
        return self.emb(x)


class FakeEmbeddingTest(TestCase):

    def _make_meta_indices(self):
        return torch.tensor([1, 2, 3], device="meta", dtype=torch.long)

    def test_fake_mode_embedding_meta_indices(self):
        m = EmbeddingModel()
        x_meta = self._make_meta_indices()

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        with fake_mode:
            out = m(x_meta)

        # before fix: crashed with Unhandled FakeTensor Device Propagation
        # after fix: this should run and produce a FakeTensor
        self.assertIsInstance(out, FakeTensor)

        self.assertEqual(out.fake_device.type, "cpu")

        self.assertEqual(tuple(out.shape), (3, 3))
        self.assertEqual(out.dtype, torch.float32)

    def test_torch_compile_eager_backend_meta_indices(self):
        m = EmbeddingModel()
        x_meta = self._make_meta_indices()

        m_opt = torch.compile(m, backend="eager")

        # before fix: this crashed inside FakeTensorMode during tracing
        out = m_opt(x_meta)

        # For backend="eager" runtime returns real tensor not FakeTensor
        self.assertIsInstance(out, torch.Tensor)
        self.assertNotIsInstance(out, FakeTensor)

        self.assertEqual(tuple(out.shape), (3, 3))
        self.assertEqual(out.dtype, torch.float32)

    def test_fake_embedding_is_shape_dtype_only(self):
        m = EmbeddingModel()
        x_meta = self._make_meta_indices()

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_mode.propagate_real_tensors = False 

        with fake_mode:
            out = m(x_meta)

        # should pass using only meta kernel
        self.assertIsInstance(out, FakeTensor)
        self.assertEqual(tuple(out.shape), (3, 3))
        self.assertEqual(out.dtype, torch.float32)

        self.assertEqual(out.fake_device.type, "cpu")

    def test_embedding_multi_dim_meta_indices(self):
        m = EmbeddingModel()
        x_meta = torch.tensor([[1, 2], [3, 4]], device="meta", dtype=torch.long)

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        with fake_mode:
            out = m(x_meta)

        self.assertIsInstance(out, FakeTensor)
        self.assertEqual(tuple(out.shape), (2, 2, 3))  # (batch, seq, emb_dim)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.fake_device.type, "cpu")

    def test_embedding_non_contiguous_meta_indices(self):
        m = EmbeddingModel()
        base = torch.tensor([1, 2, 3, 4], device="meta", dtype=torch.long)
        x_meta = base[::2]  # non-contiguous

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        with fake_mode:
            out = m(x_meta)

        self.assertIsInstance(out, FakeTensor)
        self.assertEqual(tuple(out.shape), (2, 3))  # 2 indices, emb_dim=3
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.fake_device.type, "cpu")

    def test_embedding_with_padding_idx(self):
        m = nn.Embedding(10, 3, padding_idx=0)
        x_meta = torch.tensor([0, 1, 2], device="meta", dtype=torch.long)

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        with fake_mode:
            out = m(x_meta)

        self.assertIsInstance(out, FakeTensor)
        self.assertEqual(tuple(out.shape), (3, 3))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.fake_device.type, "cpu")

    def test_two_embeddings_in_one_model(self):
        class TwoEmbeddings(nn.Module):
            def __init__(self):
                super().__init__()
                self.e1 = nn.Embedding(10, 3)
                self.e2 = nn.Embedding(20, 4)

            def forward(self, x, y):
                return self.e1(x), self.e2(y)

        m = TwoEmbeddings()
        x_meta = torch.tensor([1, 2, 3], device="meta", dtype=torch.long)
        y_meta = torch.tensor([4, 5], device="meta", dtype=torch.long)

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        with fake_mode:
            out1, out2 = m(x_meta, y_meta)

        # Both should be FakeTensors
        self.assertIsInstance(out1, FakeTensor)
        self.assertIsInstance(out2, FakeTensor)

        # Shapes
        self.assertEqual(tuple(out1.shape), (3, 3)) 
        self.assertEqual(tuple(out2.shape), (2, 4)) 

        # dtype and device checks
        self.assertEqual(out1.dtype, torch.float32)
        self.assertEqual(out2.dtype, torch.float32)
        self.assertEqual(out1.fake_device.type, "cpu")
        self.assertEqual(out2.fake_device.type, "cpu")



if __name__ == "__main__":
    run_tests()
