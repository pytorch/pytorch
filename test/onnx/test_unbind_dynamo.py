"""Test for unbind operation ONNX export with dynamo (Issue #168969)"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestUnbindDynamo(TestCase):
    """Test unbind ONNX export with dynamo=True"""

    def test_unbind_simple(self):
        """Test simple unbind operation along different dimensions"""

        class SimpleUnbind(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                # Unbind along specified dimension
                tensors = torch.unbind(x, dim=self.dim)
                # Sum all the unbound tensors to produce a single output
                return sum(tensors)

        # Test unbind along dimension 0
        model = SimpleUnbind(dim=0)
        x = torch.randn(3, 4, 5)

        # Test forward pass
        expected_output = model(x)
        self.assertEqual(expected_output.shape, (4, 5))

        # Test ONNX export with dynamo
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            torch.onnx.export(
                model,
                (x,),
                onnx_path,
                dynamo=True,
                opset_version=18,
            )

            # Verify the ONNX model is valid
            import onnx

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_unbind_dim1(self):
        """Test unbind along dimension 1"""

        class UnbindDim1(nn.Module):
            def forward(self, x):
                tensors = torch.unbind(x, dim=1)
                return sum(tensors)

        model = UnbindDim1()
        x = torch.randn(2, 3, 4)

        expected_output = model(x)
        self.assertEqual(expected_output.shape, (2, 4))

        # Test ONNX export
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            torch.onnx.export(
                model,
                (x,),
                onnx_path,
                dynamo=True,
                opset_version=18,
            )
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_unbind_in_lstm(self):
        """Test unbind in LSTM model (similar to issue #168969)"""

        class SimpleDecoder(nn.Module):
            def __init__(self, vocab_size=100, embedding_dim=64, hidden_size=64):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_size, 1, batch_first=True)
                self.fc = nn.Linear(hidden_size, vocab_size)

            def forward(self, tokens, h, c):
                embedded = self.embedding(tokens).unsqueeze(0)
                output, (h_out, c_out) = self.lstm(embedded, (h, c))
                logits = self.fc(output.squeeze(0).squeeze(0))
                return logits, h_out, c_out

        model = SimpleDecoder()
        model.eval()

        tokens = torch.tensor([1])
        h = torch.randn(1, 1, 64)  # (num_layers, batch_size, hidden_size)
        c = torch.randn(1, 1, 64)  # (num_layers, batch_size, hidden_size)

        # Test forward pass
        with torch.no_grad():
            logits, h_out, c_out = model(tokens, h, c)
            self.assertEqual(logits.shape, (100,))
            self.assertEqual(h_out.shape, (1, 1, 64))
            self.assertEqual(c_out.shape, (1, 1, 64))

        # Test ONNX export with dynamo=True
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            torch.onnx.export(
                model,
                (tokens, h, c),
                onnx_path,
                dynamo=True,
                opset_version=18,
                input_names=["tokens", "h", "c"],
                output_names=["logits", "h_out", "c_out"],
            )

            # Verify the ONNX model
            import onnx

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_unbind_negative_dim(self):
        """Test unbind with negative dimension"""

        class UnbindNegDim(nn.Module):
            def forward(self, x):
                # Unbind with negative dimension (-1 means last dimension)
                tensors = torch.unbind(x, dim=-1)
                return sum(tensors)

        model = UnbindNegDim()
        x = torch.randn(2, 3, 4)

        # Test forward pass
        expected_output = model(x)
        self.assertEqual(expected_output.shape, (2, 3))

        # Test ONNX export
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            torch.onnx.export(
                model,
                (x,),
                onnx_path,
                dynamo=True,
                opset_version=18,
            )

            # Verify the ONNX model
            import onnx

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_unbind_negative_dim2(self):
        """Test unbind with dim=-2"""

        class UnbindNegDim2(nn.Module):
            def forward(self, x):
                # Unbind with dim=-2 (second to last dimension)
                tensors = torch.unbind(x, dim=-2)
                return sum(tensors)

        model = UnbindNegDim2()
        x = torch.randn(2, 3, 4)

        # Test forward pass
        expected_output = model(x)
        self.assertEqual(expected_output.shape, (2, 4))

        # Test ONNX export
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            torch.onnx.export(
                model,
                (x,),
                onnx_path,
                dynamo=True,
                opset_version=18,
            )
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_unbind_size_one_dim(self):
        """Test unbind with dimension of size 1"""

        class UnbindSizeOne(nn.Module):
            def forward(self, x):
                # Unbind along dimension with size 1
                tensors = torch.unbind(x, dim=0)
                # Return the single tensor (size 1 tuple)
                return tensors[0]

        model = UnbindSizeOne()
        x = torch.randn(1, 4, 5)

        # Test forward pass
        expected_output = model(x)
        self.assertEqual(expected_output.shape, (4, 5))

        # Test ONNX export
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            torch.onnx.export(
                model,
                (x,),
                onnx_path,
                dynamo=True,
                opset_version=18,
            )

            # Verify the ONNX model
            import onnx

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_unbind_zero_size_dim(self):
        """Test unbind with dimension of size 0"""

        class UnbindZeroSize(nn.Module):
            def forward(self, x):
                # Unbind along dimension with size 0
                tensors = torch.unbind(x, dim=0)
                # If empty, return a zero tensor as placeholder
                if len(tensors) == 0:
                    return torch.zeros(3, 4)
                return sum(tensors)

        model = UnbindZeroSize()
        x = torch.randn(0, 3, 4)  # Dimension 0 has size 0

        # Test forward pass
        expected_output = model(x)
        self.assertEqual(expected_output.shape, (3, 4))

        # Test ONNX export
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            torch.onnx.export(
                model,
                (x,),
                onnx_path,
                dynamo=True,
                opset_version=18,
            )

            # Verify the ONNX model
            import onnx

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


if __name__ == "__main__":
    run_tests()
