"""Test for embedding operator."""

from torchfuzz.tensor import Tensor
from .embedding import EmbeddingOperator


def test_embedding_can_produce():
    """Test that EmbeddingOperator can produce appropriate tensors."""
    op = EmbeddingOperator()

    # Should produce 1D, 2D, 3D, etc. tensors with floating point types
    assert op.can_produce(Tensor((128,), (1,), "float32", "cpu", set()))
    assert op.can_produce(Tensor((10, 128), (128, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((2, 10, 128), (1280, 128, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((4, 2, 10, 128), (2560, 1280, 128, 1), "float16", "cpu", set()))

    # Should not produce integer or boolean tensors (embedding outputs are float)
    assert not op.can_produce(Tensor((128,), (1,), "int64", "cpu", set()))
    assert not op.can_produce(Tensor((10, 128), (128, 1), "int32", "cpu", set()))
    assert not op.can_produce(Tensor((128,), (1,), "bool", "cpu", set()))


def test_embedding_decompose():
    """Test embedding decomposition."""
    op = EmbeddingOperator()

    # Test 1D output (embedding_dim,)
    output_tensor = Tensor((768,), (1,), "float32", "cpu", set())
    inputs = op.decompose(output_tensor)

    assert len(inputs) == 2
    # Input indices should be int64 and have shape () (scalar) or (1,)
    assert inputs[0].dtype == "int64"
    assert inputs[0].size == (1,)  # At least 1D for indices
    # Weight should have shape (num_embeddings, embedding_dim)
    assert inputs[1].size[1] == 768  # embedding_dim matches
    assert inputs[1].size[0] > 0  # num_embeddings > 0
    assert inputs[1].dtype == "float32"

    # Test 2D output (batch_size, embedding_dim)
    output_tensor = Tensor((32, 512), (512, 1), "float32", "cpu", set())
    inputs = op.decompose(output_tensor)

    assert len(inputs) == 2
    # Input indices: (32,) - same as batch dimensions
    assert inputs[0].size == (32,)
    assert inputs[0].dtype == "int64"
    # Weight: (num_embeddings, 512)
    assert inputs[1].size[1] == 512
    assert inputs[1].size[0] > 0
    assert inputs[1].dtype == "float32"

    # Test 3D output (batch_size, seq_len, embedding_dim)
    output_tensor = Tensor((16, 50, 256), (12800, 256, 1), "float16", "cpu", set())
    inputs = op.decompose(output_tensor)

    assert len(inputs) == 2
    # Input indices: (16, 50) - same as batch dimensions
    assert inputs[0].size == (16, 50)
    assert inputs[0].dtype == "int64"
    # Weight: (num_embeddings, 256)
    assert inputs[1].size[1] == 256
    assert inputs[1].size[0] > 0
    assert inputs[1].dtype == "float16"


def test_embedding_codegen():
    """Test embedding code generation."""
    op = EmbeddingOperator()

    code = op.codegen("output", ["indices", "weight"], None)
    assert code == "output = torch.nn.functional.embedding(torch.clamp(indices, 0, weight.size(0) - 1).to(torch.long), weight)"


def test_embedding_supports_variable_inputs():
    """Test that embedding does not support variable inputs."""
    op = EmbeddingOperator()
    assert not op.supports_variable_inputs()
