"""Neural network functional operator implementations."""

import random
from typing import Optional

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class EmbeddingOperator(Operator):
    """Operator for torch.nn.functional.embedding."""

    def __init__(self):
        super().__init__("torch.nn.functional.embedding")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nn.functional.embedding"

    def can_produce(self, output_spec: Spec) -> bool:
        """Embedding can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Embedding needs at least 1 dimension (embedding_dim)
        if len(output_spec.size) == 0:
            return False
        # Embedding outputs are typically float tensors
        return output_spec.dtype in [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for embedding operation.

        Embedding requires:
        - weight tensor: (num_embeddings, embedding_dim)
        - input tensor: integer indices (any shape, but output shape + [embedding_dim])
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("EmbeddingOperator can only produce TensorSpec outputs")

        # Output shape should be input_shape + [embedding_dim]
        if len(output_spec.size) == 0:
            raise ValueError("Embedding output must have at least 1 dimension")

        embedding_dim = output_spec.size[-1]
        input_shape = output_spec.size[:-1]  # Remove last dimension for embedding_dim

        # Generate reasonable vocab size that's larger than our index generation range
        # This ensures that indices generated in range [0, 100) will always be valid
        num_embeddings = random.randint(150, 500)  # Always larger than max index (100)

        # Weight tensor: (num_embeddings, embedding_dim)
        weight_spec = TensorSpec(
            size=(num_embeddings, embedding_dim),
            stride=(embedding_dim, 1),
            dtype=output_spec.dtype,
        )

        # Input tensor: integer indices with shape that produces the output shape
        input_spec = TensorSpec(
            size=input_shape,
            stride=self._calculate_stride(input_shape),
            dtype=torch.int64,  # Indices are typically int64
        )

        return [weight_spec, input_spec]

    def _calculate_stride(self, size):
        """Calculate stride for a given size."""
        if not size:
            return ()
        stride = []
        current_stride = 1
        for dim_size in reversed(size):
            stride.append(current_stride)
            current_stride *= dim_size
        return tuple(reversed(stride))

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for embedding operation."""
        if len(input_names) != 2:
            raise ValueError("Embedding requires exactly 2 inputs: weight and input")

        weight_name, input_name = input_names
        # Ensure indices are integer type and clamped to valid range
        # This handles any arithmetic operations that might produce out-of-bounds indices
        return f"{output_name} = torch.nn.functional.embedding(torch.clamp({input_name}.to(torch.int64), 0, {weight_name}.size(0)-1), {weight_name})"


class LinearOperator(Operator):
    """Operator for torch.nn.functional.linear."""

    def __init__(self):
        super().__init__("torch.nn.functional.linear")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nn.functional.linear"

    def can_produce(self, output_spec: Spec) -> bool:
        """Linear can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Linear needs at least 1 dimension (output features)
        if len(output_spec.size) == 0:
            return False
        return output_spec.dtype in [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for linear operation.

        Linear transformation: y = xW^T + b
        - input: (..., in_features)
        - weight: (out_features, in_features)
        - bias: (out_features,) [optional]
        - output: (..., out_features)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LinearOperator can only produce TensorSpec outputs")

        if len(output_spec.size) == 0:
            raise ValueError("Linear output must have at least 1 dimension")

        out_features = output_spec.size[-1]
        batch_shape = output_spec.size[:-1]

        # Generate reasonable input features size
        in_features = random.randint(8, 256)

        # Input tensor: (..., in_features)
        input_shape = batch_shape + (in_features,)
        input_spec = TensorSpec(
            size=input_shape,
            stride=self._calculate_stride(input_shape),
            dtype=output_spec.dtype,
        )

        # Weight tensor: (out_features, in_features)
        weight_spec = TensorSpec(
            size=(out_features, in_features),
            stride=(in_features, 1),
            dtype=output_spec.dtype,
        )

        # Bias tensor: (out_features,) - make bias optional with 50% probability
        if random.random() < 0.5:
            bias_spec = TensorSpec(
                size=(out_features,), stride=(1,), dtype=output_spec.dtype
            )
            return [input_spec, weight_spec, bias_spec]
        else:
            return [input_spec, weight_spec]

    def _calculate_stride(self, size):
        """Calculate stride for a given size."""
        if not size:
            return ()
        stride = []
        current_stride = 1
        for dim_size in reversed(size):
            stride.append(current_stride)
            current_stride *= dim_size
        return tuple(reversed(stride))

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for linear operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LinearOperator can only produce TensorSpec outputs")

        # Ensure dtype compatibility by converting all inputs to the expected output dtype
        target_dtype = str(output_spec.dtype)

        if len(input_names) == 2:
            input_name, weight_name = input_names
            return f"{output_name} = torch.nn.functional.linear({input_name}.to({target_dtype}), {weight_name}.to({target_dtype}))"
        elif len(input_names) == 3:
            input_name, weight_name, bias_name = input_names
            return f"{output_name} = torch.nn.functional.linear({input_name}.to({target_dtype}), {weight_name}.to({target_dtype}), {bias_name}.to({target_dtype}))"
        else:
            raise ValueError(
                "Linear requires 2 or 3 inputs: input, weight, and optional bias"
            )


class ReLUOperator(Operator):
    """Operator for torch.nn.functional.relu."""

    def __init__(self):
        super().__init__("torch.nn.functional.relu")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nn.functional.relu"

    def can_produce(self, output_spec: Spec) -> bool:
        """ReLU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype in [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for ReLU operation.

        ReLU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ReLUOperator can only produce TensorSpec outputs")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for ReLU operation."""
        if len(input_names) != 1:
            raise ValueError("ReLU requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.nn.functional.relu({input_name})"


class SoftmaxOperator(Operator):
    """Operator for torch.nn.functional.softmax."""

    def __init__(self):
        super().__init__("torch.nn.functional.softmax")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nn.functional.softmax"

    def can_produce(self, output_spec: Spec) -> bool:
        """Softmax can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Softmax needs at least 1 dimension to apply softmax along a dimension
        if len(output_spec.size) == 0:
            return False
        return output_spec.dtype in [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for softmax operation.

        Softmax is element-wise along a dimension, input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SoftmaxOperator can only produce TensorSpec outputs")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for softmax operation."""
        if len(input_names) != 1:
            raise ValueError("Softmax requires exactly 1 input")

        input_name = input_names[0]
        # Use dim=-1 as default (last dimension)
        return f"{output_name} = torch.nn.functional.softmax({input_name}, dim=-1)"


class DropoutOperator(Operator):
    """Operator for torch.nn.functional.dropout."""

    def __init__(self):
        super().__init__("torch.nn.functional.dropout")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nn.functional.dropout"

    def can_produce(self, output_spec: Spec) -> bool:
        """Dropout can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype in [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for dropout operation.

        Dropout is element-wise, input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("DropoutOperator can only produce TensorSpec outputs")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for dropout operation."""
        if len(input_names) != 1:
            raise ValueError("Dropout requires exactly 1 input")

        input_name = input_names[0]
        # Use training=False to make it deterministic for testing
        return f"{output_name} = torch.nn.functional.dropout({input_name}, p=0.1, training=False)"


class LayerNormOperator(Operator):
    """Operator for torch.nn.functional.layer_norm."""

    def __init__(self):
        super().__init__("torch.nn.functional.layer_norm")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nn.functional.layer_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        """LayerNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # LayerNorm needs at least 1 dimension to normalize over
        if len(output_spec.size) == 0:
            return False
        return output_spec.dtype in [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for layer_norm operation.

        LayerNorm normalizes over the last dimensions specified by normalized_shape.
        - input: input tensor
        - weight: (normalized_shape,) [optional]
        - bias: (normalized_shape,) [optional]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LayerNormOperator can only produce TensorSpec outputs")

        if len(output_spec.size) == 0:
            raise ValueError("LayerNorm output must have at least 1 dimension")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # For simplicity, normalize over the last dimension
        normalized_shape = output_spec.size[-1:]

        # Weight and bias tensors (optional with 70% probability each)
        specs = [input_spec]
        if random.random() < 0.7:
            # LayerNorm weight and bias parameters should match input tensor dtype
            # for compatibility (conversion will be handled in codegen)
            weight_spec = TensorSpec(
                size=normalized_shape, stride=(1,), dtype=output_spec.dtype
            )
            specs.append(weight_spec)

            if random.random() < 0.7:
                bias_spec = TensorSpec(
                    size=normalized_shape, stride=(1,), dtype=output_spec.dtype
                )
                specs.append(bias_spec)

        # Cast to list[Spec] to fix type checking
        from typing import cast

        return cast(list[Spec], specs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for layer_norm operation."""
        if len(input_names) < 1 or len(input_names) > 3:
            raise ValueError(
                "LayerNorm requires 1-3 inputs: input, optional weight, optional bias"
            )

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LayerNormOperator can only produce TensorSpec outputs")

        # Normalize over the last dimension
        normalized_shape = f"({output_spec.size[-1]},)"

        # Ensure dtype compatibility by converting all inputs to the expected output dtype
        target_dtype = str(output_spec.dtype)

        input_name = input_names[0]

        if len(input_names) == 1:
            return f"{output_name} = torch.nn.functional.layer_norm({input_name}.to({target_dtype}), {normalized_shape})"
        elif len(input_names) == 2:
            weight_name = input_names[1]
            return f"{output_name} = torch.nn.functional.layer_norm({input_name}.to({target_dtype}), {normalized_shape}, weight={weight_name}.to({target_dtype}))"
        else:  # len(input_names) == 3
            weight_name, bias_name = input_names[1], input_names[2]
            return f"{output_name} = torch.nn.functional.layer_norm({input_name}.to({target_dtype}), {normalized_shape}, weight={weight_name}.to({target_dtype}), bias={bias_name}.to({target_dtype}))"
