"""Neural network functional operator implementations."""

import math
import random

import torch
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


def is_float_dtype(dtype: torch.dtype) -> bool:
    """Check if dtype is a floating point type."""
    return dtype in [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
    ]


class EmbeddingOperator(Operator):
    """Operator for torch.nn.functional.embedding."""

    def __init__(self):
        super().__init__("torch.nn.functional.embedding")

    @property
    def torch_op_name(self) -> str | None:
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
        return is_float_dtype(output_spec.dtype)

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
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.linear"

    def can_produce(self, output_spec: Spec) -> bool:
        """Linear can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Linear needs at least 1 dimension (output features)
        if len(output_spec.size) == 0:
            return False
        return is_float_dtype(output_spec.dtype)

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
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.relu"

    def can_produce(self, output_spec: Spec) -> bool:
        """ReLU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

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
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.softmax"

    def can_produce(self, output_spec: Spec) -> bool:
        """Softmax can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Softmax needs at least 1 dimension to apply softmax along a dimension
        if len(output_spec.size) == 0:
            return False
        return is_float_dtype(output_spec.dtype)

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
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.dropout"

    def can_produce(self, output_spec: Spec) -> bool:
        """Dropout can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

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
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.layer_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        """LayerNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # LayerNorm needs at least 1 dimension to normalize over
        if len(output_spec.size) == 0:
            return False
        return is_float_dtype(output_spec.dtype)

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


class RMSNormOperator(Operator):
    """Operator for torch.nn.functional.rms_norm (Root Mean Square Normalization).

    RMSNorm is commonly used in modern LLMs like LLaMA. It normalizes by the RMS of the input.
    """

    def __init__(self):
        super().__init__("torch.nn.functional.rms_norm")
        self.weight = 5.0

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.rms_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        """RMSNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # RMSNorm needs at least 1 dimension to normalize over
        if len(output_spec.size) == 0:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for RMSNorm operation.

        RMSNorm requires:
        - input: input tensor
        - weight: (normalized_shape,) [optional]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("RMSNormOperator can only produce TensorSpec outputs")

        if len(output_spec.size) == 0:
            raise ValueError("RMSNorm output must have at least 1 dimension")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # Weight tensor (optional with 70% probability)
        normalized_shape = output_spec.size[-1:]
        specs = [input_spec]
        if random.random() < 0.7:
            weight_spec = TensorSpec(
                size=normalized_shape, stride=(1,), dtype=output_spec.dtype
            )
            specs.append(weight_spec)

        from typing import cast

        return cast(list[Spec], specs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for RMSNorm operation."""
        if len(input_names) < 1 or len(input_names) > 2:
            raise ValueError("RMSNorm requires 1-2 inputs: input, optional weight")

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("RMSNormOperator can only produce TensorSpec outputs")

        target_dtype = str(output_spec.dtype)
        input_name = input_names[0]

        # Normalize over the last dimension
        normalized_shape = f"({output_spec.size[-1]},)"

        if len(input_names) == 1:
            return f"{output_name} = torch.nn.functional.rms_norm({input_name}.to({target_dtype}), {normalized_shape})"
        else:  # len(input_names) == 2
            weight_name = input_names[1]
            return f"{output_name} = torch.nn.functional.rms_norm({input_name}.to({target_dtype}), {normalized_shape}, weight={weight_name}.to({target_dtype}))"


class GELUOperator(Operator):
    """Operator for torch.nn.functional.gelu (Gaussian Error Linear Unit)."""

    def __init__(self):
        super().__init__("torch.nn.functional.gelu")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.gelu"

    def can_produce(self, output_spec: Spec) -> bool:
        """GELU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for GELU operation.

        GELU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GELUOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for GELU operation."""
        if len(input_names) != 1:
            raise ValueError("GELU requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.nn.functional.gelu({input_name})"


class SigmoidOperator(Operator):
    """Operator for torch.sigmoid."""

    def __init__(self):
        super().__init__("torch.sigmoid")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.sigmoid"

    def can_produce(self, output_spec: Spec) -> bool:
        """Sigmoid can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for sigmoid operation.

        Sigmoid is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SigmoidOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for sigmoid operation."""
        if len(input_names) != 1:
            raise ValueError("Sigmoid requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.sigmoid({input_name})"


class TanhOperator(Operator):
    """Operator for torch.tanh."""

    def __init__(self):
        super().__init__("torch.tanh")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.tanh"

    def can_produce(self, output_spec: Spec) -> bool:
        """Tanh can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for tanh operation.

        Tanh is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("TanhOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for tanh operation."""
        if len(input_names) != 1:
            raise ValueError("Tanh requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.tanh({input_name})"


class BatchNormOperator(Operator):
    """Operator for torch.nn.functional.batch_norm."""

    def __init__(self):
        super().__init__("torch.nn.functional.batch_norm")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.batch_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        """BatchNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # BatchNorm needs at least 2 dimensions (batch, features)
        if len(output_spec.size) < 2:
            return False
        # Channel dimension (second dimension) must be greater than 0
        if output_spec.size[1] == 0:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for batch_norm operation.

        BatchNorm requires:
        - input: (N, C, ...) where N is batch and C is channels
        - running_mean: (C,) [optional]
        - running_var: (C,) [optional]
        - weight: (C,) [optional]
        - bias: (C,) [optional]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("BatchNormOperator can only produce TensorSpec outputs")

        if len(output_spec.size) < 2:
            raise ValueError("BatchNorm output must have at least 2 dimensions")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # Channel dimension is the second dimension
        num_features = output_spec.size[1]

        specs = [input_spec]

        # Add running_mean and running_var (required for inference mode)
        running_mean_spec = TensorSpec(
            size=(num_features,), stride=(1,), dtype=output_spec.dtype
        )
        running_var_spec = TensorSpec(
            size=(num_features,), stride=(1,), dtype=output_spec.dtype
        )
        specs.extend([running_mean_spec, running_var_spec])

        # Add weight and bias (optional with 70% probability)
        if random.random() < 0.7:
            weight_spec = TensorSpec(
                size=(num_features,), stride=(1,), dtype=output_spec.dtype
            )
            specs.append(weight_spec)

            if random.random() < 0.7:
                bias_spec = TensorSpec(
                    size=(num_features,), stride=(1,), dtype=output_spec.dtype
                )
                specs.append(bias_spec)

        from typing import cast

        return cast(list[Spec], specs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for batch_norm operation."""
        if len(input_names) < 3 or len(input_names) > 5:
            raise ValueError(
                "BatchNorm requires 3-5 inputs: input, running_mean, running_var, optional weight, optional bias"
            )

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("BatchNormOperator can only produce TensorSpec outputs")

        target_dtype = str(output_spec.dtype)
        input_name = input_names[0]
        running_mean_name = input_names[1]
        running_var_name = input_names[2]

        # Use training=False for deterministic behavior
        if len(input_names) == 3:
            return f"{output_name} = torch.nn.functional.batch_norm({input_name}.to({target_dtype}), {running_mean_name}.to({target_dtype}), {running_var_name}.to({target_dtype}), training=False)"
        elif len(input_names) == 4:
            weight_name = input_names[3]
            return f"{output_name} = torch.nn.functional.batch_norm({input_name}.to({target_dtype}), {running_mean_name}.to({target_dtype}), {running_var_name}.to({target_dtype}), weight={weight_name}.to({target_dtype}), training=False)"
        else:  # len(input_names) == 5
            weight_name = input_names[3]
            bias_name = input_names[4]
            return f"{output_name} = torch.nn.functional.batch_norm({input_name}.to({target_dtype}), {running_mean_name}.to({target_dtype}), {running_var_name}.to({target_dtype}), weight={weight_name}.to({target_dtype}), bias={bias_name}.to({target_dtype}), training=False)"


class GroupNormOperator(Operator):
    """Operator for torch.nn.functional.group_norm."""

    def __init__(self):
        super().__init__("torch.nn.functional.group_norm")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.group_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        """GroupNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # GroupNorm needs at least 2 dimensions (batch, channels)
        if len(output_spec.size) < 2:
            return False

        # GroupNorm requires more than 1 value per channel
        # For shape (N, C, *), num_values_per_channel = N * prod(*)
        # We need N * prod(*) > 1
        batch_size = output_spec.size[0]
        spatial_size = math.prod(output_spec.size[2:])
        num_values_per_channel = batch_size * spatial_size

        if num_values_per_channel <= 1:
            return False

        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for group_norm operation.

        GroupNorm requires:
        - input: (N, C, ...) where N is batch and C is channels
        - weight: (C,) [optional]
        - bias: (C,) [optional]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GroupNormOperator can only produce TensorSpec outputs")

        if len(output_spec.size) < 2:
            raise ValueError("GroupNorm output must have at least 2 dimensions")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # Channel dimension is the second dimension
        num_channels = output_spec.size[1]

        specs = [input_spec]

        # Add weight and bias (optional with 70% probability)
        if random.random() < 0.7:
            weight_spec = TensorSpec(
                size=(num_channels,), stride=(1,), dtype=output_spec.dtype
            )
            specs.append(weight_spec)

            if random.random() < 0.7:
                bias_spec = TensorSpec(
                    size=(num_channels,), stride=(1,), dtype=output_spec.dtype
                )
                specs.append(bias_spec)

        from typing import cast

        return cast(list[Spec], specs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for group_norm operation."""
        if len(input_names) < 1 or len(input_names) > 3:
            raise ValueError(
                "GroupNorm requires 1-3 inputs: input, optional weight, optional bias"
            )

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GroupNormOperator can only produce TensorSpec outputs")

        target_dtype = str(output_spec.dtype)
        input_name = input_names[0]

        # Determine number of groups (must divide num_channels evenly)
        num_channels = output_spec.size[1]
        # Common choices: 32, 16, 8, or equal to channels (instance norm)
        possible_groups = [g for g in [32, 16, 8, 4, 2, 1] if num_channels % g == 0]
        num_groups = possible_groups[0] if possible_groups else 1

        if len(input_names) == 1:
            return f"{output_name} = torch.nn.functional.group_norm({input_name}.to({target_dtype}), {num_groups})"
        elif len(input_names) == 2:
            weight_name = input_names[1]
            return f"{output_name} = torch.nn.functional.group_norm({input_name}.to({target_dtype}), {num_groups}, weight={weight_name}.to({target_dtype}))"
        else:  # len(input_names) == 3
            weight_name = input_names[1]
            bias_name = input_names[2]
            return f"{output_name} = torch.nn.functional.group_norm({input_name}.to({target_dtype}), {num_groups}, weight={weight_name}.to({target_dtype}), bias={bias_name}.to({target_dtype}))"


class LeakyReLUOperator(Operator):
    """Operator for torch.nn.functional.leaky_relu."""

    def __init__(self):
        super().__init__("torch.nn.functional.leaky_relu")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.leaky_relu"

    def can_produce(self, output_spec: Spec) -> bool:
        """LeakyReLU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for LeakyReLU operation.

        LeakyReLU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LeakyReLUOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for LeakyReLU operation."""
        if len(input_names) != 1:
            raise ValueError("LeakyReLU requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.nn.functional.leaky_relu({input_name}, negative_slope=0.01)"


class ELUOperator(Operator):
    """Operator for torch.nn.functional.elu (Exponential Linear Unit)."""

    def __init__(self):
        super().__init__("torch.nn.functional.elu")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.elu"

    def can_produce(self, output_spec: Spec) -> bool:
        """ELU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for ELU operation.

        ELU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ELUOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for ELU operation."""
        if len(input_names) != 1:
            raise ValueError("ELU requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.nn.functional.elu({input_name})"


class SiLUOperator(Operator):
    """Operator for torch.nn.functional.silu (Sigmoid Linear Unit, also known as Swish)."""

    def __init__(self):
        super().__init__("torch.nn.functional.silu")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.silu"

    def can_produce(self, output_spec: Spec) -> bool:
        """SiLU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for SiLU operation.

        SiLU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SiLUOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for SiLU operation."""
        if len(input_names) != 1:
            raise ValueError("SiLU requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.nn.functional.silu({input_name})"


class ScaledDotProductAttentionOperator(Operator):
    """Operator for torch.nn.functional.scaled_dot_product_attention."""

    def __init__(self):
        super().__init__("torch.nn.functional.scaled_dot_product_attention")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.scaled_dot_product_attention"

    def can_produce(self, output_spec: Spec) -> bool:
        """Scaled dot product attention can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # SDPA needs at least 3 dimensions (batch, seq_len, embed_dim)
        if len(output_spec.size) < 3:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for scaled_dot_product_attention.

        SDPA requires:
        - query: (batch, seq_len, embed_dim) or (batch, num_heads, seq_len, head_dim)
        - key: (batch, seq_len, embed_dim) or (batch, num_heads, seq_len_kv, head_dim)
        - value: (batch, seq_len, embed_dim) or (batch, num_heads, seq_len_kv, head_dim)
        Output shape matches query shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                "ScaledDotProductAttentionOperator can only produce TensorSpec outputs"
            )

        if len(output_spec.size) < 3:
            raise ValueError("SDPA output must have at least 3 dimensions")

        # Query has the same shape as output
        query_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # Key and value: match query shape for simplicity
        # In practice, seq_len for key/value can differ, but we'll keep it simple
        key_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )
        value_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [query_spec, key_spec, value_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for scaled_dot_product_attention operation."""
        if len(input_names) != 3:
            raise ValueError("SDPA requires exactly 3 inputs: query, key, value")

        # Ensure dtype compatibility by converting all inputs to the expected output dtype
        target_dtype = str(output_spec.dtype)
        query_name, key_name, value_name = input_names
        return f"{output_name} = torch.nn.functional.scaled_dot_product_attention({query_name}.to({target_dtype}), {key_name}.to({target_dtype}), {value_name}.to({target_dtype}))"


class MultiHeadAttentionForwardOperator(Operator):
    """Operator for torch.nn.functional.multi_head_attention_forward."""

    def __init__(self):
        super().__init__("torch.nn.functional.multi_head_attention_forward")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nn.functional.multi_head_attention_forward"

    def can_produce(self, output_spec: Spec) -> bool:
        """Multi-head attention forward can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # MHA needs at least 3 dimensions (seq_len, batch, embed_dim)
        if len(output_spec.size) < 3:
            return False
        # MHA cannot handle 0-sized dimensions (seq_len, batch, or embed_dim must be > 0)
        if any(dim == 0 for dim in output_spec.size):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for multi_head_attention_forward.

        MHA requires:
        - query, key, value: (seq_len, batch, embed_dim)
        - in_proj_weight: (3*embed_dim, embed_dim) for combined QKV projection
        - in_proj_bias: (3*embed_dim,) optional
        - out_proj_weight: (embed_dim, embed_dim)
        - out_proj_bias: (embed_dim,) optional

        For simplicity, we'll use the combined in_proj_weight path.

        IMPORTANT: The order of optional parameters matters for codegen!
        We must ensure that when we have 6 inputs, they are in the order:
        query, key, value, in_proj_weight, in_proj_bias, out_proj_weight
        NOT: query, key, value, in_proj_weight, out_proj_weight, out_proj_bias
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                "MultiHeadAttentionForwardOperator can only produce TensorSpec outputs"
            )

        if len(output_spec.size) < 3:
            raise ValueError("MHA output must have at least 3 dimensions")

        # Output shape: (seq_len, batch, embed_dim)
        seq_len, batch, embed_dim = output_spec.size[:3]

        # Query, key, value have the same shape as output
        query_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )
        key_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )
        value_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # in_proj_weight: (3*embed_dim, embed_dim)
        in_proj_weight_spec = TensorSpec(
            size=(3 * embed_dim, embed_dim),
            stride=(embed_dim, 1),
            dtype=output_spec.dtype,
        )

        # out_proj_weight: (embed_dim, embed_dim)
        out_proj_weight_spec = TensorSpec(
            size=(embed_dim, embed_dim),
            stride=(embed_dim, 1),
            dtype=output_spec.dtype,
        )

        # For simplicity and correctness, always generate all required tensors
        # This avoids ambiguity in the codegen about which optional parameters are present
        # We'll use a simplified signature: query, key, value, in_proj_weight, out_proj_weight only
        specs = [
            query_spec,
            key_spec,
            value_spec,
            in_proj_weight_spec,
            out_proj_weight_spec,
        ]

        from typing import cast

        return cast(list[Spec], specs)

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
        """Generate code for multi_head_attention_forward operation."""
        if len(input_names) != 5:
            raise ValueError(
                "MHA requires exactly 5 inputs: query, key, value, in_proj_weight, out_proj_weight"
            )

        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                "MultiHeadAttentionForwardOperator can only produce TensorSpec outputs"
            )

        target_dtype = str(output_spec.dtype)
        embed_dim = output_spec.size[-1]

        # Determine number of heads (must divide embed_dim evenly)
        # Common choices: 8, 4, 2, 1
        possible_heads = [h for h in [8, 4, 2, 1] if embed_dim % h == 0]
        num_heads = possible_heads[0] if possible_heads else 1

        query_name = input_names[0]
        key_name = input_names[1]
        value_name = input_names[2]
        in_proj_weight_name = input_names[3]
        out_proj_weight_name = input_names[4]

        # Build the function call without optional biases
        code = f"""{output_name}, _ = torch.nn.functional.multi_head_attention_forward(
    {query_name}.to({target_dtype}),
    {key_name}.to({target_dtype}),
    {value_name}.to({target_dtype}),
    {embed_dim},
    {num_heads},
    {in_proj_weight_name}.to({target_dtype}),
    None,  # in_proj_bias
    None,  # bias_k
    None,  # bias_v
    False,  # add_zero_attn
    0.0,  # dropout_p (no dropout for testing)
    {out_proj_weight_name}.to({target_dtype}),
    None,  # out_proj_bias
    training=False,  # Use eval mode for deterministic behavior
    need_weights=False,  # Don't compute attention weights for performance
)"""

        return code
