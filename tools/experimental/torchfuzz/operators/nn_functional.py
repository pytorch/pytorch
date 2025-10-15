"""Neural network functional operator implementations."""

import random
from typing import Optional

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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nn.functional.group_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        """GroupNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # GroupNorm needs at least 2 dimensions (batch, channels)
        if len(output_spec.size) < 2:
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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
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
    def torch_op_name(self) -> Optional[str]:
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
