from typing import List, NamedTuple, Iterable, Tuple, Sequence

import torch

from .types import Shape, ShapeRange
from .utils import get_dynamic_dims


class InputTensorSpec(NamedTuple):
    """
    This class contains the information of a input tensor.

    shape: shape of the tensor.

    dtype: dtyep of the tensor.

    device: device of the tensor. This is only used to generate inputs to the given model
        in order to run shape prop. For TensorRT engine, inputs have to be on cuda device.

    shape_ranges: If dynamic shape is needed (shape has dimensions of -1), then this field
        has to be provided (default is empty list). Every shape_range is a tuple of three
        tuples ((min_input_shape), (optimized_input_shape), (max_input_shape)). Each shape_range
        is used to populate a TensorRT optimization profile.
        e.g. If the input shape varies from (1, 224) to (100, 224) and we want to optimize
        for (25, 224) because it's the most common input shape, then we set shape_ranges to
        ((1, 224), (25, 225), (100, 224)).

    has_batch_dim: Whether the shape includes batch dimension. Batch dimension has to be provided
        if the engine want to run with dynamic shape.
    """

    shape: Shape
    dtype: torch.dtype
    device: torch.device = torch.device("cpu")
    shape_ranges: List[ShapeRange] = []
    has_batch_dim: bool = True

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "InputTensorSpec":
        """
        Produce an InputTenosrSpec named tuple which contains the
        information of the given PyTorch tensor.

        Args:
            tensor (torch.Tensor): A PyTorch tensor.

        Returns:
            An InputTensorSpec named tuple.
        """
        return cls(tensor.shape, tensor.dtype, tensor.device)

    @classmethod
    def from_tensors(cls, tensors: Iterable[torch.Tensor]) -> List["InputTensorSpec"]:
        """
        Produce a list of InputTenosrSpec named tuples which contain
        the information of all the given PyTorch tensors.

        Args:
            tensors (Iterable[torch.Tensor]): A list of PyTorch tensors.

        Returns:
            A list of InputTensorSpec named tuples.
        """
        return [cls.from_tensor(t) for t in tensors]

    @classmethod
    def from_tensors_with_dynamic_batch_size(
        cls,
        tensors: Sequence[torch.Tensor],
        batch_size_range: Tuple[int, int, int]
    ) -> List["InputTensorSpec"]:
        """
        Produce a list of InputTenosrSpec named tuples which would contain
        the information of all the given PyTorch tensors. The produced input
        tensor specs will treat all tensors' first dimension as batch dimension
        and mark them as dynmaic.

        Args:
            tensors (Sequence[torch.Tensor]): A list of PyTorch tensors.
            batch_size_range (Tuple[int, int, int]): The first integer indicates
                the smallest batch size allowed. The second integer indiceates
                the batch size that we'll optimize for. The third integer indicates
                the largest batch size allowed.

        Returns:
            A list of InputTensorSpec named tuples with dynamic ranges.
        """
        input_specs = []
        batch_size = tensors[0].size(0)

        for i, tensor in enumerate(tensors):
            assert (
                batch_size == tensor.size(0)
            ), f"The {i}th tensor (shape: {tensor.shape}) doesn't have the correct batch size: {batch_size}."
            shape = list(tensor.shape)
            shape[0] = -1
            shape_ranges: List[ShapeRange] = [tuple(tuple([bs] + shape[1:]) for bs in batch_size_range)]  # type: ignore[list-item]
            input_specs.append(cls(tuple(shape), tensor.dtype, tensor.device, shape_ranges))

        return input_specs

    def to_random_tensor(self):
        shape = tuple(self.shape)
        if len(get_dynamic_dims(shape)):
            shape = tuple(self.shape_ranges[0][1])
        elif not self.has_batch_dim:
            shape = (1,) + tuple(shape)

        return torch.randn(shape).to(dtype=self.dtype, device=self.device)

    @staticmethod
    def create_inputs_from_specs(input_specs: Iterable["InputTensorSpec"]):
        inputs = []

        for spec in input_specs:
            inputs.append(spec.to_random_tensor())

        return inputs
