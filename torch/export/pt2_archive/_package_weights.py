import collections
import warnings

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._ordered_set import OrderedSet


def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + tensor.element_size()
    else:
        stop = tensor.data_ptr()
    return stop


class TensorProperties:
    def __init__(self, tensor: torch.Tensor):
        self.is_fake = isinstance(tensor, FakeTensor)
        self.is_contiguous = tensor.is_contiguous()
        self.storage_ptr = None
        self.storage_size = None
        self.start = None
        self.end = None

        if not self.is_fake:
            # only get the storage pointer for real tensors
            # pyrefly: ignore [bad-assignment]
            self.storage_ptr = tensor.untyped_storage().data_ptr()
            if self.is_contiguous:
                # only get storage size and start/end pointers for contiguous tensors
                # pyrefly: ignore [bad-assignment]
                self.storage_size = tensor.untyped_storage().nbytes()
                # pyrefly: ignore [bad-assignment]
                self.start = tensor.data_ptr()
                # pyrefly: ignore [bad-assignment]
                self.end = _end_ptr(tensor)

        # info to recover tensor
        self.shape = tensor.shape
        self.stride = tensor.stride()
        self.offset = tensor.storage_offset()

    def is_complete(self) -> bool:
        """
        Whether the tensor completely overlaps with its underlying storage
        """
        if self.is_fake:
            # Theoretically, fake tensors should not appear in weights
            # But we handle this corner case to make it always complete
            return True
        if not self.is_contiguous:
            return False

        if self.storage_ptr is None:
            raise AssertionError("storage_ptr cannot be None for complete check")
        if self.storage_size is None:
            raise AssertionError("storage_size cannot be None for complete check")
        if self.start is None:
            raise AssertionError("start cannot be None for complete check")
        if self.end is None:
            raise AssertionError("end cannot be None for complete check")
        return (
            self.start == self.storage_ptr
            and self.end == self.storage_ptr + self.storage_size
        )


class Weights(dict):
    """
    A dictionary mapping from weight name to a tuple of (tensor, TensorProperties).
    tensor represents the actual initial value of the weight.
    TensorProperties represents the properties of the weight that are needed to recover the weight.

    We use two separate entries because `tensor` could be a clone of the original weight tensor,
    so it doesn't have the same property as the original weight (such as underlying storage pointer).
    """

    def __init__(self, weight_dict: dict[str, tuple[torch.Tensor, TensorProperties]]):
        super().__init__(weight_dict)

    def get_weight(self, name: str) -> tuple[torch.Tensor, TensorProperties]:
        return self[name]

    def get_weight_properties(self, name: str) -> TensorProperties:
        return self[name][1]


def get_complete(
    group: OrderedSet[tuple[str, str]], models_weights: dict[str, Weights]
) -> tuple[str, str]:
    """
    `group` is a (model_name, weight_name) tuple.
    `model_weights` is a dictionary mapping from model name to its Weights.

    One of the tensor in `group` must be complete and they must share the
    same underlying storage.

    Returns the name of the complete tensor in the `group`. If multiple
    tensors are complete, returns an arbitrary one.
    """

    def get_tensor_properties(name_tuple: tuple[str, str]) -> TensorProperties:
        # returns the tensor properties
        (model_name, weight_name) = name_tuple
        return models_weights[model_name].get_weight_properties(weight_name)

    for name_tuple in group:
        tensor_property = get_tensor_properties(name_tuple)
        if tensor_property.is_complete():
            return name_tuple

    warnings.warn(
        "No complete tensor found in the group! Returning the first one. "
        "This may cause issues when your weights are not on CPU.",
        stacklevel=2,
    )
    if len(group) == 0:
        raise AssertionError("group cannot be empty")
    return next(iter(group))


def group_weights(all_weights: dict[str, Weights]) -> list[OrderedSet[tuple[str, str]]]:
    """
    Group weights that share the same underlying storage.

    Returns a list of sets, each set contains a tuple of (model_name, weight_name).
    """

    weights_dict: dict[tuple[int, torch.dtype], OrderedSet[tuple[str, str]]] = (
        collections.defaultdict(OrderedSet)
    )  # (storage_key, dtype) -> set(weight)

    for model_name, weights in all_weights.items():
        for weight_name, (tensor, properties) in weights.items():
            weights_dict[(properties.storage_ptr, tensor.dtype)].add(
                (model_name, weight_name)
            )

    return list(weights_dict.values())
