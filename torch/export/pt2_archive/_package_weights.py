import collections

import torch
from torch.utils._ordered_set import OrderedSet


def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + tensor.element_size()
    else:
        stop = tensor.data_ptr()
    return stop


class TensorProperties:
    def __init__(self, tensor: torch.Tensor):
        # info about underlying storage
        self.storage_ptr = tensor.untyped_storage().data_ptr()
        self.storage_size = tensor.untyped_storage().nbytes()

        # info to recover tensor
        self.shape = tensor.shape
        self.stride = tensor.stride()
        self.offset = tensor.storage_offset()

        self.start = tensor.data_ptr()
        self.end = _end_ptr(tensor)

    def is_complete(self) -> bool:
        """
        Whehter the tensor completely overlaps with its underlying storage
        """
        return (
            self.start == self.storage_ptr
            and self.end == self.storage_ptr + self.storage_size
        )


class Weights(dict):
    """
    A dictionary mapping from weight name to a tuple of (tensor, TensorProperties).
    tensor represents the actual intial value of the weight.
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

    raise RuntimeError("No complete tensor found in the group!")


def group_weights(all_weights: dict[str, Weights]) -> list[OrderedSet[tuple[str, str]]]:
    """
    Group weights that share the same underlying storage.

    Returns a list of sets, each set contains a tuple of (model_name, weight_name).
    """

    weights_dict: dict[int, OrderedSet[tuple[str, str]]] = collections.defaultdict(
        OrderedSet
    )  # storage_key -> set(weight)

    for model_name, weights in all_weights.items():
        for weight_name, (_, properties) in weights.items():
            weights_dict[properties.storage_ptr].add((model_name, weight_name))

    return list(weights_dict.values())
