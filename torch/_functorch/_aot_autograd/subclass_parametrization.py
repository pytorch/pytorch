from __future__ import annotations

import dataclasses
import itertools
from typing import Any, TYPE_CHECKING

import torch
from torch._library.opaque_object import is_opaque_reference_type
from torch._opaque_base import OpaqueBase
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .schemas import OpaqueMeta


if TYPE_CHECKING:
    from collections.abc import Iterable


# This is technically very similar to SubclassCreatingMeta
# in aot_autograd, but we don't need all the stuff in there
# so just recreated a new dataclass.
@dataclasses.dataclass
class SubclassCreationMeta:
    start_idx: int
    num_tensors: int
    class_type: Any
    # None means the attr is a plain tensor (base case of recursion)
    attrs: dict[str, SubclassCreationMeta | OpaqueMeta | None]
    metadata: Any
    outer_size: Iterable[None | int | torch.SymInt]
    outer_stride: Iterable[None | int | torch.SymInt]


class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors) -> torch.Tensor:  # type: ignore[no-untyped-def]
        todo: list[torch.Tensor | OpaqueBase] = list(tensors)

        def _unwrap_tensor_subclasses(subclass_meta, tensors, offset):  # type: ignore[no-untyped-def]
            if subclass_meta is None:
                return tensors[offset], offset + 1
            inner_tensors = {}
            for attr, meta in subclass_meta.attrs.items():
                if isinstance(meta, OpaqueMeta):
                    inner_tensors[attr] = tensors[offset]
                    offset += 1
                else:
                    built_tensor, offset = _unwrap_tensor_subclasses(
                        meta, tensors, offset
                    )
                    inner_tensors[attr] = built_tensor
            rebuilt = subclass_meta.class_type.__tensor_unflatten__(
                inner_tensors,
                subclass_meta.metadata,
                subclass_meta.outer_size,
                subclass_meta.outer_stride,
            )
            return rebuilt, offset

        return _unwrap_tensor_subclasses(self.subclass_meta, todo, 0)[0]

    def right_inverse(self, tensor: torch.Tensor) -> list[torch.Tensor | OpaqueBase]:
        if type(tensor) is torch.Tensor:
            raise AssertionError("tensor must be a subclass, not torch.Tensor")
        plain_tensors: list[torch.Tensor | OpaqueBase] = []

        def _create_subclass_meta(tensor, idx, plain_tensor_container):  # type: ignore[no-untyped-def]
            if type(tensor) is torch.Tensor:
                plain_tensor_container.append(tensor)
                return None, idx + 1
            inner_tensors_attrnames, metadata = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
            new_idx = idx
            attr_to_meta: dict[str, SubclassCreationMeta | OpaqueMeta | None] = {}
            for attr in inner_tensors_attrnames:
                val = getattr(tensor, attr)
                match val:
                    case OpaqueBase():
                        if not is_opaque_reference_type(type(val)):
                            raise ValueError(
                                f"{type(val).__name__!r} found in tensor attrs of "
                                f"{type(tensor).__name__}.__tensor_flatten__(). "
                                "Only tensors and reference-type opaques are allowed "
                                "in tensor attrs."
                            )
                        attr_to_meta[attr] = OpaqueMeta()
                        plain_tensor_container.append(val)
                        new_idx += 1
                    case torch.Tensor():
                        subclass_meta, new_idx = _create_subclass_meta(
                            val, new_idx, plain_tensor_container
                        )
                        attr_to_meta[attr] = subclass_meta
                    case _:
                        raise AssertionError(
                            f"expected Tensor or OpaqueBase, got {type(val)}"
                        )
            return (
                SubclassCreationMeta(
                    start_idx=idx,
                    num_tensors=new_idx - idx,
                    class_type=type(tensor),
                    attrs=attr_to_meta,
                    metadata=metadata,
                    outer_size=tensor.size(),
                    outer_stride=tensor.stride(),
                ),
                new_idx,
            )

        self.subclass_meta = _create_subclass_meta(tensor, 0, plain_tensors)[0]
        return plain_tensors


def unwrap_tensor_subclass_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Model transformation that replaces all the parameters that are subclasses to plain tensors.
    This reduces runtime overhead of flattening/unflattening the parameters.

    This transformation adds parametrization with `torch.nn.utils.parametrize`.
    The FQNs of the subclass parameters will be changed and state_dict will become incompatible with the original model.
    E.g.
    Original model state_dict: {"p1": torch.testing._internal.TwoTensor}
    becomes: {"parametrizations.p2.original0": torch.Tensor, "parametrizations.p2.original1": torch.Tensor}

    """
    for name, tensor in itertools.chain(
        list(module.named_parameters(recurse=False)),
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        list(module.named_buffers(recurse=False)),
    ):
        if is_traceable_wrapper_subclass(tensor):
            torch.nn.utils.parametrize.register_parametrization(
                module, name, UnwrapTensorSubclass()
            )

    for child in module.children():
        unwrap_tensor_subclass_parameters(child)

    return module
