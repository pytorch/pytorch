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
        if not is_traceable_wrapper_subclass(tensor):
            raise AssertionError("tensor must be a traceable wrapper subclass")
        plain_tensors: list[torch.Tensor | OpaqueBase] = []

        def _create_subclass_meta(tensor, idx, plain_tensor_container):  # type: ignore[no-untyped-def]
            if not is_traceable_wrapper_subclass(tensor):
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


def _join_fqn(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _flattened_tensor_attr_paths(
    tensor: torch.Tensor | OpaqueBase,
) -> list[tuple[str, ...]]:
    flattened_paths: list[tuple[str, ...]] = []

    def _collect_attr_paths(
        tensor: torch.Tensor | OpaqueBase, attr_path: tuple[str, ...]
    ) -> None:
        if not is_traceable_wrapper_subclass(tensor):
            flattened_paths.append(attr_path)
            return

        inner_tensors_attrnames, _ = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
        for attr in inner_tensors_attrnames:
            _collect_attr_paths(getattr(tensor, attr), (*attr_path, attr))

    _collect_attr_paths(tensor, ())
    return flattened_paths


def _flattened_state_dict_name(original_fqn: str, attr_path: tuple[str, ...]) -> str:
    if not attr_path:
        return original_fqn
    attr_suffix = "_".join(attr.replace(".", "_") for attr in attr_path)

    parent, sep, leaf = original_fqn.rpartition(".")
    flattened_leaf = f"{leaf}_{attr_suffix}"
    return f"{parent}{sep}{flattened_leaf}"


def _make_unique_state_dict_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        return name

    idx = 1
    while (uniq := f"{name}_{idx}") in used_names:
        idx += 1
    return uniq


def _collect_tensor_subclass_state_fqns(
    module: torch.nn.Module,
) -> tuple[set[str], set[str]]:
    all_state_fqns: set[str] = set()
    subclass_state_fqns: set[str] = set()

    for name, tensor in itertools.chain(
        module.named_parameters(recurse=True, remove_duplicate=False),
        module.named_buffers(recurse=True, remove_duplicate=False),
    ):
        all_state_fqns.add(name)
        if is_traceable_wrapper_subclass(tensor):
            subclass_state_fqns.add(name)

    return all_state_fqns, subclass_state_fqns


def _unwrap_tensor_subclass_parameters(
    module: torch.nn.Module,
    fqn_map: dict[str, str] | None = None,
    used_state_fqns: set[str] | None = None,
    prefix: str = "",
) -> torch.nn.Module:
    if fqn_map is not None and used_state_fqns is None:
        raise AssertionError("used_state_fqns must be provided with fqn_map")

    for name, tensor in itertools.chain(
        list(module.named_parameters(recurse=False)),
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        list(module.named_buffers(recurse=False)),
    ):
        if is_traceable_wrapper_subclass(tensor):
            original_fqn = _join_fqn(prefix, name)
            flattened_paths: list[tuple[str, ...]]
            flattened_paths = (
                _flattened_tensor_attr_paths(tensor) if fqn_map is not None else []
            )

            torch.nn.utils.parametrize.register_parametrization(
                module, name, UnwrapTensorSubclass()
            )

            if fqn_map is not None:
                if used_state_fqns is None:
                    raise AssertionError("used_state_fqns must not be None")
                for i, attr_path in enumerate(flattened_paths):
                    internal_fqn = _join_fqn(
                        prefix, f"parametrizations.{name}.original{i}"
                    )
                    flattened_fqn = _flattened_state_dict_name(original_fqn, attr_path)
                    flattened_fqn = _make_unique_state_dict_name(
                        flattened_fqn, used_state_fqns
                    )
                    used_state_fqns.add(flattened_fqn)
                    fqn_map[internal_fqn] = flattened_fqn

    for child_name, child in module.named_children():
        if (
            child_name == "parametrizations"
            and torch.nn.utils.parametrize.is_parametrized(module)
        ):
            continue
        _unwrap_tensor_subclass_parameters(
            child,
            fqn_map,
            used_state_fqns,
            _join_fqn(prefix, child_name),
        )

    return module


def _unwrap_tensor_subclass_parameters_with_state_dict_fqn_map(
    module: torch.nn.Module,
    reserved_state_fqns: set[str] | None = None,
) -> tuple[torch.nn.Module, dict[str, str]]:
    all_state_fqns, subclass_state_fqns = _collect_tensor_subclass_state_fqns(module)
    used_state_fqns = all_state_fqns - subclass_state_fqns
    if reserved_state_fqns is not None:
        used_state_fqns.update(reserved_state_fqns)
    fqn_map: dict[str, str] = {}
    return (
        _unwrap_tensor_subclass_parameters(module, fqn_map, used_state_fqns),
        fqn_map,
    )


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
    return _unwrap_tensor_subclass_parameters(module)
