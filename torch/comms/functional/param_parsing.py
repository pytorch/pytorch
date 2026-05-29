# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Parameter parsing and schema definitions for collective operations."""

import types
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, get_args, get_origin

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor


__all__ = ["CollectiveParamSchema", "ParamKind", "ParamSpec", "ParsedArgs"]


class ParamKind(Enum):
    CLASS_OBJECT = "class_object"
    INPUT = "input"
    OUTPUT = "output"
    EXTRA = "extra"


# Sentinel value to indicate no default was provided
_NO_DEFAULT = object()


@dataclass
class ParamSpec:
    name: str
    kind: ParamKind
    torch_type: (
        Any  # Can be str, type, or list[type] - converted to str during registration
    )
    default_value: Any = _NO_DEFAULT
    mutable: bool = True
    write_only: bool = (
        False  # If True, tensor is write-only (can use empty_like instead of clone)
    )

    def has_default(self) -> bool:
        return self.default_value is not _NO_DEFAULT

    def is_tensor(self) -> bool:
        return self.torch_type == "Tensor"

    def is_tensor_list(self) -> bool:
        return self.torch_type == "Tensor[]"

    def is_tensor_like(self) -> bool:
        return self.is_tensor() or self.is_tensor_list()


# Mapping from opaque type name to class, populated during registration
_TYPE_NAME_TO_CLASS: dict[str, type] = {}


@dataclass
class ParsedArgs:
    """Parsed arguments for collective operations.

    Provides a consistent interface for parsing args from different call contexts.
    The param layout is inferred from the param specs, not assumed.
    """

    all_params: list[ParamSpec]  # All params in signature order
    values: list[Any]  # Values for each param

    @staticmethod
    def from_lib_args(
        args: tuple,
        all_params: list[ParamSpec],
    ) -> "ParsedArgs":
        """Parse args from torch.library kernel call.

        Args layout: (opaque_object, *params_in_order)
        Values includes the object at index 0, so indices match args directly.
        """
        return ParsedArgs(all_params, list(args))

    @staticmethod
    def from_method_args(
        self_obj: Any,
        args: tuple,
        kwargs: dict,
        all_params: list[ParamSpec],
    ) -> "ParsedArgs":
        """Parse args from patched method call.

        Handles positional args, kwargs, and defaults.
        Note: all_params includes object_param, but args doesn't include self,
        so we skip the first param (object_param) when iterating.
        """
        values = [self_obj]
        # Skip object_param (first param) since self_obj is already added
        non_object_params = all_params[1:]
        for i, p in enumerate(non_object_params):
            if i < len(args):
                values.append(args[i])
            elif p.name in kwargs:
                values.append(kwargs[p.name])
            elif p.has_default():
                values.append(p.default_value)
            else:
                values.append(None)
        return ParsedArgs(all_params, values)

    @cached_property
    def _tensor_input_indices(self) -> list[int]:
        """Indices of tensor-like input params in values/args (includes +1 offset for object)."""
        return [
            i
            for i, p in enumerate(self.all_params)
            if p.kind == ParamKind.INPUT and p.is_tensor_like()
        ]

    @cached_property
    def _mutable_tensor_indices(self) -> list[int]:
        """Indices of mutable tensor-like input params in values/args (includes +1 offset for object)."""
        return [
            i
            for i, p in enumerate(self.all_params)
            if p.kind == ParamKind.INPUT and p.mutable and p.is_tensor_like()
        ]

    @cached_property
    def _value_by_name(self) -> dict[str, Any]:
        """Lookup table for values by param name."""
        result = {}
        for i, p in enumerate(self.all_params):
            if i < len(self.values):
                result[p.name] = self.values[i]
        return result

    @cached_property
    def _has_requires_grad(self) -> bool:
        """Check if any tensor-like input has requires_grad."""
        for idx in self._tensor_input_indices:
            if idx < len(self.values):
                tensor = self.values[idx]
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    return True
                if isinstance(tensor, (list, tuple)):
                    for t in tensor:
                        if isinstance(t, torch.Tensor) and t.requires_grad:
                            return True
        return False

    @cached_property
    def _has_meta(self) -> bool:
        """Check if any tensor-like input has requires_grad."""
        for idx in self._tensor_input_indices:
            if idx < len(self.values):
                tensor = self.values[idx]
                if isinstance(tensor, torch.Tensor) and tensor.is_meta:
                    return True
                if isinstance(tensor, (list, tuple)):
                    for t in tensor:
                        if isinstance(t, torch.Tensor) and t.is_meta:
                            return True
        return False

    @cached_property
    def _has_fake_or_functional_tensor(self) -> bool:
        """Check if any tensor-like input is a FakeTensor or FunctionalTensor."""
        for idx in self._tensor_input_indices:
            if idx < len(self.values):
                tensor = self.values[idx]
                if isinstance(tensor, (FakeTensor, FunctionalTensor)):
                    return True
                if isinstance(tensor, (list, tuple)):
                    for t in tensor:
                        if isinstance(t, (FakeTensor, FunctionalTensor)):
                            return True
        return False

    @cached_property
    def _mutable_outputs(self) -> list:
        """Values at mutable tensor indices."""
        return [
            self.values[idx]
            for idx in self._mutable_tensor_indices
            if idx < len(self.values)
        ]

    @cached_property
    def _mutable_outputs_flat(self) -> list:
        """Flattened mutable outputs."""
        flat_tensors = []
        for t in self._mutable_outputs:
            if isinstance(t, (list, tuple)):
                flat_tensors.extend(t)
            else:
                flat_tensors.append(t)
        return flat_tensors

    @cached_property
    def _tensor_inputs(self) -> list:
        """All tensor input values."""
        return [
            self.values[idx]
            for idx in self._tensor_input_indices
            if idx < len(self.values)
        ]

    @cached_property
    def _tensor_inputs_flat_with_mask(self) -> tuple[list, list[bool]]:
        """Flattened tensor inputs with mutable mask."""
        mutable_set = set(self._mutable_tensor_indices)
        flat_inputs = []
        flat_mutable_mask = []
        for idx in self._tensor_input_indices:
            if idx >= len(self.values):
                continue
            inp = self.values[idx]
            is_mutable = idx in mutable_set
            if isinstance(inp, (list, tuple)):
                flat_inputs.extend(inp)
                flat_mutable_mask.extend([is_mutable] * len(inp))
            else:
                flat_inputs.append(inp)
                flat_mutable_mask.append(is_mutable)
        return flat_inputs, flat_mutable_mask

    def get_value(self, name: str) -> Any:
        """Get a value by param name."""
        return self._value_by_name.get(name)

    def get_tensor_input_indices(self) -> list[int]:
        """Get indices of tensor-like input params."""
        return self._tensor_input_indices

    def get_mutable_tensor_indices(self) -> list[int]:
        """Get indices of mutable tensor-like input params."""
        return self._mutable_tensor_indices

    def has_requires_grad(self) -> bool:
        """Check if any tensor-like input has requires_grad."""
        return self._has_requires_grad

    def has_meta(self) -> bool:
        """Check if any tensor-like input has is_meta."""
        return self._has_meta

    def has_fake_or_functional_tensor(self) -> bool:
        """Check if any tensor-like input is a FakeTensor or FunctionalTensor."""
        return self._has_fake_or_functional_tensor

    def get_mutable_outputs(self) -> list:
        """Get values at mutable tensor indices."""
        return self._mutable_outputs

    def get_mutable_outputs_flat(self) -> list:
        """Get values at mutable tensor indices, flattened if any are lists."""
        return self._mutable_outputs_flat

    def get_tensor_inputs(self) -> list:
        """Get all tensor input values."""
        return self._tensor_inputs

    def get_tensor_inputs_flat_with_mutable_mask(self) -> tuple[list, list[bool]]:
        """Get flattened tensor inputs with a mask indicating which are mutable.

        Returns:
            (flat_inputs, flat_mutable_mask) where flat_mutable_mask[i] is True
            if flat_inputs[i] came from a mutable tensor input.
        """
        return self._tensor_inputs_flat_with_mask

    def to_values(self) -> list:
        """Convert to call arguments."""
        return list(self.values)


@dataclass
class CollectiveParamSchema:
    """Schema for collective operation parameters.

    Centralizes param specs and provides methods to create ParsedArgs.
    """

    object_param: ParamSpec
    input_params: list[ParamSpec]
    output_params: list[ParamSpec]
    extra_params: list[ParamSpec]
    needs_async_dummy_return: bool = False

    @cached_property
    def all_params(self) -> list[ParamSpec]:
        """All params in signature order (obj + inputs + extras)."""
        return [self.object_param] + self.input_params + self.extra_params

    @property
    def has_tensor_outputs(self) -> bool:
        """Whether this op has tensor output params."""
        return len(self.output_params) > 0

    @property
    def num_input_tensors(self) -> int:
        """Number of input tensor params."""
        return len(self.input_params)

    @property
    def num_output_tensors(self) -> int:
        """Number of output tensor params."""
        return len(self.output_params)

    @cached_property
    def mutable_params(self) -> list[ParamSpec]:
        """Mutable tensor params."""
        return [p for p in self.input_params if p.mutable and p.is_tensor_like()]

    @cached_property
    def mutable_indices(self) -> list[int]:
        """Indices of mutable tensor params in all_params (obj + inputs + extras)."""
        return [
            i for i, p in enumerate(self.all_params) if p.mutable and p.is_tensor_like()
        ]

    @staticmethod
    def from_raw_specs(
        target_class: type,
        param_specs: list[ParamSpec],
    ) -> "CollectiveParamSchema":
        """Create a CollectiveParamSchema from raw param specs and target class.

        This handles:
        - Registering the target class as an opaque type
        - Processing param specs to convert types to schema strings
        - Categorizing params into input/output/extra
        - Determining if async_op needs a dummy return tensor

        Args:
            target_class: The class this op belongs to (e.g., TorchComm)
            param_specs: Raw param specs with Python types

        Returns:
            A CollectiveParamSchema ready for use
        """
        from torch._library.opaque_object import (
            get_opaque_type_name,
            is_opaque_type,
            register_opaque_type,
        )

        def _get_schema_type(torch_type: Any) -> str | None:
            """Return the schema type string for known torch types."""
            origin = get_origin(torch_type)
            if origin is types.UnionType:
                args = get_args(torch_type)
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1 and type(None) in args:
                    inner_type = _get_schema_type(non_none_args[0])
                    if inner_type:
                        return f"{inner_type}?"
            if torch_type is torch.dtype:
                return "ScalarType"
            if torch_type is torch.device:
                return "Device"
            if torch_type is torch.Tensor:
                return "Tensor"
            if torch_type is int:
                return "int"
            if torch_type is bool:
                return "bool"
            if torch_type is str:
                return "str"
            if torch_type is float:
                return "float"
            if origin is list:
                args = get_args(torch_type)
                if len(args) == 1 and args[0] is torch.Tensor:
                    return "Tensor[]"
                if len(args) == 1 and args[0] is int:
                    return "int[]"
            if origin is dict:
                args = get_args(torch_type)
                if len(args) == 2 and args[0] is str and args[1] is str:
                    return "Dict(str, str)"
            return None

        # Register the class as opaque type
        if not is_opaque_type(target_class):
            register_opaque_type(target_class, typ="reference")
        opaque_type_name = get_opaque_type_name(target_class)
        _TYPE_NAME_TO_CLASS[opaque_type_name] = target_class

        # Process param specs to convert types
        processed_specs = []
        for spec in param_specs:
            schema_type = _get_schema_type(spec.torch_type)
            if schema_type is not None:
                processed_specs.append(
                    ParamSpec(
                        spec.name,
                        spec.kind,
                        schema_type,
                        spec.default_value,
                        spec.mutable,
                    )
                )
            elif isinstance(spec.torch_type, type):
                if not is_opaque_type(spec.torch_type):
                    register_opaque_type(spec.torch_type, typ="reference")
                type_name = get_opaque_type_name(spec.torch_type)
                _TYPE_NAME_TO_CLASS[type_name] = spec.torch_type
                processed_specs.append(
                    ParamSpec(
                        spec.name,
                        spec.kind,
                        type_name,
                        spec.default_value,
                        spec.mutable,
                    )
                )
            elif isinstance(spec.torch_type, str):
                processed_specs.append(spec)
            else:
                origin = get_origin(spec.torch_type)
                if origin is types.UnionType:
                    args = get_args(spec.torch_type)
                    non_none_args = [a for a in args if a is not type(None)]
                    if len(non_none_args) == 1 and type(None) in args:
                        inner_type = non_none_args[0]
                        if isinstance(inner_type, type):
                            if not is_opaque_type(inner_type):
                                register_opaque_type(inner_type, typ="reference")
                            type_name = get_opaque_type_name(inner_type)
                            _TYPE_NAME_TO_CLASS[type_name] = inner_type
                            processed_specs.append(
                                ParamSpec(
                                    spec.name,
                                    spec.kind,
                                    f"{type_name}?",
                                    spec.default_value,
                                    spec.mutable,
                                )
                            )
                            continue
                raise ValueError(
                    f"Unknown torch_type for param '{spec.name}': {spec.torch_type}. "
                    f"Expected str, type, torch.Tensor, or list[torch.Tensor]"
                )

        # Create object param
        object_param = ParamSpec(
            "self",
            ParamKind.CLASS_OBJECT,
            opaque_type_name,
        )

        # Categorize params
        all_specs = [object_param] + processed_specs
        input_params = [
            p for p in all_specs if p.kind == ParamKind.INPUT and p.is_tensor_like()
        ]
        output_params = [
            p for p in all_specs if p.kind == ParamKind.OUTPUT and p.is_tensor_like()
        ]
        extra_params = [p for p in all_specs if p.kind == ParamKind.EXTRA]

        # Check if this op needs async dummy return
        has_async_op = any(p.name == "async_op" for p in extra_params)
        mutable_params = [p for p in input_params if p.mutable]
        needs_async_dummy_return = has_async_op and len(mutable_params) == 0

        assert not (len(output_params) > 0 and has_async_op), (
            "Async ops cannot have outputs at the moment. Please pass the output as a mutable input instead, or remove the 'async_op' argument from the spec."
        )

        return CollectiveParamSchema(
            object_param=object_param,
            input_params=input_params,
            output_params=output_params,
            extra_params=extra_params,
            needs_async_dummy_return=needs_async_dummy_return,
        )

    @cached_property
    def signature(self) -> str:
        """Build the torch.library signature string for this schema (inplace version).

        Returns a signature like:
        "__torch_bind__.TorchComm self, Tensor(a!) tensor, int op, bool async_op"
        """
        parts = []
        parts.append(f"{self.object_param.torch_type} {self.object_param.name}")

        for i, param in enumerate(self.input_params):
            # For mutable tensor inputs, use Tensor(a!) syntax
            if param.mutable and param.is_tensor_like():
                alias = chr(ord("a") + i)
                if param.is_tensor():
                    type_str = f"Tensor({alias}!)"
                else:  # tensor list
                    type_str = f"Tensor({alias}!)[]"
            else:
                type_str = param.torch_type
            parts.append(f"{type_str} {param.name}")

        for param in self.extra_params:
            parts.append(f"{param.torch_type} {param.name}")

        return ", ".join(parts)

    @cached_property
    def functional_signature(self) -> str:
        """Build the torch.library signature string for functional version.

        Returns a signature like:
        "__torch_bind__.TorchComm self, Tensor tensor, int op, bool async_op"
        (no mutation annotations)
        """
        parts = []
        parts.append(f"{self.object_param.torch_type} {self.object_param.name}")

        for param in self.input_params:
            # Use base type without mutation annotation
            parts.append(f"{param.torch_type} {param.name}")

        for param in self.extra_params:
            parts.append(f"{param.torch_type} {param.name}")

        return ", ".join(parts)

    @cached_property
    def functional_return_type(self) -> str:
        """Build the return type string for functional version.

        Returns the mutable tensor(s) that would be mutated in the inplace version.
        """
        mutable_params = self.mutable_params
        if len(mutable_params) == 0:
            # No mutable inputs - use regular return type
            return self.return_type
        elif len(mutable_params) == 1:
            return mutable_params[0].torch_type
        else:
            return f"({', '.join(p.torch_type for p in mutable_params)})"

    @cached_property
    def inplace_return_type(self) -> str:
        """Build the return type string for inplace version.

        Returns the aliased tensor type(s) for mutable inputs, matching PyTorch
        native inplace ops like `add_(Tensor(a!) self, ...) -> Tensor(a!)`.
        This enables proper mutation tracking in dynamo/functionalization.
        """
        mutable_params = self.mutable_params
        if len(mutable_params) == 0:
            # No mutable inputs - use regular return type
            return self.return_type

        # Build aliased return types matching the input aliases
        # The signature uses aliases a, b, c, ... for mutable inputs
        aliased_types = []
        for i, param in enumerate(self.input_params):
            if param.mutable and param.is_tensor_like():
                alias = chr(ord("a") + i)
                if param.is_tensor():
                    aliased_types.append(f"Tensor({alias}!)")
                else:  # tensor list
                    aliased_types.append(f"Tensor({alias}!)[]")

        if len(aliased_types) == 1:
            return aliased_types[0]
        else:
            return f"({', '.join(aliased_types)})"

    @cached_property
    def return_type(self) -> str:
        """Build the return type string for this schema.

        Only output params (new tensors) are returned.
        Mutable inputs are NOT returned - auto_functionalize handles mutation.
        For async ops without mutable inputs, includes optional Tensor for work tracking.
        """
        return_parts = []
        for param in self.output_params:
            return_parts.append(param.torch_type)

        # Add optional tensor return for async ops without mutable inputs
        if self.needs_async_dummy_return:
            return_parts.append("Tensor?")

        if len(return_parts) == 0:
            return "()"
        elif len(return_parts) == 1:
            return return_parts[0]
        else:
            return f"({', '.join(return_parts)})"

    def parse_lib_args(self, args: tuple) -> ParsedArgs:
        """Parse args from torch.library kernel call."""
        return ParsedArgs.from_lib_args(args, self.all_params)

    def parse_method_args(self, self_obj: Any, args: tuple, kwargs: dict) -> ParsedArgs:
        """Parse args from patched method call."""
        return ParsedArgs.from_method_args(
            self_obj, args, kwargs or {}, self.all_params
        )
