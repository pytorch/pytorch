"""
Registry and API for functional ↔ out variant mappings.

This module provides the infrastructure for registering mappings between
functional custom ops and their corresponding out (mutable) variants.
This enables Inductor to automatically convert functional ops to out variants
for CUDAGraph compatibility and memory optimization.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch._ops import OpOverload


log = logging.getLogger(__name__)


@dataclass
class TensorSpec:
    """Specification for allocating a tensor."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: Union[torch.device, str]
    requires_grad: bool = False

    def allocate(self) -> torch.Tensor:
        """Allocate a tensor with this specification."""
        return torch.empty(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )


@dataclass
class FunctionalToOutMapping:
    """
    Mapping between functional and out variants of an operation.
    """

    functional_op: OpOverload
    out_op: OpOverload
    out_arg_positions: tuple[int, ...]
    output_specs_fn: Optional[Callable[..., list[TensorSpec]]] = None

    @property
    def num_outputs(self) -> int:
        """Number of output tensors."""
        return len(self.out_arg_positions)

    def get_output_specs(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[TensorSpec]:
        """
        Get output tensor specifications from input arguments.
        """
        if self.output_specs_fn is not None:
            return self.output_specs_fn(*args, **kwargs)

        # Default: use fake tensor mode to infer shapes/dtypes
        return self._infer_output_specs_from_fake(args, kwargs)

    def _infer_output_specs_from_fake(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[TensorSpec]:
        """Infer output specs using fake tensor mode."""
        from torch._guards import detect_fake_mode
        from torch._subclasses.fake_tensor import FakeTensorMode

        # Check if we're already in a fake mode
        existing_fake_mode = detect_fake_mode(args)

        if existing_fake_mode is not None:
            # Already have fake tensors, just call the op
            fake_outputs = self.functional_op(*args, **kwargs)
        else:
            # Need to create fake tensors
            with FakeTensorMode() as fake_mode:
                fake_args = tuple(
                    fake_mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
                    for a in args
                )
                fake_kwargs = {
                    k: fake_mode.from_tensor(v) if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
                fake_outputs = self.functional_op(*fake_args, **fake_kwargs)

        # Normalize to tuple
        if isinstance(fake_outputs, torch.Tensor):
            fake_outputs = (fake_outputs,)
        elif not isinstance(fake_outputs, (tuple, list)):
            fake_outputs = (fake_outputs,)

        # Extract specs
        specs = []
        for t in fake_outputs:
            if isinstance(t, torch.Tensor):
                specs.append(
                    TensorSpec(
                        shape=tuple(t.shape),
                        dtype=t.dtype,
                        device=t.device,
                        requires_grad=t.requires_grad,
                    )
                )
            else:
                # Non-tensor output - shouldn't happen for our use case
                log.warning("Non-tensor output in functional op: %s", type(t))

        return specs

    def build_out_args(
        self,
        output_buffers: Sequence[torch.Tensor],
        functional_args: tuple[Any, ...],
        functional_kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Build arguments for the out variant from functional arguments.

        The out variant signature is: out_op(out1, out2, ..., in1, in2, ..., **kwargs)
        """
        # Prepend output buffers to args
        out_args = tuple(output_buffers) + functional_args
        return out_args, functional_kwargs


_FUNCTIONAL_TO_OUT_REGISTRY: dict[OpOverload, FunctionalToOutMapping] = {}


def has_any_registered_mappings() -> bool:
    """
    Check if any functional-to-out mappings are registered.

    This is used for early exit optimization in the decompose pass.
    If no mappings are registered, the pass can skip entirely.
    """
    return len(_FUNCTIONAL_TO_OUT_REGISTRY) > 0


def register_functional_to_out(
    functional_op: OpOverload,
    out_op: OpOverload,
    out_arg_positions: tuple[int, ...],
    output_specs_fn: Optional[Callable[..., list[TensorSpec]]] = None,
) -> None:
    """
    Register a mapping between functional and out variants of an operation.

    This enables Inductor to automatically convert the functional op to its
    out variant during compilation, which is necessary for:
    - CUDAGraph compatibility (requires fixed buffer addresses)
    - Memory optimization (buffer reuse in Phase 2)

    Example:
        >>> # Register scaled_fp4_quant mapping
        >>> register_functional_to_out(
        ...     functional_op=torch.ops.vllm.scaled_fp4_quant,
        ...     out_op=torch.ops.vllm.scaled_fp4_quant_out,
        ...     out_arg_positions=(0, 1),  # output, output_scale
        ... )
    """
    if functional_op in _FUNCTIONAL_TO_OUT_REGISTRY:
        raise ValueError(
            f"Functional op {functional_op} is already registered. "
            f"Existing mapping: {_FUNCTIONAL_TO_OUT_REGISTRY[functional_op]}"
        )

    if not out_arg_positions:
        raise ValueError("out_arg_positions cannot be empty")

    mapping = FunctionalToOutMapping(
        functional_op=functional_op,
        out_op=out_op,
        out_arg_positions=out_arg_positions,
        output_specs_fn=output_specs_fn,
    )

    _FUNCTIONAL_TO_OUT_REGISTRY[functional_op] = mapping


def get_out_variant(functional_op: OpOverload) -> Optional[FunctionalToOutMapping]:
    """
    Get the out variant mapping for a functional op.
    """
    return _FUNCTIONAL_TO_OUT_REGISTRY.get(functional_op)


def clear_registry() -> None:
    """
    Clear all registered mappings.
    """
    _FUNCTIONAL_TO_OUT_REGISTRY.clear()


# =============================================================================
# Decorator API (Alternative registration style)
# =============================================================================


def functional_to_out(
    out_op: Union[str, OpOverload],
    out_arg_positions: tuple[int, ...],
    output_specs_fn: Optional[Callable[..., list[TensorSpec]]] = None,
):
    """
    Decorator to register a functional op with its out variant.

    This is an alternative to calling register_functional_to_out directly.

    Example:
        >>> @functional_to_out(
        ...     out_op="mylib::quant_out",
        ...     out_arg_positions=(0, 1),
        ... )
        >>> @torch.library.custom_op("mylib::quant", mutates_args=())
        >>> def quant_functional(x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]: ...
    """

    def decorator(func_or_op):
        # Get the actual op if this is used after @torch.library.custom_op
        if hasattr(func_or_op, "_opoverload"):
            functional_op = func_or_op._opoverload
        elif isinstance(func_or_op, OpOverload):
            functional_op = func_or_op
        else:
            # Might be the decorated function - defer registration
            # Store metadata for later registration
            func_or_op._functional_to_out_metadata = {
                "out_op": out_op,
                "out_arg_positions": out_arg_positions,
                "output_specs_fn": output_specs_fn,
            }
            return func_or_op

        # Resolve out_op string to OpOverload if needed
        resolved_out_op: OpOverload
        if isinstance(out_op, str):
            # Parse "namespace::op_name" format
            parts = out_op.split("::")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid op name format: {out_op}. Expected 'namespace::op_name'"
                )
            ns, name = parts
            resolved_out_op = getattr(getattr(torch.ops, ns), name)
        else:
            resolved_out_op = out_op

        register_functional_to_out(
            functional_op=functional_op,
            out_op=resolved_out_op,
            out_arg_positions=out_arg_positions,
            output_specs_fn=output_specs_fn,
        )

        return func_or_op

    return decorator
