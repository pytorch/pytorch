"""
This module provides common utilities and base classes for TorchDynamo backends.

Key components:
- AotAutograd: Base class for implementing AOT (Ahead-of-Time) autograd backends
- Backend utilities for handling:
  - Fake tensor conversion
  - Device/dtype detection from inputs
  - Memory efficient fusion
  - Graph flattening
  - Common compiler configurations

The utilities here are used by various backend implementations to handle
common operations and provide consistent behavior across different backends.
AOT autograd functionality is particularly important as it enables ahead-of-time
optimization of both forward and backward passes.
"""

import contextlib
import functools
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from typing_extensions import ParamSpec, TypeVar
from unittest.mock import patch

import torch
from torch._dynamo import disable
from torch._dynamo.exc import TensorifyScalarRestartAnalysis
from torch._dynamo.utils import counters, defake, flatten_graph_inputs, GmWrapper
from torch._functorch.aot_autograd import (
    aot_module_simplified,
    SerializableAOTDispatchCompiler,
)
from torch.utils._python_dispatch import _disable_current_modes


log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _output_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    output_node = next(node for node in gm.graph.nodes if node.op == "output")
    leaves, _ = torch.utils._pytree.tree_flatten(output_node.args[0])
    return [leaf for leaf in leaves if isinstance(leaf, torch.fx.Node)]


def _node_requires_grad(node: torch.fx.Node) -> bool:
    example_value = node.meta.get("example_value")
    if isinstance(example_value, torch.Tensor):
        return example_value.requires_grad
    return True


def _has_output_node_dependency(gm: torch.fx.GraphModule) -> bool:
    # AOTAutograd wraps the FX graph in a custom autograd.Function. When one
    # differentiable output is also an ancestor of another output, a hook
    # registered on the ancestor after this graph returns would need to observe
    # the descendant's internal gradient. Sibling outputs of a custom Function
    # cannot represent that edge, so this graph must run without AOTAutograd.
    outputs = [node for node in _output_nodes(gm) if _node_requires_grad(node)]
    if len(outputs) < 2:
        return False

    output_set = set(outputs)
    for output in outputs:
        seen: set[torch.fx.Node] = set()
        stack = list(output.all_input_nodes)
        while stack:
            node = stack.pop()
            if node in output_set and node is not output:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(node.all_input_nodes)

    return False


def _aot_autograd_fallback_reason(gm: torch.nn.Module) -> str | None:
    graph_module = gm.gm if isinstance(gm, GmWrapper) else gm
    if isinstance(graph_module, torch.fx.GraphModule) and _has_output_node_dependency(
        graph_module
    ):
        return "graph has dependent outputs"
    return None


def _make_aot_autograd_fallback(gm: torch.nn.Module) -> Callable[..., Any]:
    if isinstance(gm, torch.fx.GraphModule):
        return gm

    if not isinstance(gm, GmWrapper):
        raise AssertionError(f"Unexpected AOTAutograd fallback module type: {type(gm)}")

    def boxed_gm(runtime_args: list[Any]) -> Any:
        return gm(*runtime_args)

    return boxed_gm


class AotAutograd:
    def __init__(self, **kwargs: Any) -> None:
        self.__name__ = "compiler_fn"
        self.kwargs = kwargs

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any], **kwargs: Any
    ) -> Callable[..., Any]:
        if kwargs:
            log.warning("aot_autograd-based backend ignoring extra kwargs %s", kwargs)

        fallback_reason = _aot_autograd_fallback_reason(gm)
        if fallback_reason is not None:
            # NB: don't delete counter increment
            counters["aot_autograd"]["total"] += 1
            log.debug("Unable to use AOT Autograd because %s", fallback_reason)
            counters["aot_autograd"]["not_ok"] += 1
            return _make_aot_autograd_fallback(gm)

        if any(isinstance(x, (list, tuple, dict)) for x in example_inputs):
            return flatten_graph_inputs(
                gm,
                example_inputs,
                self,
            )

        # Hack to get around circular import problems with aot_eager_decomp_partition
        if callable(self.kwargs.get("decompositions")):
            self.kwargs["decompositions"] = self.kwargs["decompositions"]()

        # NB: don't delete counter increment
        counters["aot_autograd"]["total"] += 1

        def wrap_bw_compiler(bw_compiler_fn: Callable[P, R]) -> Callable[..., R]:
            def _wrapped_bw_compiler(*args: P.args, **kwargs: P.kwargs) -> R:
                # Note [Wrapping bw_compiler in disable]
                # The two disables here:
                # - stop TorchDynamo from trying to compile the bw_compiler function itself
                # - stop TorchDynamo from trying to compile our the generated backwards pass bw_compiler produces

                return disable(
                    disable(
                        bw_compiler_fn, reason="do not trace backward compiler function"
                    )(*args, **kwargs),  # type: ignore[misc]
                    reason="do not trace generated backwards pass",
                )

            _wrapped_bw_compiler._is_wrapped_bw_compiler = (  # pyrefly: ignore [missing-attribute]
                True
            )
            return _wrapped_bw_compiler

        bw_compiler = self.kwargs.get("bw_compiler") or self.kwargs["fw_compiler"]

        if isinstance(bw_compiler, SerializableAOTDispatchCompiler):
            bw_compiler.compiler_fn = wrap_bw_compiler(bw_compiler.compiler_fn)
        elif getattr(bw_compiler, "_is_wrapped_bw_compiler", False):
            bw_compiler.compiler_fn = bw_compiler
        else:
            bw_compiler = wrap_bw_compiler(bw_compiler)

        self.kwargs["bw_compiler"] = bw_compiler
        self.kwargs["inference_compiler"] = (
            self.kwargs.get("inference_compiler") or self.kwargs["fw_compiler"]
        )

        from functorch.compile import nop
        from torch._inductor.debug import enable_aot_logging

        # debug asserts slow down compile time noticeably,
        # So only default them on when the aot_eager backend is used.
        if self.kwargs.get("fw_compiler", None) is nop:
            patch_config: contextlib.AbstractContextManager[Any] = patch(
                "functorch.compile.config.debug_assert", True
            )
        else:
            patch_config = contextlib.nullcontext()

        try:
            # NB: NOT cloned!
            with enable_aot_logging(), patch_config:
                cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
                counters["aot_autograd"]["ok"] += 1
                return disable(cg, reason="do not trace AOT-compiled graph")
        except TensorifyScalarRestartAnalysis:
            raise
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise


def aot_autograd(**kwargs: Any) -> AotAutograd:
    return AotAutograd(**kwargs)


def mem_efficient_fusion_kwargs(use_decomps: bool) -> dict[str, Any]:
    from functorch.compile import (
        default_decompositions,
        min_cut_rematerialization_partition,
        ts_compile,
    )

    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
    }

    if use_decomps:
        # pyrefly: ignore [bad-typed-dict-key]
        kwargs["decompositions"] = default_decompositions

    return kwargs


def fake_tensor_unsupported(fn: Callable[[Any, list[Any], Any], R]) -> Any:
    """
    Decorator for backends that need real inputs.  We swap out fake
    tensors for zero tensors.
    """

    @functools.wraps(fn)
    def wrapper(model: Any, inputs: Any, **kwargs: Any) -> Any:
        with _disable_current_modes():
            inputs = list(map(defake, inputs))
            return fn(model, inputs, **kwargs)  # type: ignore[call-arg]

    return wrapper


def device_from_inputs(example_inputs: Iterable[Any]) -> torch.device:
    for x in example_inputs:
        if hasattr(x, "device"):
            return x.device
    return torch.device("cpu")  # Default fallback


def dtype_from_inputs(example_inputs: Iterable[Any]) -> torch.dtype:
    for x in example_inputs:
        if hasattr(x, "dtype"):
            return x.dtype
    return torch.float32  # Default fallback
