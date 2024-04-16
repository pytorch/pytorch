# mypy: ignore-errors

import contextlib
import functools
import logging
from unittest.mock import patch

import torch
from torch._dynamo import disable
from torch._dynamo.utils import counters, defake, flatten_graph_inputs
from torch._functorch.aot_autograd import aot_module_simplified
from torch.utils._python_dispatch import _disable_current_modes

log = logging.getLogger(__name__)


def aot_autograd(**kwargs):
    def compiler_fn(gm: torch.fx.GraphModule, example_inputs):
        if any(isinstance(x, (list, tuple, dict)) for x in example_inputs):
            return flatten_graph_inputs(
                gm,
                example_inputs,
                compiler_fn,
            )

        # Hack to get around circular import problems with aot_eager_decomp_partition
        if callable(kwargs.get("decompositions")):
            kwargs["decompositions"] = kwargs["decompositions"]()

        # NB: dont delete counter increment
        counters["aot_autograd"]["total"] += 1
        use_fallback = False

        if use_fallback:
            log.debug("Unable to use AOT Autograd because graph has mutation")
            counters["aot_autograd"]["not_ok"] += 1
            return gm

        # OK attempt to compile

        def _wrapped_bw_compiler(*args, **kwargs):
            # stop TorchDynamo from trying to compile our generated backwards pass
            return disable(disable(bw_compiler)(*args, **kwargs))

        bw_compiler = kwargs.get("bw_compiler") or kwargs["fw_compiler"]
        kwargs["bw_compiler"] = _wrapped_bw_compiler
        kwargs["inference_compiler"] = (
            kwargs.get("inference_compiler") or kwargs["fw_compiler"]
        )

        from functorch.compile import nop

        from torch._inductor.debug import enable_aot_logging

        # debug asserts slow down compile time noticeably,
        # So only default them on when the aot_eager backend is used.
        if kwargs.get("fw_compiler", None) == nop:
            patch_config = patch("functorch.compile.config.debug_assert", True)
        else:
            patch_config = contextlib.nullcontext()

        try:
            # NB: NOT cloned!
            with enable_aot_logging(), patch_config:
                cg = aot_module_simplified(gm, example_inputs, **kwargs)
                counters["aot_autograd"]["ok"] += 1
                return disable(cg)
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise

    return compiler_fn


def mem_efficient_fusion_kwargs(use_decomps):
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
        kwargs["decompositions"] = default_decompositions

    return kwargs


def fake_tensor_unsupported(fn):
    """
    Decorator for backends that need real inputs.  We swap out fake
    tensors for zero tensors.
    """

    @functools.wraps(fn)
    def wrapper(model, inputs, **kwargs):
        with _disable_current_modes():
            inputs = list(map(defake, inputs))
            return fn(model, inputs, **kwargs)

    return wrapper


def device_from_inputs(example_inputs) -> torch.device:
    for x in example_inputs:
        if hasattr(x, "device"):
            return x.device


def dtype_from_inputs(example_inputs) -> torch.dtype:
    for x in example_inputs:
        if hasattr(x, "dtype"):
            return x.dtype
