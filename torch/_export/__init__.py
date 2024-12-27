# mypy: allow-untyped-defs
import copy
import dataclasses
import functools
import io
import json
import logging
import os
import re
import sys
import types
import warnings
import weakref
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import torch
import torch.fx
import torch.utils._pytree as pytree

from torch._dispatch.python import enable_python_dispatcher
from torch._guards import compile_context
from torch._utils_internal import log_export_usage
from torch.export._tree_utils import reorder_kwargs
from torch.export.graph_signature import (
    ArgumentSpec,
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    SymBoolArgument,
    SymFloatArgument,
    TensorArgument,
)
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from .wrappers import _wrap_submodules
from .utils import _materialize_cpp_cia_ops

log = logging.getLogger(__name__)

@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """
    allow_rnn: bool = True


# We only want to print this once to avoid flooding logs in workflows where aot_compile_warning
# is called multiple times.
@lru_cache
def aot_compile_warning():
    from torch._inductor import config

    log.warning("+============================+")
    log.warning("|     !!!   WARNING   !!!    |")
    log.warning("+============================+")
    log.warning(
        "torch._export.aot_compile()/torch._export.aot_load() is being deprecated, please switch to "
        "directly calling torch._inductor.aoti_compile_and_package(torch.export.export())/"
        "torch._inductor.aoti_load_package() instead.")


def aot_compile(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
    same_signature: bool = True,
) -> Union[List[str], str]:
    """
    Note: this function is not stable yet

    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside, generates executable cpp code from the program, and returns
    the path to the generated shared library

    Args:
        f: the `nn.Module` or callable to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        dynamic_shapes: Should either be:
            1) a dict from argument names of ``f`` to their dynamic shape specifications,
            2) a tuple that specifies dynamic shape specifications for each input in original order.
            If you are specifying dynamism on keyword args, you will need to pass them in the order that
            is defined in the original function signature.

            The dynamic shape of a tensor argument can be specified as either
            (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
            not required to include static dimension indices in this dict, but when they are,
            they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
            where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
            are denoted by None. Arguments that are dicts or tuples / lists of tensors are
            recursively specified by using mappings or sequences of contained specifications.

        options: A dictionary of options to control inductor

        disable_constraint_solver: Whether the dim constraint solver must be disabled.

    Returns:
        Path to the generated shared library
    """
    from torch.export._trace import _export_to_torch_ir
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor import config

    aot_compile_warning()

    if config.is_predispatch:
        gm = torch.export._trace._export(f, args, kwargs, dynamic_shapes, pre_dispatch=True).module()
    else:
        # We want to export to Torch IR here to utilize the pre_grad passes in
        # inductor, which run on Torch IR.
        gm = _export_to_torch_ir(
            f,
            args,
            kwargs,
            dynamic_shapes,
            disable_constraint_solver=disable_constraint_solver,
            same_signature=same_signature,
            # Disabling this flag, because instead we can rely on the mapping
            # dynamo_flat_name_to_original_fqn which is coming from Dynamo.
            restore_fqn=False,
        )

    with torch.no_grad():
        so_path = torch._inductor.aot_compile(gm, args, kwargs, options=options)  # type: ignore[arg-type]

    return so_path

def aot_load(so_path: str, device: str) -> Callable:
    """
    Loads a shared library generated by aot_compile and returns a callable

    Args:
        so_path: Path to the shared library

    Returns:
        A callable
    """
    aot_compile_warning()

    if device == "cpu":
        runner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)  # type: ignore[call-arg]
    elif device == "cuda" or device.startswith("cuda:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)  # type: ignore[assignment, call-arg]
    elif device == "xpu" or device.startswith("xpu:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerXpu(so_path, 1, device)  # type: ignore[assignment, call-arg]

    else:
        raise RuntimeError("Unsupported device " + device)

    def optimized(*args, **kwargs):
        call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
        flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized
