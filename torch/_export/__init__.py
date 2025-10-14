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

from typing import Any, Optional, TYPE_CHECKING, Union
from collections.abc import Callable
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
from . import config

import contextlib
import operator

import torch




if TYPE_CHECKING:
    from torch._C._aoti import AOTIModelContainerRunner

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

    log.warning("+============================+")
    log.warning("|     !!!   WARNING   !!!    |")
    log.warning("+============================+")
    log.warning(
        "torch._export.aot_compile()/torch._export.aot_load() is being deprecated, please switch to "
        "directly calling torch._inductor.aoti_compile_and_package(torch.export.export())/"
        "torch._inductor.aoti_load_package() instead.")


def aot_compile(
    f: Callable,
    args: tuple[Any],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[dict[str, Any]] = None,
    options: Optional[dict[str, Any]] = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
    same_signature: bool = True,
) -> Union[list[Any], str]:
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
    from torch._inductor import config as inductor_config

    aot_compile_warning()

    if inductor_config.is_predispatch:
        gm = torch.export._trace._export(f, args, kwargs, dynamic_shapes, pre_dispatch=True).module()
    else:
        # We want to export to Torch IR here to utilize the pre_grad passes in
        # inductor, which run on Torch IR.
        with torch._export.config.patch(use_new_tracer_experimental=True):
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

    assert isinstance(so_path, (str, list))
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
        runner: AOTIModelContainerRunner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
    elif device == "cuda" or device.startswith("cuda:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)
    elif device == "xpu" or device.startswith("xpu:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerXpu(so_path, 1, device)
    elif device == "mps" or device.startswith("mps:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerMps(so_path, 1)
    else:
        raise RuntimeError("Unsupported device " + device)

    def optimized(*args, **kwargs):
        call_spec = runner.get_call_spec()
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
        flat_outputs = runner.run(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized


class ModuleWrapper(torch.nn.Module):
    """
    Similar to OptimizedModule in eval_frame.py so that users can find the
    parameters and buffers of the original module.
    """
    def __init__(self, orig_mod, dynamo_gm, jd, optimized_callable):
        super().__init__()
        self._orig_mod = orig_mod
        self.optimized_callable = optimized_callable
        self.jd = jd
        self.dynamo_gm = dynamo_gm
        params_and_buffers = []
        for name in self.jd.params_spec + self.jd.buffers_spec:
            params_and_buffers.append(getattr(dynamo_gm, name))
        self.params_and_buffers = tuple(params_and_buffers)

    def __call__(self, *args, **kwargs):
        # get the params and buffers from the dynamo_gm because that's the input
        # to the aot_export_joint_with_descriptors
        return self.optimized_callable(*self.params_and_buffers, *args, **kwargs)

    # Passthrough for all the methods that make the returned module have same
    # behavior as the original module.
    def __getattr__(self, name: str) -> Any:
        if name == "_orig_mod":
            return self._modules["_orig_mod"]
        return getattr(self._orig_mod, name)


class AotTrainerExport:

    def __init__(self, use_dynamo: bool = False):
        self.orig_mod = None
        self.use_dynamo = use_dynamo
        self.optimized_callable = None

    def generate_joint_graph(self, module: torch.nn.Module, *, args, kwargs):
        self.orig_mod = module
        from torch._functorch.aot_autograd import (
            aot_export_joint_with_descriptors,
            JointWithDescriptors,
        )
        kwargs = kwargs or {}

        # EXPORT_USE_DYNAMO=0 to disable dynamo
        use_dynamo = self.use_dynamo and os.getenv("EXPORT_USE_DYNAMO", "1") == "1"

        if use_dynamo:
            # Remove this once install_free_tensors is on by default
            with torch._dynamo.config.patch(install_free_tensors=True):
                assert isinstance(module, torch.nn.Module)
                from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
                dynamo_gm = _dynamo_graph_capture_for_export(module.__call__)(*args, **kwargs)
        else:
            dynamo_gm = module

        # Set the dynamo_gm because we will use it for the joint graph
        self.dynamo_gm = dynamo_gm


        with contextlib.ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack,
                self.dynamo_gm,
                args=args,
                kwargs=kwargs,
            )
            self.jd = joint_with_descriptors
            self.joint_graph_module = joint_with_descriptors.graph_module

        return self.joint_graph_module

    def compile(self, partition_fn=None, fw_compiler=None, bw_compiler=None):
        from torch._functorch.partitioners import default_partition
        from torch._functorch.aot_autograd import boxed_nop_preserve_node_meta
        from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors

        partition_fn = partition_fn or default_partition
        fw_compiler = fw_compiler or boxed_nop_preserve_node_meta
        bw_compiler = bw_compiler or boxed_nop_preserve_node_meta
        self.optimized_callable = aot_compile_joint_with_descriptors(
            self.jd,
            partition_fn=partition_fn,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler
        )
        return ModuleWrapper(self.orig_mod, self.dynamo_gm, self.jd, self.optimized_callable)


def aot_export_partitioned_graphs_v2(
    module: torch.nn.Module,
    *,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    transforms: Any,
    num_fwd_outputs: int = 1,
    use_dynamo: bool = False,
):
    exporter = AotTrainerExport(use_dynamo=use_dynamo)

    exporter.generate_joint_graph(module, args=args, kwargs=kwargs)
    return exporter
