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
from contextlib import ExitStack, contextmanager
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

if TYPE_CHECKING:
    from torch._C._aoti import AOTIModelContainerRunner

log = logging.getLogger(__name__)


# Dataclasses for compiler pipeline outputs
@dataclasses.dataclass
class PartitionOutput:
    """Output from the partition stage of the joint graph compiler."""

    fw_module: torch.fx.GraphModule
    bw_module: torch.fx.GraphModule
    num_fw_outs_saved_for_bw: int
    num_symints_saved_for_bw: int
    indices_of_inps_to_detach: list[int]
    adjusted_flat_args: list[Any]


@dataclasses.dataclass
class ForwardCompileOutput:
    """Output from the forward compilation stage of the joint graph compiler."""

    fwd_output_strides: Optional[list[Optional[tuple[int, ...]]]]
    compiled_fw_func: Callable


@dataclasses.dataclass
class BackwardCompileOutput:
    """Output from the backward compilation stage of the joint graph compiler."""

    lazy_backward_info: Any  # AutogradLazyBackwardCompileInfo - avoid circular import
    compiled_bw_func: Optional[Callable]


def _make_callable(
    compiled_fn: Callable,
    gm: torch.fx.GraphModule,
    params_spec: list[str],
    buffers_spec: list[str],
    in_spec: pytree.TreeSpec,
    out_spec: pytree.TreeSpec,
) -> Callable:
    """
    Wrap the compiled function to provide a cleaner calling convention.

    The compiled function expects flat args: [*params, *buffers, *flat_inputs]
    This wrapper allows calling with just: (*inputs) where inputs can be structured

    Args:
        compiled_fn: The compiled function from make_autograd_function
        gm: The graph module from capture_graph (should have _restore_state_dict called on it)
        params_spec: List of parameter FQNs in order
        buffers_spec: List of buffer FQNs in order
        in_spec: Input pytree spec for flattening structured inputs
        out_spec: Output pytree spec for unflattening structured outputs

    Returns:
        A callable that takes structured user inputs and returns structured outputs
    """
    # Get parameter and buffer dictionaries from graph module
    params_dict = dict(gm.named_parameters())
    buffers_dict = dict(gm.named_buffers())

    # Look up params and buffers by FQN in the order specified by specs
    params = [params_dict[fqn] for fqn in params_spec]
    buffers = [buffers_dict[fqn] for fqn in buffers_spec]

    def wrapper(*args, **kwargs):
        # Flatten the inputs using in_spec to handle structured inputs (dicts, tuples, etc.)
        # The in_spec includes params/buffers/inputs, so we reconstruct the full input structure
        # and flatten just the user inputs portion
        user_inputs_flat, _ = pytree.tree_flatten((args, kwargs))
        # Construct the full flat args list
        flat_args = [*params, *buffers, *user_inputs_flat]
        # Call the compiled function
        flat_outputs = compiled_fn(flat_args)
        # Unflatten outputs using out_spec to handle structured outputs
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return wrapper


class CompilerPipeline:
    """
    A unified pipeline for graph capture, joint graph generation, and compilation.

    This class provides an end-to-end API for:
    1. Capturing a graph from an nn.Module using Dynamo
    2. Generating a joint forward-backward graph with descriptors
    3. Partitioning the joint graph into forward and backward graphs
    4. Compiling forward and backward graphs
    5. Creating autograd or inference functions
    """

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the CompilerPipeline.

        Args:
            model: The nn.Module to capture and compile
            inputs: Example input tensors for tracing
            kwargs: Optional keyword arguments for the model

        NOTE: You must use the following pattern to use this class:

        pipeline = CompilerPipeline(...)  # capture graph
        with pipeline.stack:
            # call pipeline methods to compile
        """
        self.model = model
        self.inputs = inputs
        self.kwargs = kwargs if kwargs is not None else {}
        self._gm = self.capture_graph()

        self.stack = ExitStack()

        self.joint_with_descriptors: Optional[Any] = None  # JointWithDescriptors

        # Internal state for compilation phases
        self._aot_config: Optional[Any] = None
        self._fw_metadata: Optional[Any] = None
        self._maybe_subclass_meta: Optional[Any] = None

    def capture_graph(self) -> torch.fx.GraphModule:
        """
        Capture the graph from the model using Dynamo graph capture.

        Returns:
            The captured GraphModule
        """
        from torch._dynamo.functional_export import _dynamo_graph_capture_for_export, _restore_state_dict

        # Capture graph for export
        with torch._dynamo.config.patch(install_free_tensors=True):
            assert isinstance(self.model, torch.nn.Module)
            gm = _dynamo_graph_capture_for_export(self.model)(*self.inputs, **self.kwargs)

        # Restore state dict to the captured graph module
        _restore_state_dict(self.model, gm)

        return gm

    def generate_aten_ir_for_training(
        self,
        decompositions: Optional[dict] = None,
        keep_inference_input_mutations: bool = False,
        ignore_shape_env: bool = False,
        disable_functionalization: bool = False,
    ) -> Any:  # JointWithDescriptors
        """
        Generate ATen IR for training (joint forward-backward graph with descriptors).

        Args:
            decompositions: Optional decomposition table for operations
            keep_inference_input_mutations: Whether to keep input mutations in inference mode
            ignore_shape_env: Whether to ignore shape environment
            disable_functionalization: Whether to disable functionalization

        Returns:
            JointWithDescriptors object containing the joint graph and metadata
        """
        from torch._functorch.aot_autograd import aot_export_joint_with_descriptors

        if self._gm is None:
            self._gm = self.capture_graph()

        self.stack.enter_context(self._gm.meta["fake_mode"])  # type: ignore[union-attr]

        # Export joint graph with descriptors
        self.joint_with_descriptors = aot_export_joint_with_descriptors(
            self.stack,
            self._gm,  # type: ignore[arg-type]
            self.inputs,
            self.kwargs,
            decompositions=decompositions,
            keep_inference_input_mutations=keep_inference_input_mutations,
            ignore_shape_env=ignore_shape_env,
            disable_functionalization=disable_functionalization,
        )

        return self.joint_with_descriptors

    def generate_aten_ir_for_inference(
        self,
        decompositions: Optional[dict] = None,
        keep_inference_input_mutations: bool = False,
        ignore_shape_env: bool = False,
        disable_functionalization: bool = False,
    ) -> Any:  # JointWithDescriptors
        """
        Generate ATen IR for inference (forward graph with descriptors).

        Args:
            decompositions: Optional decomposition table for operations
            keep_inference_input_mutations: Whether to keep input mutations in inference mode
            ignore_shape_env: Whether to ignore shape environment
            disable_functionalization: Whether to disable functionalization

        Returns:
            JointWithDescriptors object containing the forward graph and metadata
        """
        with torch.no_grad():
            return self.generate_aten_ir_for_training(
                decompositions=decompositions,
                keep_inference_input_mutations=keep_inference_input_mutations,
                ignore_shape_env=ignore_shape_env,
                disable_functionalization=disable_functionalization,
            )

    def _ensure_state_initialized(self):
        """Initialize internal state from joint_with_descriptors if not already done."""
        if self._aot_config is None:
            if self.joint_with_descriptors is None:
                raise RuntimeError("Must call generate_aten_ir_for_training() or generate_aten_ir_for_inference() first")

            aot_state = self.joint_with_descriptors._aot_state
            aot_graph_capture = self.joint_with_descriptors._aot_graph_capture

            self._aot_config = aot_state.aot_config
            self._fw_metadata = aot_state.fw_metadata
            self._maybe_subclass_meta = aot_graph_capture.maybe_subclass_meta

    def partition(
        self,
        partition_fn: Callable,
        fx_g: torch.fx.GraphModule,
        joint_inputs: Union[list[Any], tuple[list[Any], list[Any]]],
    ) -> PartitionOutput:
        """
        Partition the joint graph into a forward graph and a backward graph.

        Args:
            partition_fn: Partition function to use
            fx_g: The joint graph module to partition
            joint_inputs: Flattened inputs to the joint graph

        Returns:
            PartitionOutput containing the partitioned forward and backward modules
        """
        self._ensure_state_initialized()
        self._aot_config.partition_fn = partition_fn  # type: ignore[union-attr]

        from torch._functorch._aot_autograd.graph_compile import _aot_stage2a_partition

        result = _aot_stage2a_partition(
            fx_g, joint_inputs, self._maybe_subclass_meta, self._fw_metadata, self._aot_config  # type: ignore[arg-type]
        )
        return PartitionOutput(
            fw_module=result[0],
            bw_module=result[1],
            num_fw_outs_saved_for_bw=result[2],
            num_symints_saved_for_bw=result[3],
            indices_of_inps_to_detach=result[4],
            adjusted_flat_args=result[5],
        )

    def fw_compile(
        self,
        fw_compiler: Callable,
        fw_module: torch.fx.GraphModule,
        adjusted_flat_args: list[Any],
        num_fw_outs_saved_for_bw: int,
    ) -> ForwardCompileOutput:
        """
        Compile the forward graph.

        Args:
            fw_compiler: Compiler function to use for forward graph
            fw_module: The forward graph module to compile
            adjusted_flat_args: Flattened arguments after adjustments
            num_fw_outs_saved_for_bw: Number of forward outputs saved for backward

        Returns:
            ForwardCompileOutput containing strides and compiled forward function
        """
        self._ensure_state_initialized()
        self._aot_config.fw_compiler = fw_compiler  # type: ignore[union-attr]

        from torch._functorch._aot_autograd.graph_compile import _aot_stage2b_fw_compile

        result = _aot_stage2b_fw_compile(
            fw_module,
            adjusted_flat_args,
            self._maybe_subclass_meta,
            self._fw_metadata,  # type: ignore[arg-type]
            num_fw_outs_saved_for_bw,
            self._aot_config,  # type: ignore[arg-type]
        )
        return ForwardCompileOutput(
            fwd_output_strides=result[0],
            compiled_fw_func=result[1],
        )

    def bw_compile(
        self,
        bw_compiler: Callable,
        bw_module: torch.fx.GraphModule,
        fwd_output_strides: Optional[list[Optional[tuple[int, ...]]]],
        num_symints_saved_for_bw: int,
    ) -> BackwardCompileOutput:
        """
        Compile the backward graph.

        Args:
            bw_compiler: Compiler function to use for backward graph
            bw_module: The backward graph module to compile
            fwd_output_strides: Output strides from forward compilation
            num_symints_saved_for_bw: Number of symbolic ints saved for backward

        Returns:
            BackwardCompileOutput containing lazy backward info and compiled backward function
        """
        self._ensure_state_initialized()
        self._aot_config.bw_compiler = bw_compiler  # type: ignore[union-attr]

        from torch._functorch._aot_autograd.graph_compile import _aot_stage2b_bw_compile

        result = _aot_stage2b_bw_compile(
            bw_module,
            self._maybe_subclass_meta,
            self._fw_metadata,  # type: ignore[arg-type]
            fwd_output_strides,
            num_symints_saved_for_bw,
            self._aot_config,  # type: ignore[arg-type]
        )
        return BackwardCompileOutput(
            lazy_backward_info=result[0],
            compiled_bw_func=result[1],
        )

    def inference_compile(
        self,
        inference_compiler: Callable,
        fw_module: torch.fx.GraphModule,
        updated_flat_args: list[Any],
    ) -> Callable:
        """
        Compile the inference graph (no autograd).

        Args:
            inference_compiler: Compiler function to use for inference graph
            fw_module: The forward/inference graph module to compile
            updated_flat_args: Flattened arguments after adjustments

        Returns:
            Compiled inference function
        """
        self._ensure_state_initialized()
        self._aot_config.inference_compiler = inference_compiler  # type: ignore[union-attr]

        from torch._functorch._aot_autograd.graph_compile import _aot_stage2b_inference_compile

        return _aot_stage2b_inference_compile(
            fw_module,
            updated_flat_args,
            self._maybe_subclass_meta,
            self._fw_metadata,  # type: ignore[arg-type]
            self._aot_config,
        )

    def make_inference_function(
        self,
        compiled_fw: Callable,
        wrappers: list[Any],  # list[CompilerWrapper]
        entry: Optional[Any],  # Optional[GenericAOTAutogradCacheEntry]
    ) -> Callable:
        """
        Make the final inference function with clean calling convention.

        Args:
            compiled_fw: Compiled forward function
            wrappers: List of compiler wrappers to apply
            entry: Optional cache entry

        Returns:
            Callable with clean calling convention (takes structured user inputs)
        """
        self._ensure_state_initialized()

        from torch._functorch._aot_autograd.graph_compile import _aot_stage2c_make_inference_function

        compiled_fn, _ = _aot_stage2c_make_inference_function(
            self._aot_config,
            self._fw_metadata,
            compiled_fw,
            wrappers,
            entry,
        )

        return _make_callable(
            compiled_fn,
            self._gm,
            self.joint_with_descriptors.params_spec,  # type: ignore[union-attr]
            self.joint_with_descriptors.buffers_spec,  # type: ignore[union-attr]
            self.joint_with_descriptors.in_spec,  # type: ignore[union-attr]
            self.joint_with_descriptors.out_spec,  # type: ignore[union-attr]
        )

    def make_autograd_function(
        self,
        flat_args: list[Any],
        wrappers: list[Any],  # list[CompilerWrapper]
        compiled_fw_func: Callable,
        compiled_bw_func: Optional[Callable],
        lazy_backward_info: Any,  # AutogradLazyBackwardCompileInfo
        indices_of_inps_to_detach: list[int],
        num_symints_saved_for_bw: int,
        try_save_cache_entry: Optional[Callable] = None,
        entry: Optional[Any] = None,  # Optional[GenericAOTAutogradCacheEntry]
    ) -> Callable:
        """
        Make the final autograd function with clean calling convention.

        Args:
            flat_args: Flattened input arguments
            wrappers: List of compiler wrappers to apply
            compiled_fw_func: Compiled forward function
            compiled_bw_func: Optional compiled backward function
            lazy_backward_info: Information for lazy backward compilation
            indices_of_inps_to_detach: Indices of inputs to detach
            num_symints_saved_for_bw: Number of symbolic ints saved for backward
            try_save_cache_entry: Optional callback to save cache entry
            entry: Optional existing cache entry

        Returns:
            Callable with clean calling convention (takes structured user inputs)
        """
        self._ensure_state_initialized()

        from torch._functorch._aot_autograd.graph_compile import _aot_stage2c_make_autograd_function

        compiled_fn, _ = _aot_stage2c_make_autograd_function(
            self._aot_config,
            flat_args,
            self._fw_metadata,
            self._maybe_subclass_meta,
            wrappers,
            compiled_fw_func,
            compiled_bw_func,
            lazy_backward_info,
            try_save_cache_entry,
            entry,
            indices_of_inps_to_detach,
            num_symints_saved_for_bw,
        )

        return _make_callable(
            compiled_fn,
            self._gm,
            self.joint_with_descriptors.params_spec,  # type: ignore[union-attr]
            self.joint_with_descriptors.buffers_spec,  # type: ignore[union-attr]
            self.joint_with_descriptors.in_spec,  # type: ignore[union-attr]
            self.joint_with_descriptors.out_spec,  # type: ignore[union-attr]
        )

    @property
    def graph_module(self) -> Optional[torch.fx.GraphModule]:
        """Get the captured graph module."""
        return self._gm

    @property
    def aot_state(self) -> Optional[Any]:
        """Get the AOT state from the joint graph."""
        return self.joint_with_descriptors._aot_state if self.joint_with_descriptors else None

    @property
    def aot_graph_capture(self) -> Optional[Any]:
        """Get the AOT graph capture from the joint graph."""
        return self.joint_with_descriptors._aot_graph_capture if self.joint_with_descriptors else None

    @property
    def params_spec(self) -> Optional[list[str]]:
        """Get the parameter specification from the joint graph."""
        return self.joint_with_descriptors.params_spec if self.joint_with_descriptors else None

    @property
    def buffers_spec(self) -> Optional[list[str]]:
        """Get the buffer specification from the joint graph."""
        return self.joint_with_descriptors.buffers_spec if self.joint_with_descriptors else None

    @property
    def in_spec(self) -> Optional[pytree.TreeSpec]:
        """Get the input tree specification from the joint graph."""
        return self.joint_with_descriptors.in_spec if self.joint_with_descriptors else None

    @property
    def out_spec(self) -> Optional[pytree.TreeSpec]:
        """Get the output tree specification from the joint graph."""
        return self.joint_with_descriptors.out_spec if self.joint_with_descriptors else None



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
