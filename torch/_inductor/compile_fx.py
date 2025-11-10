from __future__ import annotations

import contextlib
import copy
import enum
import functools
import io
import itertools
import json
import logging
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from inspect import currentframe
from itertools import count
from operator import attrgetter
from typing import Any, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import Never, override, ParamSpec, Protocol, TypedDict, Unpack
from unittest import mock

import torch._inductor.async_compile
import torch.fx
import torch.utils._pytree as pytree
from functorch.compile import min_cut_rematerialization_partition
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import (
    compiled_autograd,
    config as dynamo_config,
    logging as dynamo_logging,
    utils as dynamo_utils,
)
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.repro.after_aot import wrap_compiler_debug
from torch._dynamo.utils import (
    chromium_event_timed,
    CompileEventLogger,
    counters,
    detect_fake_mode,
    dynamo_timed,
    flatten_graph_inputs,
    get_metrics_context,
    lazy_format_graph_code,
    set_feature_use,
)
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.subclass_parametrization import (
    unwrap_tensor_subclass_parameters,
)
from torch._functorch.aot_autograd import (
    aot_export_module,
    GraphOutputName,
    make_boxed_func,
    SerializableAOTDispatchCompiler,
)
from torch._inductor.codecache import code_hash, FxGraphCache, output_code_log
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    format_default_skip_message,
    log_cudagraph_skip_and_bump_counter,
    PlaceholderInfo,
)
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch._inductor.debug import (
    create_mapping_pre_post_grad_nodes,
    save_args_for_compile_fx_inner,
)
from torch._inductor.output_code import (
    CompiledAOTI,
    CompiledFxGraph,
    CompiledFxGraphConstantsWithGm,
    get_expanded_dims,
    index_expanded_dims,
    OutputCode,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.utils import (
    BoxedBool,
    count_tangents,
    fresh_cache,
    get_all_devices,
    InputType,
    is_gpu,
    should_assume_input_aligned,
    should_use_remote_fx_graph_cache,
    tensor_is_aligned,
)
from torch._library.fake_class_registry import FakeScriptObject
from torch._logging import trace_structured
from torch._utils_internal import compile_time_strobelight_meta
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymExprPrinter
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.monitor import _WaitCounter
from torch.utils._ordered_set import OrderedSet

from .._dynamo.backends.common import aot_autograd
from .._dynamo.exc import ShortenTraceback, SkipFrame
from ..fx._lazy_graph_module import _use_lazy_graph_module
from ..fx.graph import _PyTreeCodeGen
from ..utils._triton import has_triton
from . import config, distributed_autotune, metrics
from .codegen.common import get_wrapper_codegen_for_device, init_backend_registration
from .debug import DebugContext
from .decomposition import select_decomp_table
from .exc import InductorError
from .fx_passes.joint_graph import joint_graph_passes
from .fx_passes.post_grad import post_grad_passes, view_to_reshape
from .fx_passes.pre_grad import pre_grad_passes
from .graph import GraphLowering
from .ir import get_device_type, IRNode
from .output_code import complex_memory_overlap  # noqa: F401
from .triton_bundler import TritonBundler
from .utils import (
    align_inputs_from_check_idxs,
    clone_preserve_strides,
    copy_misaligned_inputs,
    get_cloned_parameter_buffer_name,
    get_first_incompatible_cudagraph_node,
    maybe_get_suppress_shape_guards_ctx,
    output_node,
    remove_unaligned_input_idxs,
    shape_env_from_inputs,
)
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch._inductor.output_code import _StrideExprStr
    from torch._ops import OpOverload
    from torch.export.pt2_archive._package_weights import Weights

    from .ir import ExternKernelNode


_P = ParamSpec("_P")
_T = TypeVar("_T")

if TYPE_CHECKING or not config.is_fbcode():
    # no-op decorator
    def time_and_log(attr: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        return dynamo_utils.identity

    def log_optimus_to_scuba(*args: object, **kwargs: object) -> None:
        pass

else:
    from torch._inductor.fb.utils import log_optimus_to_scuba, time_and_log

if TYPE_CHECKING:
    import types

    from torch._functorch._aot_autograd.schemas import (
        FQN,
        GraphInputName,
        GraphSignature,
    )

    CompileFxOutput = Union[
        Callable[[list[object]], Sequence[torch.Tensor]],
        str,
        list[str],
        Weights,
    ]


class FxCompileMode(enum.Enum):
    NORMAL = 0
    # For testing - use the serde FxCompile scheme to debug serialization and
    # deserialization of GraphMoule and CompiledFxGraph.
    SERIALIZE = 1
    # Compile using a subprocess instead of in-process.
    SUBPROCESS = 2


@dataclass
class FxCompileConfig:
    mode: FxCompileMode
    use_async: bool
    use_progressive: bool


def _fx_compile_mode_default() -> FxCompileConfig:
    name = "TORCHINDUCTOR_FX_COMPILE_MODE"
    value = os.environ.get(name)
    if value is None:
        return FxCompileConfig(FxCompileMode.NORMAL, False, False)

    use_async = False
    use_progressive = False

    if value.lower().startswith("progressive+"):
        use_progressive = True
        value = value[12:]
    if value.lower().startswith("async+"):
        use_async = True
        value = value[6:]

    try:
        value = value.upper()
        return FxCompileConfig(FxCompileMode[value], use_async, use_progressive)
    except KeyError:
        import logging

        log = logging.getLogger(__name__)
        log.error(
            "Invalid value of %s for %s. Expected one of %s. Using default.",
            value,
            name,
            ", ".join(sorted(repr(x) for x in FxCompileMode.__members__)),
        )
        # Remove from the environment so subprocesses don't ALSO complain.
        os.environ.pop(name)
        return FxCompileConfig(FxCompileMode.NORMAL, False, False)


def _get_progression_configs() -> list[dict[str, Any]]:
    # TODO make this configurable
    return [
        {"max_autotune": True},
    ]


_fx_compile_config = _fx_compile_mode_default()
fx_compile_mode = _fx_compile_config.mode
fx_compile_async = _fx_compile_config.use_async
fx_compile_progressive = _fx_compile_config.use_progressive

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
pre_grad_graphs_log = torch._logging.getArtifactLogger(__name__, "pre_grad_graphs")
post_grad_graphs_log = torch._logging.getArtifactLogger(__name__, "post_grad_graphs")
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)
inductor_metrics_log = torch._logging.getArtifactLogger(__name__, "inductor_metrics")


def get_static_input_idxs(num_fixed: int) -> list[int]:
    # If we are inlining NNModules, we treat all torch.nn.Parameters as static for the purposes
    # of cudagraphs. Rather than copying these into cudagraph-owned memory
    # like we do for normal inputs on each run, we will re-record a cudagraph if these
    # parameter locations change.
    context = torch._guards.TracingContext.try_get()
    fixed = list(range(num_fixed))
    if not context or not context.fw_metadata:
        return fixed

    return context.fw_metadata.static_input_indices


def record_original_output_strides(gm: GraphModule) -> None:
    output_node = gm.graph.find_nodes(op="output")[0]
    output_strides = []

    if not isinstance(output_node.args[0], torch.fx.Node):
        output_node_args = output_node.args[0]
    else:
        output_node_args = output_node.args

    for output in output_node_args:
        if (
            isinstance(output, torch.fx.Node)
            and (val := output.meta.get("val")) is not None
            and isinstance(val, torch.Tensor)
        ):
            output_strides.append(val.stride())
        else:
            # pyrefly: ignore [bad-argument-type]
            output_strides.append(None)
    output_node.meta["original_output_strides"] = output_strides


def _recursive_record_original_output_strides(gm: GraphModule) -> None:
    # invoke_subgraph HOP requires output strides to be respected
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        subgraph = getattr(gm, node.args[0].target)
        _recursive_record_original_output_strides(subgraph)

    record_original_output_strides(gm)


def _recursive_record_user_visible_output_idxs(gm: GraphModule) -> None:
    # invoke_subgraph HOP requires output strides to be respected
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        subgraph = getattr(gm, node.args[0].target)

        for node in subgraph.graph.find_nodes(op="output"):
            node.meta["user_visible_output_idxs"] = [
                idx
                for idx in range(len(node.args[0]))
                if isinstance(node.args[0][idx], torch.fx.Node)
            ]
        _recursive_record_user_visible_output_idxs(subgraph)


@functools.lru_cache(None)
def _step_logger() -> Callable[..., None]:
    return dynamo_logging.get_step_logger(log)


@functools.cache
def _warn_tf32_disabled() -> None:
    if (
        torch.cuda.is_available()
        and not torch.backends.cuda.matmul.allow_tf32
        and torch.cuda.get_device_capability() >= (8, 0)
    ):
        warnings.warn(
            "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. "
            "Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
        )


def _resolve_name_collision(mod: GraphModule, gm: GraphModule) -> None:
    """
    In aot_export_module (make_fx), we create get_attr nodes with name prefix
    "_tensor_constant" and "_torchbind_obj". See Tracer.create_arg() in
    torch/fx/_symbolic_trace.py

    However, this might result in name collision if the original mod already
    has a different buffer with the same name.

    We resolve this potential name collision here by changing the target name
    with a new number post fix.
    """

    existing_keys = OrderedSet(
        [name for name, val in mod.named_parameters(remove_duplicate=False)]
    )
    existing_keys.update(
        OrderedSet([name for name, val in mod.named_buffers(remove_duplicate=False)])
    )

    def find_smallest_i(graph: fx.Graph, prefix: str) -> int:
        i = 0
        for node in graph.nodes:
            if node.op == "get_attr" and node.target.startswith(prefix):
                if len(node.target) > len(prefix):
                    post_fix = node.target.split(prefix)[-1]
                    if post_fix.isdigit():
                        i = max(i, int(post_fix))
        for key in existing_keys:
            if key.startswith(prefix):
                if len(key) > len(prefix):
                    post_fix = key.split(prefix)[-1]
                    if post_fix.isdigit():
                        i = max(i, int(post_fix))
        return i + 1

    for node in gm.graph.nodes:
        if node.op == "get_attr":
            target_name = node.target
            if not target_name.startswith(
                "_tensor_constant"
            ) and not target_name.startswith("_torchbind_obj"):
                continue

            if not hasattr(mod, target_name):
                continue
            gm_target = attrgetter(target_name)(gm)
            model_target = attrgetter(target_name)(mod)
            if isinstance(gm_target, FakeScriptObject):
                if (
                    isinstance(model_target, FakeScriptObject)
                    and gm_target.real_obj is model_target.real_obj
                ):
                    continue
            elif (
                gm_target.device == model_target.device
                and gm_target.dtype == model_target.dtype
                and torch.equal(gm_target, model_target)
            ):
                # If tensors with same name from gm and model are indeed the same, we don't need to rename
                # Check device first, to avoid torch.equal(wrapper_CUDA__equal) raise when different device
                continue

            prefix = (
                "_tensor_constant"
                if target_name.startswith("_tensor_constant")
                else "_torchbind_obj"
            )
            new_id = find_smallest_i(gm.graph, prefix)
            new_target_name = f"{prefix}{new_id}"
            node.target = new_target_name
            setattr(gm, new_target_name, gm_target)
            existing_keys.add(new_target_name)


def _unlift_graph(
    mod: GraphModule, gm: GraphModule, graph_signature: GraphSignature
) -> GraphModule:
    from torch.export.unflatten import _assign_attr, _AttrKind

    _resolve_name_collision(mod, gm)

    state_dict: dict[str, Union[torch.nn.parameter.Parameter, torch.Tensor]] = {}
    for name, param in mod.named_parameters(remove_duplicate=False):
        state_dict[name] = param
        _assign_attr(
            param,
            gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )
    for name, buffer in mod.named_buffers(remove_duplicate=False):
        state_dict[name] = buffer
        _assign_attr(
            buffer,
            gm,
            name,
            attr_kind=_AttrKind.BUFFER,
        )

    placeholder_nodes = gm.graph.find_nodes(op="placeholder")
    lifted_inputs: list[Optional[FQN]] = []

    # In AOTI, module parameters and buffers are not lifted as graph inputs.
    # As a result, mutation to buffers has side effect which makes their initial
    # values different from Eager. So we clone them here as a copy.
    # We are not cloning for parameters, although it will be needed if we want to
    # support training.
    for node in placeholder_nodes:
        node_name = node.name
        if node_name in graph_signature.inputs_to_parameters:
            parameter_name = graph_signature.inputs_to_parameters[node_name]
            lifted_inputs.append(parameter_name)
        elif node_name in graph_signature.inputs_to_buffers:
            buffer_name = graph_signature.inputs_to_buffers[node_name]
            lifted_inputs.append(buffer_name)
            gm.meta[get_cloned_parameter_buffer_name(buffer_name)] = (
                clone_preserve_strides(state_dict[buffer_name])
            )
        else:
            assert node_name in graph_signature.user_inputs
            lifted_inputs.append(None)

    from torch.export._unlift import _unlift

    outputs: tuple[torch.fx.Node, ...] = tuple(gm.graph.output_node().args[0])  # type: ignore[arg-type]
    mutated_outputs = []
    buffer_mutations = graph_signature.buffers_to_mutate
    user_input_mutations = graph_signature.user_inputs_to_mutate
    output_tokens = graph_signature.output_tokens
    for idx, out in enumerate(outputs):
        value: Optional[Union[FQN, GraphInputName]] = None

        if idx < len(buffer_mutations) + len(user_input_mutations) + len(output_tokens):
            name = GraphOutputName(out.name)
            if name in buffer_mutations:
                value = buffer_mutations[name]
            elif name in user_input_mutations:
                value = user_input_mutations[name]

        mutated_outputs.append(value)

    unlifted_gm = _unlift(
        gm,
        lifted_inputs,
        mutated_outputs,
        pytree.treespec_leaf(),
        None,
    )
    return unlifted_gm


def _get_subgraph_names(
    gm: GraphModule, skip_invoke_subgraph: bool = False
) -> Generator[str, None, None]:
    all_subgraph_names: OrderedSet[str] = OrderedSet(
        x.target for x in gm.graph.find_nodes(op="get_attr")
    )
    fx_subgraph_names: OrderedSet[str] = OrderedSet()
    for child_name, child_module in gm.named_children():
        # Sometimes an owning_module can have unused children. Skip them
        # by checking them from get_attr node targets.
        if child_name in all_subgraph_names and isinstance(
            child_module, torch.fx.GraphModule
        ):
            fx_subgraph_names.add(child_name)

    if skip_invoke_subgraph:
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        ):
            fx_subgraph_names.discard(node.args[0].target)

    yield from fx_subgraph_names


def _recursive_pre_grad_passes(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
) -> GraphModule:
    with dynamo_timed(
        "_recursive_pre_grad_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="pre_grad_pass_time_us",
    ):
        add_passes = config.add_pre_grad_passes
        remove_passes = config.remove_pre_grad_passes
        for subgraph_name in _get_subgraph_names(gm):
            subgraph = getattr(gm, subgraph_name)
            # as we don't have recursive example inputs, passing empty set here
            new_subgraph = _recursive_pre_grad_passes(subgraph, ())
            setattr(gm, subgraph_name, new_subgraph)
        return pre_grad_passes(gm, example_inputs, add_passes, remove_passes)


def _recursive_joint_graph_passes(
    gm: GraphModule, skip_invoke_subgraph: bool = False
) -> None:
    with dynamo_timed(
        "_recursive_joint_graph_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="joint_graph_pass_time_us",
    ):
        # invoke_subgraph already runs the _recursive_joint_graph_passes.  In
        # AOTAutograd, `run_joint_graph_passes_on_hops` partitions the
        # invoke_subgraph HOP before calling the partitioner on the outer graph.
        # AOTAutograd has access to partition_fn, which internally calls the
        # `_recursive_joint_graph_passes` for the subgraph. So, skip recursing
        # skip_invoke_subgraph.
        for subgraph_name in _get_subgraph_names(gm, skip_invoke_subgraph):
            subgraph = getattr(gm, subgraph_name)
            _recursive_joint_graph_passes(subgraph, skip_invoke_subgraph)
        joint_graph_passes(gm)


def _recursive_post_grad_passes(gm: GraphModule, is_inference: bool = False) -> None:
    with dynamo_timed(
        "_recursive_post_grad_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="post_grad_pass_time_us",
    ):
        for subgraph_name in _get_subgraph_names(gm):
            subgraph = getattr(gm, subgraph_name)
            _recursive_post_grad_passes(subgraph, is_inference)
        post_grad_passes(gm, is_inference)


def split_const_gm(
    gm: GraphModule,
    skip_constructor: bool = True,
    lifted_constant_names: Optional[list[str]] = None,
    skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> tuple[GraphModule, dict[str, int]]:
    """
    This function takes an GraphModule input "gm".
    The gm will be split into 2 components,
      1) const_gm, which consists the subgraph of gm that can be constant folded.
      2) gm (being inplace modified,) which returns the graph after constant folding.

    If an additional "lifted_constants" argument is passed in, we will assume the gm has
    been lifted and run the transformation accordingly.

    When a "skip_folding_node_fn" callback is passed, we will skip constant folding on
    the nodes for which the callback returns True.

    const_output_index is a mapping of corresponding node name from gm to the
    output index of const_gm.
    Returns (const_gm, const_output_index)
    """
    from torch._inductor.constant_folding import (
        CONST_MODULE_TAG,
        META_TAG,
        MODULE_TAG,
        replace_node_with_constant,
        run_and_get_constant_graph,
    )

    const_gm = run_and_get_constant_graph(
        gm, skip_constructor, lifted_constant_names, skip_folding_node_fn
    )
    const_result = const_gm() if lifted_constant_names is None else None

    const_outputs = {
        x.name: idx for idx, x in enumerate(tuple(const_gm.graph.nodes)[-1].args[0])
    }

    to_erase_node = []
    to_replace_node = []
    const_output_index = {}
    for node in gm.graph.nodes:
        if node.name in const_outputs:
            to_replace_node.append(node)
        elif node.meta[META_TAG] == CONST_MODULE_TAG and node.op != "placeholder":
            to_erase_node.append(node)

    for node in to_replace_node:
        new_const_name = "_FOLDED_CONST_" + node.name
        replace_node_with_constant(
            gm,
            node,
            (
                const_result[const_outputs[node.name]]  # type:ignore[index]
                if lifted_constant_names is None
                else None
            ),
            new_const_name,
        )
        const_output_index[new_const_name] = const_outputs[node.name]
    for node in to_erase_node[::-1]:
        if node.users:
            for n in node.users:
                assert n.meta[META_TAG] == MODULE_TAG, f"node: {node} user not empty."
        else:
            gm.graph.erase_node(node)
    gm.recompile()

    return const_gm, const_output_index


def is_tf32_warning_applicable(gm: GraphModule) -> bool:
    aten = torch.ops.aten
    tf32_ops = OrderedSet(
        [
            aten.mm.default,
            aten.addmm.default,
            aten.bmm.default,
            aten.baddbmm.default,
        ]
    )
    for target in tf32_ops:
        for node in gm.graph.find_nodes(op="call_function", target=target):
            if (
                isinstance(node.meta.get("val", None), torch.Tensor)
                and node.meta["val"].dtype == torch.float32
                and node.meta["val"].device.type == "cuda"
            ):
                return True
    return False


def maybe_disable_comprehensive_padding(
    example_inputs: Sequence[InputType],
) -> AbstractContextManager[None, None]:
    """
    For CPU backend, enable comprehensive padding causes some unit tests
    fail due to changing number of generated kernels. Skip for now.
    """
    has_gpu = any(
        is_gpu(t.device.type) for t in example_inputs if isinstance(t, torch.Tensor)
    )

    if config.disable_padding_cpu and config.comprehensive_padding and not has_gpu:
        perf_hint_log.info("Skip comprehensive padding on CPU")
        return config.patch(comprehensive_padding=False)
    elif config.aot_inductor.use_runtime_constant_folding:
        perf_hint_log.info(
            "Skip comprehensive padding for use_runtime_constant_folding"
        )
        return config.patch(comprehensive_padding=False)
    else:
        return contextlib.nullcontext()


def maybe_disable_graph_partition(
    cpp_wrapper: bool, aot_mode: bool
) -> AbstractContextManager[None, None]:
    """
    graph partition does not support cpp_wrapper and aot_mode yet.
    """
    if cpp_wrapper or aot_mode:
        return config.patch(graph_partition=False)
    else:
        return contextlib.nullcontext()


def fake_tensor_prop(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    force_allow_non_fake_inputs: bool = False,
) -> torch._subclasses.FakeTensorMode:
    """
    If we can not detect fake mode from the context of inputs, create one.

    The created fake mode will be returned.
    """
    # Ensure that decomps that support symbolic shapes are used
    with enable_python_dispatcher():
        fake_mode = detect_fake_mode(example_inputs)
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
            FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
        else:
            ctx = (
                contextlib.nullcontext()
                if not force_allow_non_fake_inputs
                else mock.patch.object(fake_mode, "allow_non_fake_inputs", True)
            )
            with ctx:  # type: ignore[attr-defined]
                FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
                    *example_inputs
                )

    return fake_mode


# pass config dict back to user
def get_patched_config_dict(
    config_patches: Optional[Union[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    with config.patch(config_patches):
        return config.get_config_copy()


@contextlib.contextmanager
def with_fresh_cache_if_config() -> Generator[None, None, None]:
    if config.force_disable_caches:
        # Don't delete the cache dir because it has to survive beyond the
        # compile_fx call. Let's put the temp dirs under the default cache
        # dir so they're easier to locate.
        with fresh_cache(dir=cache_dir(), delete=False):
            yield
    else:
        yield


class _CompileFxKwargs(TypedDict, total=False):
    cudagraphs: Optional[BoxedBool]
    static_input_idxs: Sequence[int]
    is_backward: bool
    graph_id: Optional[int]
    cpp_wrapper: bool
    aot_mode: bool
    is_inference: bool
    layout_opt: Optional[bool]
    extern_node_serializer: Optional[Callable[[list[ExternKernelNode]], Any]]
    boxed_forward_device_index: Optional[BoxedDeviceIndex]
    fx_wrapper: bool


class _CompileFxCallable(Protocol):
    def __call__(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        **kwargs: Unpack[_CompileFxKwargs],
    ) -> OutputCode: ...


def compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    **kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode:
    kwargs.setdefault("cudagraphs", None)
    kwargs.setdefault("static_input_idxs", ())
    kwargs.setdefault("is_backward", False)
    kwargs.setdefault("graph_id", None)
    kwargs.setdefault("cpp_wrapper", False)
    kwargs.setdefault("fx_wrapper", False)
    kwargs.setdefault("is_inference", False)
    kwargs.setdefault("boxed_forward_device_index", None)
    kwargs.setdefault("layout_opt", None)
    kwargs.setdefault("extern_node_serializer", None)

    # Need with_fresh_cache_if_config for compile_fx_inner even if we already have one for
    # compile_fx. The reason is the compilation for backward graph may happen after
    # compile_fx return and we may want to use the _LazyGraphModule for compiling
    # the backward graph as well.
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.utils._python_dispatch._disable_current_modes())
        stack.enter_context(_use_lazy_graph_module(dynamo_config.use_lazy_graph_module))
        stack.enter_context(
            dynamo_utils.dynamo_timed(
                "compile_fx_inner",
                phase_name="inductor_compile",
                log_pt2_compile_event=True,
                log_waitcounter=True,
                waitcounter_name_override="compile_inductor",
                dynamo_compile_column_us="inductor_cumulative_compile_time_us",
            )
        )
        stack.enter_context(with_fresh_cache_if_config())
        stack.enter_context(DebugContext())
        CompileEventLogger.pt2_compile(
            "inductor_compile",
            is_backward=kwargs["is_backward"],
        )
        return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
            gm,
            example_inputs,
            **kwargs,
        )


@time_and_log(attr="compilation time (in seconds)")
def _compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    **graph_kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode:
    """
    Inductor API that compiles a single graph.

    If you change the argument list for this function, make sure you
    also update the call to save_args_for_compile_fx_inner below accordingly.
    """
    aot_mode: bool = V.aot_compilation

    # Clean up Compiled Triton Kernels per inductor compile, as the future objects
    # may not be valid for use after they are run/autotuned
    torch._inductor.async_compile.CompiledTritonKernels.cache_clear()

    if dynamo_utils.count_calls(gm.graph) == 0 and not aot_mode:
        # trigger the real recompilation for _LazyGraphModule before returning
        # the forward method.
        from torch._dynamo.utils import CompileEventLogLevel
        from torch.fx._lazy_graph_module import _LazyGraphModule

        _LazyGraphModule.force_recompile(gm)
        compile_id = torch._guards.CompileContext.current_compile_id()
        CompileEventLogger.log_instant_event(
            "backward no-op",
            metadata={"compile_id": compile_id},
            log_level=CompileEventLogLevel.PT2_COMPILE,
        )

        return make_boxed_func(gm.forward)

    static_input_idxs: Sequence[int] = graph_kwargs.setdefault("static_input_idxs", ())
    static_inputs_log.debug("static input idxs compile_fx_inner: %s", static_input_idxs)
    inputs_to_check = get_input_idxs_to_check(example_inputs, static_input_idxs)

    assert isinstance(next(iter(reversed(gm.graph.nodes))).args[0], (tuple, list)), (
        f"inductor can only compile FX graphs which return a tuple/list, but got {gm.graph}"
    )

    if graph_kwargs.get("cudagraphs") is None:
        graph_kwargs["cudagraphs"] = BoxedBool(config.triton.cudagraphs)
    if config.save_args:
        save_args_for_compile_fx_inner(
            gm,
            example_inputs,
            **graph_kwargs,
        )

    start = time.time()

    fx_graph_remote_cache = should_use_remote_fx_graph_cache()

    # Check if the registered backend(s) support caching.
    init_backend_registration()
    backends_support_caching = all(
        backend.supports_caching
        for backend in (
            get_wrapper_codegen_for_device(
                device.type, config.cpp_wrapper, config.fx_wrapper
            )
            for device in get_all_devices(gm)
        )
        if backend is not None
    )

    with dynamo_timed(
        "fx_codegen_and_compile", log_pt2_compile_event=True, log_waitcounter=True
    ):
        use_cache = (
            not config.force_disable_caches
            and (config.fx_graph_cache or fx_graph_remote_cache)
            and not aot_mode
            and backends_support_caching
            and not torch._functorch.config.bundled_autograd_cache
        )
        local = config.fx_graph_cache
        remote = fx_graph_remote_cache
        set_feature_use("fx_cache", use_cache)

        log.debug(
            "FX cache status: use_cache=%s, local=%s, remote=%s, aot_mode=%s, force_disable_caches=%s",
            use_cache,
            local,
            remote,
            aot_mode,
            config.force_disable_caches,
        )

        # TODO: This is a hack purely to get some info to extract_tensor_metadata_for_cache_key,
        # figure out how to not have to modify example inputs
        for i, input in enumerate(example_inputs):
            if (
                isinstance(input, torch.Tensor)
                and is_gpu(input.device.type)
                and i in static_input_idxs
            ):
                input._is_inductor_static = True  # type: ignore[attr-defined]

        mb_compiled_graph: Optional[OutputCode] = None
        key_info = None
        cache_info = None
        remote_cache = None
        constants = CompiledFxGraphConstantsWithGm(gm)
        # TODO: this time will be slightly inconsistent with the one computed
        # in prepare_key/load_with_key, dump those settings of "cache_event_time"
        start_time = time.time_ns()

        if use_cache:
            (key_info, cache_info) = FxGraphCache.prepare_key(
                gm, example_inputs, graph_kwargs, inputs_to_check, remote
            )

            # Attempt a cache lookup
            if key_info is not None:
                key, debug_lines = key_info
                log.debug("FX cache key generated: %s", key)
                if remote:
                    remote_cache = FxGraphCache.get_remote_cache()
                    log.debug("Using remote FX cache")
                mb_compiled_graph, cache_info = FxGraphCache.load_with_key(
                    key,
                    debug_lines,
                    example_inputs,
                    local,
                    remote_cache,
                    is_backward=graph_kwargs.get("is_backward", False),
                    constants=constants,
                )
            else:
                log.debug("Failed to generate FX cache key")

        if torch._functorch.config.bundled_autograd_cache:
            assert mb_compiled_graph is None
            assert cache_info is None
            # When using bundled autograd cache, we still want
            # to use the TritonBundler, but we don't want to save
            # the results here. The results will get saved directly
            # to AOTAutogradCache.
            TritonBundler.begin_compile()
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm, example_inputs, inputs_to_check, **graph_kwargs
                )
                assert mb_compiled_graph is not None
                (
                    triton_bundle,
                    triton_bundler_meta,
                ) = TritonBundler.collect()
                mb_compiled_graph.set_triton_bundle(triton_bundle)
            except (ShortenTraceback, SkipFrame):
                raise
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None
            finally:
                TritonBundler.end_compile()

        # CACHE BYPASS: Compile the graph, don't save it to the cache
        # (this can happen either because cache was disabled, or we
        # determined the input is uncacheable)
        elif cache_info is None or cache_info["cache_state"] == "bypass":
            assert mb_compiled_graph is None
            log.debug(
                "FX cache bypass reason: %s",
                (
                    cache_info.get("cache_bypass_reason", "unknown")
                    if cache_info is not None
                    else "FX cache disabled or key generation failed"
                ),
            )
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm, example_inputs, inputs_to_check, **graph_kwargs
                )
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None

        # CACHE MISS: Compile the graph and save to cache
        elif cache_info["cache_state"] == "miss":
            assert mb_compiled_graph is None
            assert key_info is not None
            log.debug("FX cache miss, compiling and saving to cache")
            TritonBundler.begin_compile()
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm, example_inputs, inputs_to_check, **graph_kwargs
                )
                assert mb_compiled_graph is not None
                mb_compiled_graph._time_taken_ns = time.time_ns() - start_time
                cache_key, debug_lines = key_info
                mb_compiled_graph._fx_graph_cache_key = cache_key
                mb_compiled_graph._fx_graph_cache_debug_lines = debug_lines
                (
                    triton_bundle,
                    triton_bundler_meta,
                ) = TritonBundler.collect()
                mb_compiled_graph.set_triton_bundle(triton_bundle)
            except (ShortenTraceback, SkipFrame):
                raise
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None
            finally:
                TritonBundler.end_compile()
            if triton_bundler_meta is not None:
                cache_info["triton_bundler_meta"] = str(triton_bundler_meta)
            cache_info["time_taken_ns"] = mb_compiled_graph._time_taken_ns
            log.debug("Saving compiled graph to FX cache with key: %s", cache_key)
            FxGraphCache._save_graph(
                cache_key,
                mb_compiled_graph,
                example_inputs,
                local,
                remote_cache,
            )

        # CACHE HIT: not much to really do, just make sure the cache key
        # is recorded on the graph
        else:
            assert cache_info["cache_state"] == "hit"
            assert mb_compiled_graph is not None
            assert key_info is not None
            (cache_key, debug_lines) = key_info
            log.debug("FX cache hit with key: %s", cache_key)
            mb_compiled_graph._fx_graph_cache_key = cache_key
            mb_compiled_graph._fx_graph_cache_debug_lines = debug_lines

        assert mb_compiled_graph is not None
        compiled_graph = mb_compiled_graph

        # Logging and observability: we log a single chromium event
        # and a tlparse log for every cache action.
        # In the event of a bypass, we also logged to the remote table earlier
        # with log_cache_bypass.
        cache_state = (
            cache_info["cache_state"] if cache_info is not None else "disabled"
        )
        # Here for grepping:
        # fx_graph_cache_hit
        # fx_graph_cache_miss
        # fx_graph_cache_bypass
        # fx_graph_cache_disabled
        CompileEventLogger.instant(
            f"fx_graph_cache_{cache_state}",
            metadata=cache_info or {},
            time_ns=start_time,
        )
        # Add event data about cache hits/miss
        # TODO: add remote cache get/put timings here too
        CompileEventLogger.pt2_compile(
            "inductor_compile",
            cache_state=cache_state,
            cache_event_time=start_time,
            key=cache_info.get("key") if cache_info else None,
            components=cache_info.get("components") if cache_info else None,
            cache_bypass_reason=(
                cache_info.get("cache_bypass_reason")
                if cache_info
                else "cache not enabled"
            ),
            remote_cache_enabled=remote,
            local_cache_enabled=local,
        )

        # Don't clog up the main tlparse output with disabled cache
        if cache_info is not None:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": f"fx_graph_cache_{cache_state}",
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )
        compiled_graph.post_compile(example_inputs, constants, graph_kwargs)

    log.debug("FX codegen and compilation took %.3fs", time.time() - start)

    # This message is for printing overview information of inductor mm counts, shapes,etc after lowering
    if log.isEnabledFor(logging.INFO):
        mm_table_data = []
        for key, value in counters["aten_mm_info"].items():
            parts = key.split("_")
            if len(parts) < 3:
                # Unexpected format, show as-is
                mm_table_data.append([key, "-", "?", "?", "?", value])
                continue

            # Determine if this is a batched operation by checking the operation name
            name = "_".join(parts[:-4]) if len(parts) >= 4 else "_".join(parts[:-3])
            is_batched = name.endswith(("bmm", "baddbmm"))

            if is_batched and len(parts) >= 4:
                # Batched operation: last 4 parts are batch, m, n, k
                batch, m, n, k = parts[-4:]
                name = "_".join(parts[:-4])
                mm_table_data.append([name, batch, m, n, k, value])
            else:
                # Non-batched operation: last 3 parts are m, n, k
                m, n, k = parts[-3:]
                name = "_".join(parts[:-3])
                mm_table_data.append([name, "-", m, n, k, value])

        log.info("Overview info of inductor aten mms: ")
        log.info(
            "{:<30} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(  # noqa: G001
                "Name", "B", "M", "N", "K", "Count"
            )
        )
        log.info("-" * 130)
        for row in mm_table_data:
            # pyrefly: ignore [not-iterable]
            log.info("{:<30} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(*row))  # noqa: G001
            log.info("-" * 130)

    # Not strictly necessary, but good to clean up straggling futures
    # that are unused to reclaim memory.
    torch._inductor.async_compile.CompiledTritonKernels.cache_clear()

    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if graph_kwargs['is_backward'] else 'FORWARDS'} "
        f"graph {graph_kwargs['graph_id']}",
    )
    return compiled_graph


class _FxCompileStat:
    # Count of successful compiles of this type
    codegen_and_compile: int = 0

    def __repr__(self) -> str:
        return f"codegen_and_compile: {self.codegen_and_compile}"


class FxCompile(ABC):
    """
    An FxCompile represents a mechanism that can turn a GraphModule into an
    OutputCode.
    """

    # Some stats for logging/debugging
    _compile_stats: dict[type[FxCompile], _FxCompileStat] = defaultdict(_FxCompileStat)

    # TODO: We should probably eventually add some kind of async version of this
    # so we can kick off a compile and then go do other things - but we'll need
    # to know what kind of API we want for that first.
    @abstractmethod
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...

    @classmethod
    def _reset_stats(cls) -> None:
        cls._compile_stats.clear()


class _InProcessFxCompile(FxCompile):
    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        """
        Generates the OutputCode from the GraphModule and example_inputs.
        """
        # Sorry about the mess, we need graph_kwargs to continue to be able
        # to propagate it further on
        # TODO: _CompileFxKwargs actually has stronger types than in the
        # signature, need to tighten it up

        assert "cudagraphs" in graph_kwargs and graph_kwargs["cudagraphs"] is not None
        cudagraphs: BoxedBool = graph_kwargs["cudagraphs"]
        static_input_idxs: Sequence[int] = graph_kwargs.get("static_input_idxs", ())
        is_backward: bool = graph_kwargs.get("is_backward", False)
        graph_id: Optional[int] = graph_kwargs.get("graph_id", None)
        cpp_wrapper: bool = graph_kwargs.get("cpp_wrapper", False)
        fx_wrapper: bool = graph_kwargs.get("fx_wrapper", False)
        aot_mode: bool = V.aot_compilation
        is_inference: bool = graph_kwargs.get("is_inference", False)
        extern_node_serializer: Optional[Callable[[list[ExternKernelNode]], Any]] = (
            graph_kwargs.get("extern_node_serializer", None)
        )

        with (
            _WaitCounter("pytorch.wait_counter.actual_codegen_and_compile").guard(),
            dynamo_utils.preserve_rng_state(),
        ):
            if (sleep_sec := config.sleep_sec_TESTING_ONLY) is not None:
                import time

                log.warning(
                    "Sleeping for %s since sleep_sec_TESTING_ONLY is set", sleep_sec
                )
                time.sleep(sleep_sec)

            if is_tf32_warning_applicable(gm):
                _warn_tf32_disabled()

            inductor_counters = counters["inductor"].copy()

            # lift the maximum depth of the Python interpreter stack
            # to adapt large/deep models
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

            _step_logger()(
                logging.INFO,
                "torchinductor compiling "
                f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
                f"graph {graph_id}",
            )

            fd = io.StringIO()
            torch._dynamo.repro.after_aot.save_graph_repro(
                fd, gm, example_inputs, "inductor", save_dir=None
            )
            runnable_graph_str = fd.getvalue()

            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_runnable",
                    "encoding": "string",
                },
                payload_fn=lambda: runnable_graph_str,
            )

            V.debug.fx_graph(gm, example_inputs)
            # TODO: Should we actually dump this?  It should be redundant with the aot
            # structured logs...
            # trace_structured("inductor_input_graph", payload_fn=lambda: gm.print_readable(print_output=False))

            shape_env = gm.shape_env
            if shape_env is None:
                shape_env = shape_env_from_inputs(example_inputs)

            # Convert view to reshape in the graph. This is necessary primarily for
            # layout optimization. Do it unconditionally for uniformity.
            #
            # It's needed because when we do layout optimization, an contiguous tensor
            # in eager mode may becomes a channels last tensor. A view op previously
            # can be applied to the contiguous tensor may not be able to be applied
            # on the channels tensor any more. An error like
            #   RuntimeError: view size is not compatible with input tensor's size and stride
            #   (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            # will be printed.
            #
            # Replace view op to reshape op in this case.
            # As an example, timm_resnest/botnet26t_256/convnext_base etc. will fail if we don't do this.
            #
            # Also this has to be done before FakeTensorProp below to avoid the failed
            # .view() call.
            view_to_reshape(gm)

            with dynamo_timed(
                "additional_fake_tensor_prop", log_pt2_compile_event=True
            ):
                # It is safe to run FakeTensorProp under no_grad because by the time
                # we're in inductor, we assume that AOTAutograd has already "taken care"
                # of autograd, so there should be no more autograd-related API's in the
                # graph.
                with torch.no_grad():
                    fake_mode = fake_tensor_prop(gm, example_inputs)

            _recursive_record_original_output_strides(gm)

            # pattern matcher passes might not preserve striding information
            # on node.meta["val"]. if in the future we rely on these being
            # correct we will need to fix.
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "before_post_grad_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: gm.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            with V.set_fake_mode(fake_mode):
                # has some issues with memory in training
                cuda_context = get_cuda_device_context(gm)
                with cuda_context:
                    _recursive_post_grad_passes(gm, is_inference=is_inference)
                V.debug.fx_graph_transformed(gm, example_inputs)
                post_grad_graphs_log.debug(
                    "%s",
                    lazy_format_graph_code(
                        "AFTER POST GRAD",
                        gm,
                        include_stride=True,
                        include_device=True,
                        colored=True,
                    ),
                )

                # We're printing the graph to be used as a cache key - so a
                # printer which is a little less readable but faster is
                # appropriate.
                inductor_post_grad_graph_str = gm.print_readable(
                    print_output=False,
                    include_stride=True,
                    include_device=True,
                    fast_sympy_print=True,
                )
                # "inductor_post_grad_graph" is used in inductor provenance
                # tracking highlighter front-end.
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "inductor_post_grad_graph",
                        "encoding": "string",
                    },
                    payload_fn=lambda: inductor_post_grad_graph_str,
                )
                if config.trace.provenance_tracking_level != 0:
                    provenance_tracking_json = (
                        torch.fx.traceback.get_graph_provenance_json(gm.graph)
                    )
                    torch._inductor.debug._inductor_post_to_pre_grad_nodes = (
                        create_mapping_pre_post_grad_nodes(
                            torch._inductor.debug._pre_grad_graph_id,
                            provenance_tracking_json,
                        )
                    )

                metrics_context = get_metrics_context()
                if metrics_context.in_progress():
                    num_graph_breaks = counters["graph_break"].total()
                    CompileEventLogger.compilation_metric(
                        overwrite=True, num_graph_breaks=num_graph_breaks
                    )
                if config.is_fbcode():
                    try:
                        log_optimus_to_scuba(
                            extra_logging={
                                "pt2_configs": str(get_patched_config_dict())
                            }
                        )
                    except Exception:
                        # TODO(T216453900): need to work around for now to support vllm
                        # See details in vllm/compilation/pass_manager.py.
                        log.warning("failed to log pt2_configs")

            with (
                V.set_fake_mode(fake_mode),
                maybe_disable_comprehensive_padding(example_inputs),
                maybe_disable_graph_partition(cpp_wrapper, aot_mode),
            ):
                const_output_index = None
                const_graph = None
                const_wrapper_code = None
                const_kernel_code = None

                if aot_mode and config.aot_inductor.use_runtime_constant_folding:
                    # torchbind objects have name that starts with _torchbind_obj
                    # See caffe2/torch/fx/_symbolic_trace.py?lines=406
                    const_gm, const_output_index = split_const_gm(
                        gm,
                        skip_folding_node_fn=lambda node: node.op == "get_attr"
                        and isinstance(node.target, str)
                        and (
                            node.target.startswith("_torchbind_obj")
                            or isinstance(node.meta.get("val", None), FakeScriptObject)
                        ),
                    )

                    const_graph = GraphLowering(
                        const_gm,
                        example_inputs=[],
                        shape_env=shape_env,
                        graph_id=graph_id,
                        cpp_wrapper=cpp_wrapper,
                        aot_mode=aot_mode,
                        extern_node_serializer=extern_node_serializer,
                        is_inference=is_inference,
                        is_backward=is_backward,
                        is_const_graph=True,
                        fx_wrapper=fx_wrapper,
                    )
                    with (
                        V.set_graph_handler(const_graph),
                        V.set_extern_kernel_nodes([]),
                    ):
                        assert cpp_wrapper, "AOT mode only supports C++ wrapper"
                        const_graph.run()
                        const_wrapper_code, const_kernel_code = (
                            const_graph.codegen_with_cpp_wrapper()
                        )

                graph = GraphLowering(
                    gm,
                    # example_inputs will be used by AOTInductor to dry-run the generated code for Triton kernel tuning.
                    # For the forward pass, we have the real inputs to be used as example_inputs. For the backward pass,
                    # we currently use fake tensors and defake them later.
                    example_inputs=example_inputs,
                    shape_env=shape_env,
                    graph_id=graph_id,
                    cpp_wrapper=cpp_wrapper,
                    aot_mode=aot_mode,
                    extern_node_serializer=extern_node_serializer,
                    is_inference=is_inference,
                    is_backward=is_backward,
                    const_output_index=const_output_index,
                    const_wrapper_code=(
                        const_wrapper_code.value if const_wrapper_code else None
                    ),
                    const_kernel_code=(
                        const_kernel_code.value if const_kernel_code else None
                    ),
                    const_module=const_graph,
                    inputs_to_check=inputs_to_check,
                    fx_wrapper=fx_wrapper,
                )
                metrics_helper = metrics.CachedMetricsHelper()

                # We are going to start code generating runtime asserts, so make sure
                # you don't start adding new ones in the lowering process
                graph.freeze_runtime_asserts()
                with (
                    V.set_graph_handler(graph),
                    V.set_extern_kernel_nodes([]),
                    distributed_autotune.graph_context(),
                ):
                    graph.run(*example_inputs)
                    output_strides: list[Optional[tuple[_StrideExprStr, ...]]] = []
                    if graph.graph_outputs is not None:
                        # We'll put the output strides in the compiled graph so we
                        # can later return them to the caller via TracingContext
                        p = SymExprPrinter()
                        for out in graph.graph_outputs:
                            if (
                                isinstance(out, IRNode)
                                and out.has_tensor_output()
                                and len(free_unbacked_symbols(out.get_stride())) == 0
                            ):
                                # Convert to string for eval on the load path
                                output_strides.append(
                                    tuple(p.doprint(s) for s in out.get_layout().stride)
                                )
                            else:
                                output_strides.append(None)

                    _check_triton_bf16_support(graph)

                    # TODO: The switching between AOT mode and not here is a bit
                    # messy, but it's localized to the block of code below so I'm
                    # not going to touch it for now

                    compiled_fn: Any
                    compiled_fn_runner = None
                    with dynamo_timed(
                        "GraphLowering.compile_to_fn", log_pt2_compile_event=True
                    ):
                        if graph.aot_mode and graph.fx_wrapper:
                            assert not graph.cpp_wrapper
                            compiled_fn = graph.codegen()[0].gm  # type: ignore[attr-defined]
                            output_code_log.debug(
                                "Output graph module: \n%s",
                                compiled_fn.print_readable(print_output=False),
                            )

                        elif graph.aot_mode:
                            from .codecache import AotCodeCompiler

                            assert graph.cpp_wrapper, (
                                "AOT mode only supports C++ wrapper"
                            )
                            wrapper_code, kernel_code = graph.codegen_with_cpp_wrapper()
                            output_code_log.debug(
                                "Output wrapper code: \n%s", wrapper_code.value
                            )
                            if kernel_code.value:
                                output_code_log.debug(
                                    "Output kernel code:\n%s", kernel_code.value
                                )

                            serialized_extern_kernel_nodes = None
                            if V.extern_kernel_nodes:
                                serialized_extern_kernel_nodes = (
                                    graph.extern_node_serializer(V.extern_kernel_nodes)
                                )
                                output_code_log.debug(
                                    "Serialized Extern Kernel Nodes: \n%s",
                                    serialized_extern_kernel_nodes,
                                )

                            with dynamo_timed(
                                "AotCodeCompiler.compile", log_pt2_compile_event=True
                            ):
                                # Directly return the file path with the compiled code
                                compiled_fn = AotCodeCompiler.compile(
                                    graph,
                                    wrapper_code.value,
                                    kernel_code.value,
                                    serialized_extern_kernel_nodes,
                                    device_type=graph.device_type,
                                    additional_files=[
                                        *dict.fromkeys(
                                            graph.wrapper_code.additional_files
                                            + (
                                                const_graph.wrapper_code.additional_files
                                                if const_graph
                                                else []
                                            )
                                        )
                                    ],
                                )
                        else:
                            compiled_module = graph.compile_to_module()
                            compiled_fn = compiled_module.call
                            compiled_fn_runner = getattr(
                                compiled_module, "runner", None
                            )

                    # Dump provenance artifacts for debugging trace
                    inductor_provenance_tracking_node_mappings = None
                    inductor_kernel_stack_trace_str = None
                    if config.trace.provenance_tracking_level != 0:
                        inductor_provenance_tracking_node_mappings = json.dumps(
                            torch._inductor.debug.dump_inductor_provenance_info()
                        )
                        inductor_kernel_stack_trace_str = json.dumps(
                            torch._inductor.debug._inductor_kernel_stack_trace
                        )
                        trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "inductor_provenance_tracking_node_mappings",
                                "encoding": "json",
                            },
                            payload_fn=lambda: inductor_provenance_tracking_node_mappings,
                        )
                        trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "inductor_provenance_tracking_kernel_stack_traces",
                                "encoding": "json",
                            },
                            payload_fn=lambda: inductor_kernel_stack_trace_str,
                        )
                        if inductor_kernel_stack_trace_str:
                            metrics_context = get_metrics_context()
                            if metrics_context.in_progress():
                                metrics_context.add_to_set(
                                    "inductor_provenance",
                                    inductor_kernel_stack_trace_str,
                                )

                    node_runtimes = None
                    if inductor_metrics_log.isEnabledFor(logging.INFO):
                        num_bytes, nodes_num_elem, node_runtimes = graph.count_bytes()
                        # pyrefly: ignore [bad-assignment]
                        metrics.num_bytes_accessed += num_bytes
                        metrics.node_runtimes += node_runtimes
                        metrics.nodes_num_elem += nodes_num_elem
                        inductor_metrics_log.info(
                            "Graph Metrics:\n%s",
                            {
                                "num_bytes_accessed": num_bytes,
                                "nodes_num_elem": nodes_num_elem,
                                "node_runtimes": node_runtimes,
                            },
                        )

                    # Collect and dump op runtimes and tensor metadata for TLParse
                    if config.log_tlparse:
                        _, _, node_runtimes = graph.count_bytes()
                        torch._inductor.debug.log_runtime_and_tensor_meta(node_runtimes)

                    # Collect and dump collective-op schedule for external diagnostics
                    torch._inductor.debug.log_collective_schedule(graph.scheduler.nodes)

                    if (
                        cudagraphs
                        and config.triton.cudagraph_skip_dynamic_graphs
                        and not V.graph.disable_cudagraphs_reason
                        and torch._inductor.utils.any_is_symbolic(*example_inputs)
                    ):
                        stack_trace = None
                        for node in gm.graph.nodes:
                            meta_val = node.meta.get("val", None)
                            if (
                                node.op == "placeholder"
                                or not isinstance(meta_val, torch.Tensor)
                                or not torch._inductor.utils.any_is_symbolic(meta_val)
                            ):
                                continue

                            if stack_trace := node.meta.get("stack_trace", None):
                                break
                        disable = "graph with symbolic shapes inputs and config.triton.cudagraph_skip_dynamic_graphs=True."
                        if stack_trace:
                            disable = f"{disable} Found from {stack_trace}\n"
                        else:
                            disable = f"{disable}\n"
                        # pyrefly: ignore [unbound-name]
                        V.graph.disable_cudagraphs_reason = disable

                    # pyrefly: ignore [unbound-name]
                    if cudagraphs and not V.graph.disable_cudagraphs_reason:
                        maybe_incompat_node = get_first_incompatible_cudagraph_node(gm)
                        if maybe_incompat_node:
                            disable = f"disabling cudagraphs due to incompatible op {maybe_incompat_node.target}"
                            if stack_trace := maybe_incompat_node.meta.get(
                                "stack_trace", None
                            ):
                                disable = f"{disable} Found from {stack_trace}\n"
                            # pyrefly: ignore [unbound-name]
                            V.graph.disable_cudagraphs_reason = disable

                    # pyrefly: ignore [unbound-name]
                    if V.aot_compilation:
                        assert isinstance(
                            compiled_fn,
                            # pyrefly: ignore [unbound-name]
                            (str, list, torch.fx.GraphModule),
                        ), type(compiled_fn)
                        return CompiledAOTI(compiled_fn)

                    # TODO: Hoist this above V.aot_compilation
                    # pyrefly: ignore [unbound-name]
                    if cudagraphs and not V.graph.disable_cudagraphs_reason:
                        from torch._inductor.cudagraph_utils import (
                            check_lowering_disable_cudagraph,
                        )

                        # pyrefly: ignore [unbound-name]
                        V.graph.disable_cudagraphs_reason = (
                            check_lowering_disable_cudagraph(
                                # pyrefly: ignore [unbound-name]
                                V.graph.device_node_mapping
                            )
                        )

                    self._compile_stats[type(self)].codegen_and_compile += 1

                    if (
                        # pyrefly: ignore [unbound-name]
                        torch._inductor.debug.RECORD_GRAPH_EXECUTION
                        # pyrefly: ignore [unbound-name]
                        and torch._inductor.debug.GRAPH_COMPILE_IDS is not None
                    ):
                        compile_id = str(
                            # pyrefly: ignore [unbound-name]
                            torch._guards.CompileContext.current_compile_id()
                        )
                        graph_id = graph_kwargs.get("graph_id")
                        if graph_id is not None:
                            # pyrefly: ignore [unbound-name]
                            torch._inductor.debug.GRAPH_COMPILE_IDS[graph_id] = (
                                compile_id
                            )

                    return CompiledFxGraph(
                        # pyrefly: ignore [bad-argument-type]
                        compiled_fn,
                        graph,
                        gm,
                        output_strides,
                        # pyrefly: ignore [unbound-name]
                        V.graph.disable_cudagraphs_reason,
                        metrics_helper.get_deltas(),
                        counters["inductor"] - inductor_counters,
                        cudagraphs,
                        example_inputs,
                        static_input_idxs,
                        graph_kwargs,
                        inputs_to_check,
                        runnable_graph_str,
                        inductor_post_grad_graph_str,
                        compiled_fn_runner,
                        inductor_provenance_tracking_node_mappings,
                        inductor_kernel_stack_trace_str,
                    )


def fx_codegen_and_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    # This is derivable from the other inputs to this function, but we pass it
    # in explicitly because it's nontrivial to compute
    inputs_to_check: Sequence[int],
    **graph_kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode:
    scheme: FxCompile

    if fx_compile_mode == FxCompileMode.NORMAL:
        scheme = _InProcessFxCompile()
    elif fx_compile_mode == FxCompileMode.SERIALIZE:
        from .compile_fx_ext import _DebugSerdeFxCompile

        scheme = _DebugSerdeFxCompile()
    elif fx_compile_mode == FxCompileMode.SUBPROCESS:
        from .compile_fx_subproc import _SubprocessFxCompile

        scheme = _SubprocessFxCompile()

    if fx_compile_async:
        from .compile_fx_async import _AsyncFxCompile
        from .compile_fx_ext import _OutOfProcessFxCompile

        # pyrefly: ignore [unbound-name]
        assert isinstance(scheme, _OutOfProcessFxCompile), (
            "async is only valid with an out-of-process compile mode"
        )
        # pyrefly: ignore [unbound-name]
        scheme = _AsyncFxCompile(scheme)

    if fx_compile_progressive:
        from .compile_fx_async import _ProgressiveFxCompile
        from .compile_fx_ext import _OutOfProcessFxCompile

        # pyrefly: ignore [unbound-name]
        assert isinstance(scheme, _OutOfProcessFxCompile), (
            "progressive is only valid with an out-of-process compile mode"
        )

        progression_configs = _get_progression_configs()

        # Use in-process compile for the fast version
        fast_scheme = _InProcessFxCompile()

        # pyrefly: ignore [unbound-name]
        scheme = _ProgressiveFxCompile(fast_scheme, scheme, progression_configs)

    # pyrefly: ignore [unbound-name]
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)


def get_input_idxs_to_check(
    inputs: Sequence[InputType],
    static_input_idxs: Sequence[int],
) -> Sequence[int]:
    """
    This function runs at compile time, and generates a list of indices for which we
    might need to do a copy to preserve alignment requirements.
    """
    ids_to_check = []

    for i, input in enumerate(inputs):
        if not isinstance(input, torch.Tensor):
            # non-tensors don't need alignment
            continue
        if not is_gpu(input.device.type):
            # right now we only care for gpu tensors
            continue
        with maybe_get_suppress_shape_guards_ctx():
            # suppress guards so that tensor_is_aligned and should_assume_input_aligned
            # do not add guards on input's storage offset
            if i in static_input_idxs and tensor_is_aligned(input):
                continue
            if not should_assume_input_aligned(input):
                continue

        # if we get here, then
        # (a) our triton code assumes that the input is aligned
        # (b) we can't be sure ahead of time that the input will actually be aligned.
        # therefore, at runtime, we'll need to check that the input is aligned
        # (and if not, clone it to make it aligned.)
        ids_to_check.append(i)

    return ids_to_check


def cudagraphify(
    model: Callable[..., Any],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    stack_traces: list[Optional[str]],
    is_backward: bool,
    is_inference: bool,
    constants: tuple[torch.Tensor, ...] = (),
    placeholders: Sequence[PlaceholderInfo] = (),
    mutated_input_idxs: tuple[int, ...] = (),
) -> Callable[..., Any]:
    from torch._inductor.cudagraph_trees import (
        cudagraphify_impl as new_cudagraphify_impl,
    )

    cudagraphify_fn: Callable[..., Any]
    if config.triton.cudagraph_trees:
        cudagraphify_fn = functools.partial(
            new_cudagraphify_impl,
            device_index=device_index,
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=constants,
            placeholders=placeholders,
            mutated_input_idxs=mutated_input_idxs,
            compile_id=torch._guards.CompileContext.current_compile_id(),
        )
    else:
        cudagraphify_fn = cudagraphify_impl

    compiled_fn = None

    def run(new_inputs: Sequence[InputType]) -> Any:
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)  # type: ignore[arg-type]
        return compiled_fn(new_inputs)  # type: ignore[arg-type]

    return run


def static_input(x: torch.Tensor) -> torch.Tensor:
    """
    Copy and input while preserving strides
    """
    return torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)


def index_expanded_dims_and_copy_(
    dst: torch.Tensor,
    src: torch.Tensor,
    expanded_dims: list[int],
) -> None:
    "Index into expanded dimensions of both dst and src then copy_"
    dst = index_expanded_dims(dst, expanded_dims)
    src = index_expanded_dims(src, expanded_dims)
    dst.copy_(src)


def cudagraphify_impl(
    model: Callable[..., Any],
    inputs: list[torch.Tensor],
    static_input_idxs: Sequence[int] = (),
) -> Callable[[list[InputType]], Any]:
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)  # type: ignore[arg-type]
    # pyrefly: ignore [annotation-mismatch]
    static_input_idxs: OrderedSet[int] = OrderedSet(
        remove_unaligned_input_idxs(inputs, static_input_idxs)  # type: ignore[arg-type]
    )
    copy_misaligned_inputs(inputs, check_input_idxs)  # type: ignore[arg-type]

    assert isinstance(inputs, list)

    inps_expanded_dims = [
        get_expanded_dims(x) if idx not in static_input_idxs else []
        for idx, x in enumerate(inputs)
    ]

    # allocate static tensor inputs
    static_inputs = [
        (
            x
            if not isinstance(x, torch.Tensor)
            else static_input(x)
            if idx not in static_input_idxs
            else x.detach()
        )
        for idx, x in enumerate(inputs)
    ]

    # copy over input values for fresh allocations
    for idx, (x, expanded_dims) in enumerate(zip(inputs, inps_expanded_dims)):
        if isinstance(x, torch.Tensor) and idx not in static_input_idxs:
            index_expanded_dims_and_copy_(static_inputs[idx], x, expanded_dims)

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # copy static_inputs because it will be cleared in model
    with torch.cuda.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        static_outputs = model(list(static_inputs))
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    if config.size_asserts:

        def run(new_inputs: list[InputType]) -> Callable[[list[InputType]], Any]:
            assert len(static_inputs) == len(new_inputs)
            for idx, (dst, src, expanded_dims) in enumerate(
                zip(static_inputs, new_inputs, inps_expanded_dims)
            ):
                if not isinstance(dst, torch.Tensor):
                    continue
                assert isinstance(src, torch.Tensor)
                if idx in static_input_idxs:
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    # TODO - could make one single op of multiple slices
                    # and avoid dispatch.
                    # Could also pre-index the `dst` tensors
                    index_expanded_dims_and_copy_(dst, src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            # pyrefly: ignore [bad-return]
            return static_outputs

    else:
        copy_indices = [
            idx for idx in range(len(static_inputs)) if idx not in static_input_idxs
        ]

        def run(new_inputs: list[InputType]) -> Callable[[list[InputType]], Any]:
            for idx in copy_indices:
                expanded_dims = inps_expanded_dims[idx]
                src = new_inputs[idx]
                assert isinstance(src, torch.Tensor)
                index_expanded_dims_and_copy_(static_inputs[idx], src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            # pyrefly: ignore [bad-return]
            return static_outputs

    return align_inputs_from_check_idxs(run, check_input_idxs, OrderedSet())


def compile_fx_aot(
    model_: GraphModule,
    example_inputs_: list[InputType],
    inner_compile: _CompileFxCallable = compile_fx_inner,
    config_patches: Optional[dict[str, Any]] = None,
) -> Union[list[Union[str, Weights]], str, GraphModule]:
    assert isinstance(model_, GraphModule), model_

    # [See NOTE] Unwrapping subclasses AOT
    unwrap_tensor_subclass_parameters(model_)

    # pyrefly: ignore [annotation-mismatch]
    config_patches: dict[str, Any] = copy.deepcopy(config_patches or {})

    if not (config_patches.get("fx_wrapper", False) or config.fx_wrapper):
        # If fx_wrapper is not set, then set cpp_wrapper
        config_patches["cpp_wrapper"] = True

    output_path = config_patches.get(
        "aot_inductor.output_path", config.aot_inductor.output_path
    )

    if output_path:
        assert not output_path.endswith(".pt2"), (
            "The output path for aot_compile should not have an extension with .pt2 "
            "this is for specifying the output path for the .so in AOTInductor. "
            "If you would like to package the AOTInductor generated files "
            "into a pt2, please call `torch._inductor.aoti_compile_and_package`."
        )
    else:
        config_patches = {
            **config_patches,
            "aot_inductor.output_path": code_hash(model_.code),
        }

    from .utils import maybe_aoti_standalone_config

    config_patches = maybe_aoti_standalone_config(config_patches)

    extern_node_serializer = config_patches.pop("extern_node_serializer", None)
    saved_compile_id = model_.meta.get("dynamo_compile_id", None)
    saved_compile_context = torch._guards.CompileContext(saved_compile_id)
    with (
        V.set_aot_compilation(True),
        torch._guards.compile_context(saved_compile_context),
        chromium_event_timed(
            "compile_fx_aot",
            log_pt2_compile_event=True,
            reset_event_log_on_exit=True,
        ),
        get_metrics_context(),
    ):
        compiled_artifacts = compile_fx(
            model_,
            example_inputs_,
            inner_compile=functools.partial(
                inner_compile,
                extern_node_serializer=extern_node_serializer,
            ),
            config_patches=config_patches,
        )

        assert isinstance(compiled_artifacts, CompiledAOTI)

        return compiled_artifacts.filename


_graph_counter = count(0)


def fw_compiler_freezing(
    aot_autograd_model: GraphModule,
    aot_example_inputs: Sequence[InputType],
    dynamo_model: GraphModule,
    num_example_inputs: int,
    inner_compile: Callable[..., Any],
    cudagraphs: BoxedBool,
    graph_id: int,
    forward_device: BoxedDeviceIndex,
) -> Callable[[list[object]], Sequence[torch.Tensor]]:
    from torch._inductor.freezing import convert_conv_weights_to_channels_last, freeze

    # partition_fn won't be called
    _recursive_joint_graph_passes(aot_autograd_model)

    layout_opt = GraphLowering.decide_layout_opt(aot_autograd_model, is_inference=True)
    if layout_opt:
        # make sure meta['val'] is properly setup
        fake_tensor_prop(aot_autograd_model, aot_example_inputs, True)
        convert_conv_weights_to_channels_last(aot_autograd_model)

    opt_model, preserved_arg_indices = freeze(
        dynamo_model,
        aot_autograd_model,
        aot_example_inputs,  # type: ignore[arg-type]
    )

    aot_example_inputs = [aot_example_inputs[ind] for ind in preserved_arg_indices]

    fake_mode = detect_fake_mode(aot_example_inputs)

    # for freezing, all graph outputs should be user visible
    *_, model_outputs_node = opt_model.graph.nodes
    model_outputs = model_outputs_node.args[0]
    model_outputs_node.meta["user_visible_output_idxs"] = [
        idx for idx, n in enumerate(model_outputs) if isinstance(n, torch.fx.Node)
    ]

    static_input_idxs: list[Any] = []
    # constant params will be real tensors, not fake
    tracing_context = torch._guards.TracingContext.try_get()
    unwrapped_args_offsets = [0]
    max_offset_idx = 0
    if tracing_context is not None:
        assert tracing_context.params_flat_unwrap_subclasses is not None
        params_flat_unwrap = tracing_context.params_flat_unwrap_subclasses
        max_offset_idx = max(0, len(params_flat_unwrap) - 1)
        preserved_indices_params_flat = OrderedSet[int]()
        unwrapped_idxs = tracing_context.params_unwrapped_to_flat_index
        assert unwrapped_idxs is not None
        current_offset = 0
        if len(params_flat_unwrap) > 0:
            unwrapped_args_offsets = []

        for i in range(len(params_flat_unwrap)):
            if i not in preserved_arg_indices:
                params_flat_unwrap[i] = None
                if i > 0 and unwrapped_idxs[i] == unwrapped_idxs[i - 1]:
                    current_offset += 1
            else:
                preserved_indices_params_flat.add(unwrapped_idxs[i])
            unwrapped_args_offsets.append(current_offset)

        # Deallocate wrapped params, if all subelements were deallocated
        assert tracing_context.params_flat is not None
        for i in range(len(tracing_context.params_flat)):
            if i not in preserved_indices_params_flat:
                tracing_context.params_flat[i] = None

        if tracing_context.fw_metadata:
            static_input_idxs = tracing_context.fw_metadata.static_input_indices

    with mock.patch.object(fake_mode, "allow_non_fake_inputs", True):
        optimized_function = inner_compile(
            opt_model,
            aot_example_inputs,
            static_input_idxs=static_input_idxs,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
            is_inference=True,
            boxed_forward_device_index=forward_device,
            layout_opt=layout_opt,
        )

    # aot_inductor codegens a call that takes in just the inputs, so we don't return a wrapper
    # that drops constant-ified params
    if V.aot_compilation:
        return optimized_function

    def wrapper(args: list[object]) -> Sequence[torch.Tensor]:
        args_new = [
            args[i - unwrapped_args_offsets[min(i, max_offset_idx)]]
            for i in preserved_arg_indices
        ]
        args.clear()
        return optimized_function(args_new)

    wrapper._boxed_call = True  # type: ignore[attr-defined]

    return wrapper


def get_cpp_wrapper_config() -> dict[str, object]:
    if config.triton.cudagraphs:
        log_cudagraph_skip_and_bump_counter(
            format_default_skip_message("cpp wrapper enabled")
        )

    return {
        # Set autotune_at_compile_time to True as default if the option is not explicitly set
        "triton.autotune_at_compile_time": (
            config.triton.autotune_at_compile_time
            if config.triton.autotune_at_compile_time is not None
            else has_triton()
        ),
        "triton.autotune_cublasLt": False,
        "triton.cudagraphs": False,  # TODO: to be removed
        "triton.store_cubin": True,
    }


def get_cuda_device_context(gm: torch.fx.GraphModule) -> AbstractContextManager[None]:
    """
    Returns a cuda device context manager if there is a single device in the graph
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()

    cuda_devices: OrderedSet[torch.device] = OrderedSet(
        device for device in get_all_devices(gm) if device.type == "cuda"
    )

    return (
        torch.cuda.device(next(iter(cuda_devices)))  # type: ignore[return-value]
        if len(cuda_devices) == 1
        else contextlib.nullcontext()
    )


def partition_fn(
    gm: GraphModule,
    joint_inputs: Sequence[object],
    **kwargs: object,
) -> tuple[GraphModule, GraphModule]:
    cuda_context = get_cuda_device_context(gm)
    with cuda_context:
        # We can skip the invoke_subgraph because the
        # entire_partition_fn is called recursively for invoke_subgraph
        # in partitioning.
        _recursive_joint_graph_passes(gm, skip_invoke_subgraph=True)

    static_lifetime_input_indices: Optional[list[int]] = kwargs.pop(  # type: ignore[assignment]
        "static_lifetime_input_indices", None
    )

    if config.custom_partitioner_fn is None:
        with dynamo_utils.dynamo_timed(
            "min_cut_rematerialization_partition", log_pt2_compile_event=True
        ):
            return min_cut_rematerialization_partition(
                gm,
                joint_inputs,
                compiler="inductor",
                static_lifetime_input_indices=static_lifetime_input_indices,
                **kwargs,
            )
    else:
        assert isinstance(config.custom_partitioner_fn, CustomPartitionerFn)
        with dynamo_utils.dynamo_timed(
            config.custom_partitioner_fn.__class__.__name__,
            log_pt2_compile_event=True,
        ):
            return config.custom_partitioner_fn(
                gm,
                joint_inputs,
                compiler="inductor",
                static_lifetime_input_indices=static_lifetime_input_indices,
                **kwargs,
            )


def get_num_model_outputs(model: GraphModule) -> int:
    model_outputs_node = output_node(model)
    model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
    return len(model_outputs)


@dataclass(frozen=True)
class CompilerConfigExtra:
    cudagraphs: BoxedBool
    graph_id: int
    forward_device: BoxedDeviceIndex


def create_compiler_config_extra(config: types.ModuleType) -> CompilerConfigExtra:
    # Although cudagraphs may have been enabled via config, various
    # conditions (which are tested within the bowels of Inductor) may
    # force cudagraphs to be disabled.  This mutable box lets us retrieve
    # the final determination if cudagraphs actually can be used or not.
    cudagraphs = BoxedBool(config.triton.cudagraphs)

    # TODO: The modern style is to use CompileId from TracingContext to
    # identify Inductor compilation.  However, this CompileId cannot
    # uniquely identify multiple Inductor compilations that arise from
    # DDPOptimizer
    graph_id = next(_graph_counter)

    # See [Backward Generation Handling]
    forward_device = BoxedDeviceIndex(None)

    return CompilerConfigExtra(
        cudagraphs=cudagraphs,
        graph_id=graph_id,
        forward_device=forward_device,
    )


def compile_fx_forward(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    num_orig_model_outputs: int,
    num_example_inputs: int,
    compiler_config_extra: CompilerConfigExtra,
    inner_compile: Callable[..., OutputCode] = compile_fx_inner,
    is_inference: bool = False,
) -> OutputCode:
    """
    Compile the forward graph of the given graph module.

    Args:
        gm: The graph module to compile.
        example_inputs: The example inputs to use for compilation.
        num_orig_model_outputs: The number of model outputs from the original dynamo graph.
        num_example_inputs: The number of example inputs from the original dynamo graph.
        compiler_config_extra: Extra configuration for the compiler.
        inner_compile: The inner compile function to use.
        is_inference: Whether this is an inference graph.
    """

    if is_inference:
        # partition_fn won't be called
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "before_joint_graph",
                "encoding": "string",
            },
            payload_fn=lambda: gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )

        _recursive_joint_graph_passes(gm)

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "after_joint_graph",
                "encoding": "string",
            },
            payload_fn=lambda: gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )

    fixed = torch._inductor.utils.num_fw_fixed_arguments(
        num_example_inputs, len(example_inputs)
    )

    model_outputs_node = output_node(gm)
    if config.keep_output_stride:
        model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
        num_model_outputs = len(model_outputs)

        context = torch._guards.TracingContext.try_get()
        # See Note [User Outputs in the inductor graph]
        if context is not None and context.fw_metadata and not is_inference:
            original_output_start_index = (
                context.fw_metadata.num_mutated_inp_runtime_indices
            )
        else:
            original_output_start_index = 0

        assert num_orig_model_outputs <= num_model_outputs

        # Note [User Outputs in the inductor graph]
        # We makes the following assumption
        # For inference
        #   len(orig_model_outputs) == len(model_outputs)
        # For training
        #   len(orig_model_outputs) <= len(model_outputs)
        # During training, most of the time the model_outputs starts with
        # original module's outputs followed by saved activations.
        # But this can be not true if the model have inplace updated tensors.
        # AOTAutograd will make those tensors being returned before the original
        # module's output.
        # To make things safe, we'll use original_output_start_index field
        # set by AOTAutograd to decide where the original module outputs start.
        orig_output_end_idx = original_output_start_index + num_orig_model_outputs
        # Sanity check: we are about to splice out the "user" outputs from the full set
        # of "graph" outputs. Make sure we're within bounds.
        assert orig_output_end_idx <= num_model_outputs

        model_outputs_node.meta["user_visible_output_idxs"] = [
            idx
            for idx in range(original_output_start_index, orig_output_end_idx)
            if isinstance(model_outputs[idx], torch.fx.Node)
        ]
    else:
        model_outputs_node.meta["user_visible_output_idxs"] = []

    # We also mark the invoke_subgraph outputs as user_visible to
    # force the outputs of invoke_subgraph subgraph to follow the
    # original strides
    _recursive_record_user_visible_output_idxs(gm)

    return inner_compile(
        gm,
        example_inputs,
        static_input_idxs=get_static_input_idxs(fixed),
        cudagraphs=compiler_config_extra.cudagraphs,
        graph_id=compiler_config_extra.graph_id,
        is_inference=is_inference,
        boxed_forward_device_index=compiler_config_extra.forward_device,
    )


def compile_fx_backward(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    compiler_config_extra: CompilerConfigExtra,
    inner_compile: Callable[..., OutputCode] = compile_fx_inner,
) -> OutputCode:
    """
    Compile the backward graph of the given graph module.

    Args:
        gm: The graph module to compile.
        example_inputs: The example inputs to use for compilation.
        compiler_config_extra: Extra configuration for the compiler.
        inner_compile: The inner compile function to use.
    """
    from torch._dynamo.convert_frame import compile_lock

    with compile_lock:
        model_outputs_node = output_node(gm)
        if config.bw_outputs_user_visible:
            model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
            model_outputs_node.meta["user_visible_output_idxs"] = [
                idx
                for idx, n in enumerate(model_outputs)
                if isinstance(n, torch.fx.Node)
            ]
        else:
            model_outputs_node.meta["user_visible_output_idxs"] = []

        fixed = count_tangents(gm)
        with (
            config.patch(get_cpp_wrapper_config())
            if config.cpp_wrapper
            else contextlib.nullcontext()
        ):
            return inner_compile(
                gm,
                example_inputs,
                static_input_idxs=list(range(fixed)),
                cudagraphs=compiler_config_extra.cudagraphs,
                is_backward=True,
                graph_id=compiler_config_extra.graph_id,
                boxed_forward_device_index=compiler_config_extra.forward_device,
            )


def run_pre_grad_passes(
    model_: GraphModule, example_inputs_: Sequence[InputType]
) -> GraphModule:
    # "before_pre_grad_graph" is used in inductor provenance
    # tracking highlighter front-end.
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "before_pre_grad_graph",
            "encoding": "string",
        },
        payload_fn=lambda: model_.print_readable(
            print_output=False, include_stride=True, include_device=True
        )
        + f"\n\n # graph id: {id(model_.graph)}",
    )
    pre_grad_graphs_log.debug(
        "%s",
        lazy_format_graph_code(
            "BEFORE PRE GRAD",
            model_,
            include_stride=True,
            include_device=True,
            colored=True,
        ),
    )
    torch._inductor.debug._pre_grad_graph_id = id(model_.graph)

    if config.trace.provenance_tracking_level == 1:
        for node in model_.graph.nodes:
            if node.stack_trace:
                torch._inductor.debug._inductor_pre_grad_node_stack_trace[node.name] = (
                    node.stack_trace
                )

    model_ = _recursive_pre_grad_passes(model_, example_inputs_)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "after_pre_grad_graph",
            "encoding": "string",
        },
        payload_fn=lambda: model_.print_readable(
            print_output=False, include_stride=True, include_device=True
        )
        + f"\n\n # graph id: {id(model_.graph)}",
    )
    return model_


def compile_fx(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile: Callable[..., OutputCode] = compile_fx_inner,
    config_patches: Optional[dict[str, Any]] = None,
    decompositions: Optional[dict[OpOverload, Callable[..., Any]]] = None,
    ignore_shape_env: bool = False,
) -> CompileFxOutput:
    """
    Main entry point for compiling given FX graph.  Despite the fact that this
    lives in :mod:`torch._inductor`, this function is responsible for calling
    into AOT Autograd (and we will eventually get a callback to
    ``inner_compile`` to perform actual compilation.  In other words, this
    function orchestrates end-to-end compilation for the inductor backend when
    you use :func:`torch.compile`.

    NB: This function TAKES OWNERSHIP of the input ``model_`` and can potentially
    mutate it!  Make a copy if you need to preserve the original GraphModule.
    """
    # Some arguments trigger a recursive call to compile_fx.  Handle these
    # short circuits first, before anything else

    from torch._inductor.compiler_bisector import CompilerBisector

    if CompilerBisector.disable_subsystem("inductor", "pre_grad_graph"):
        return model_

    if config_patches:
        with config.patch(config_patches):
            return compile_fx(
                model_,
                example_inputs_,
                # need extra layer of patching as backwards is compiled out of scope
                inner_compile=config.patch(config_patches)(inner_compile),
                decompositions=decompositions,
                ignore_shape_env=ignore_shape_env,
            )

    # Wake up the AsyncCompile subproc pool as early as possible (if there's cuda).
    if any(
        isinstance(e, torch.Tensor) and e.device.type in ("cuda", "xpu")
        for e in example_inputs_
    ):
        torch._inductor.async_compile.AsyncCompile.wakeup()

    if config.cpp_wrapper or config.fx_wrapper:
        from torch._export.non_strict_utils import _fakify_script_objects

        cpp_wrapper_config = config.cpp_wrapper
        fx_wrapper_config = config.fx_wrapper

        with (
            config.patch(get_cpp_wrapper_config()),
            V.set_real_inputs(example_inputs_),
        ):
            inputs_: Sequence[InputType] = (
                _extract_inputs_from_exported_gm(model_, example_inputs_)
                if isinstance(model_, GraphModule)
                else example_inputs_
            )
            fake_mode = detect_fake_mode(inputs_)
            with _fakify_script_objects(model_, inputs_, {}, fake_mode) as (
                patched_mod,
                fake_args,
                _,
                _,
                _,
            ):
                return _maybe_wrap_and_compile_fx_main(
                    patched_mod,
                    fake_args,
                    inner_compile=functools.partial(
                        inner_compile,
                        cpp_wrapper=cpp_wrapper_config,
                        fx_wrapper=fx_wrapper_config,
                    ),
                    decompositions=decompositions,
                    ignore_shape_env=ignore_shape_env,
                )

    return _maybe_wrap_and_compile_fx_main(
        model_,
        example_inputs_,
        inner_compile,
        decompositions,
        ignore_shape_env,
    )


def _extract_inputs_from_exported_gm(
    gm: GraphModule, example_inputs_: Sequence[InputType]
) -> Sequence[InputType]:
    fake_inputs = [
        node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
    ]
    # Replace non-tensor (constant) inputs with Nones, since these are not being
    # used anyways by the graph
    fake_inputs = [
        inp if isinstance(inp, torch.Tensor) else None for inp in fake_inputs
    ]

    if any(v is not None for v in fake_inputs):
        # Validate devices before switching to fake tensors.
        for idx, fi, i in zip(count(), fake_inputs, example_inputs_):
            if fi is not None:
                assert isinstance(i, torch.Tensor)
                if fi.device != i.device:
                    raise ValueError(
                        f"Device mismatch between fake input and example input at position #{idx}: "
                        f"{fi.device} vs {i.device}. If the model was exported via torch.export(), "
                        "make sure torch.export() and torch.aot_compile() run on the same device."
                    )
        return fake_inputs

    return example_inputs_


def _maybe_wrap_and_compile_fx_main(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile: Callable[..., OutputCode],
    decompositions: Optional[dict[OpOverload, Callable[..., Any]]],
    ignore_shape_env: bool,
) -> CompileFxOutput:
    """
    Part of compile_fx, called after patching configs.

    Ultimately we want to call _compile_fx_main, where the actual work happens.
    But under various conditions, various forms of wrapping might be needed
    around _compile_fx_main.
    """
    # Each wrapper below takes a self-contained compile_gm function which is
    # called inside the wrapper. This just recursively calls this function.
    compile_gm = functools.partial(
        _maybe_wrap_and_compile_fx_main,
        inner_compile=inner_compile,
        decompositions=decompositions,
        ignore_shape_env=ignore_shape_env,
    )
    if not graph_returns_tuple(model_):
        return make_graph_return_tuple(model_, example_inputs_, compile_gm)

    if isinstance(model_, GraphModule) and isinstance(
        model_.graph._codegen, _PyTreeCodeGen
    ):
        # this graph is the result of dynamo.export()
        return handle_dynamo_export_graph(model_, example_inputs_, compile_gm)

    if any(isinstance(x, (list, tuple, dict)) for x in example_inputs_):
        # NB: this short circuit never occurs for Dynamo produced graphs
        # (which are pre-flattened)
        return flatten_graph_inputs(model_, example_inputs_, compile_gm)

    # Finally do the actual work!
    return _compile_fx_main(
        model_,
        example_inputs_,
        inner_compile,
        decompositions,
        ignore_shape_env,
    )


def _compile_fx_main(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile: Callable[..., OutputCode],
    decompositions: Optional[dict[OpOverload, Callable[..., Any]]],
    ignore_shape_env: bool,
) -> CompileFxOutput:
    """
    Main part of compile_fx, called after wrapping is done.

    Roughly speaking, here the steps will be:
    (1) apply pre-grad passes
    (2) create `fw_compiler` and `bw_compiler` functions out of `inner_compile`
    (3) call aot_autograd, which:
    - (3a) creates a joint graph with `decompositions`,
    - (3b) partitions it with `partition_fn` into fw and bw graphs (applying joint-graph passes),
    - (3c) calls `fw_compiler` and `bw_compiler` on those graphs (applying post-grad passes)
    - (3d) finally, assembles the fw and bw compiled functions back together and returns.
    """
    with (
        _use_lazy_graph_module(dynamo_config.use_lazy_graph_module),
        enable_python_dispatcher(),
        torch.fx.traceback.preserve_node_meta(
            config.trace.provenance_tracking_level == 1
        ),
        torch._inductor.debug.reset_provenance_globals(),
    ):
        # Pre-grad passes cannot be run if we weren't given a GraphModule.
        # Dynamo will always produce a GraphModule, but this handles cases
        # where a user directly passes a plain Module with the intention of
        # having AOTAutograd trace it.
        # TODO: Get rid of this?
        if isinstance(model_, GraphModule):
            model_ = run_pre_grad_passes(model_, example_inputs_)

        assert not config._raise_error_for_testing

        num_example_inputs = len(example_inputs_)

        compiler_config_extra = create_compiler_config_extra(config)

        decompositions = (
            decompositions if decompositions is not None else select_decomp_table()
        )

        def fw_compiler_base(
            gm: GraphModule,
            example_inputs: Sequence[InputType],
            is_inference: bool,
        ) -> OutputCode:
            with dynamo_utils.dynamo_timed("compile_fx.<locals>.fw_compiler_base"):
                if isinstance(model_, GraphModule):
                    num_orig_model_outputs = get_num_model_outputs(model_)
                else:
                    num_orig_model_outputs = get_num_model_outputs(gm)
                return compile_fx_forward(
                    gm,
                    example_inputs,
                    num_orig_model_outputs=num_orig_model_outputs,
                    num_example_inputs=num_example_inputs,
                    compiler_config_extra=compiler_config_extra,
                    inner_compile=inner_compile,
                    is_inference=is_inference,
                )

        fw_compiler: Callable[[GraphModule, Sequence[InputType]], OutputCode] = (
            functools.partial(fw_compiler_base, is_inference=False)
        )
        fw_compiler = SerializableAOTDispatchCompiler(OutputCode, fw_compiler)

        if config.freezing and not torch.is_grad_enabled():
            inference_compiler: Callable[..., Any] = functools.partial(
                fw_compiler_freezing,
                dynamo_model=model_,
                num_example_inputs=num_example_inputs,
                inner_compile=inner_compile,
                cudagraphs=compiler_config_extra.cudagraphs,
                graph_id=compiler_config_extra.graph_id,
                forward_device=compiler_config_extra.forward_device,
            )
        else:
            inference_compiler = functools.partial(fw_compiler_base, is_inference=True)
            inference_compiler = SerializableAOTDispatchCompiler(
                OutputCode, inference_compiler
            )

        @compile_time_strobelight_meta(phase_name="backward")
        def bw_compiler(
            gm: GraphModule, example_inputs: Sequence[InputType]
        ) -> OutputCode:
            with (
                dynamo_utils.dynamo_timed("compile_fx.<locals>.bw_compiler"),
            ):
                return compile_fx_backward(
                    gm,
                    example_inputs,
                    compiler_config_extra=compiler_config_extra,
                    inner_compile=inner_compile,
                )

        bw_compiler = SerializableAOTDispatchCompiler(OutputCode, bw_compiler)

        fake_mode = detect_fake_mode(
            example_inputs_
        ) or torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        tracing_context = (
            torch._guards.TracingContext.try_get()
            or torch._guards.TracingContext(fake_mode)
        )

        if V.aot_compilation:
            from .utils import is_valid_aoti_model_name

            is_valid_aoti_model_name()

            with functorch_config.patch(unlift_effect_tokens=True):
                gm, graph_signature = aot_export_module(
                    model_,
                    example_inputs_,
                    trace_joint=False,
                    decompositions=decompositions,
                )

                from torch._export.utils import _detect_fake_mode_from_gm

                fake_mode = _detect_fake_mode_from_gm(gm)  # type: ignore[assignment]
                # aot_export_module doesn't account for constant tensor attributes
                # so we end up having tensors that don't have fake vals attached.
                # This can happen when upstream export is non-strict where we
                # preserve the original module params/buffers. Once AOTI switches
                # to ep.run_decompositions() flow to lower to post-autograd opset
                # this will go away.
                for node in gm.graph.nodes:
                    if node.op == "get_attr" and "val" not in node.meta:
                        target = attrgetter(node.target)(gm)
                        if isinstance(target, torch.Tensor):
                            assert fake_mode is not None
                            node.meta["val"] = fake_mode.from_tensor(
                                target, static_shapes=True
                            )
                        elif isinstance(target, torch.ScriptObject):
                            node.meta["val"] = (
                                torch._library.fake_class_registry.maybe_to_fake_obj(
                                    fake_mode, target
                                )
                            )
                        elif isinstance(target, FakeScriptObject):
                            node.meta["val"] = target

            unlifted_gm = _unlift_graph(model_, gm, graph_signature)
            if "dynamo_flat_name_to_original_fqn" in model_.meta:
                unlifted_gm.meta["dynamo_flat_name_to_original_fqn"] = model_.meta[
                    "dynamo_flat_name_to_original_fqn"
                ]

            if "dynamo_compile_id" in model_.meta:
                unlifted_gm.meta["dynamo_compile_id"] = model_.meta["dynamo_compile_id"]

            # Disable amp as in aot_dispatch_autograd (https://github.com/pytorch/pytorch/pull/86515)
            # In inference_compiler (fw_compiler_base), _recursive_joint_graph_passes will call into
            # _sfdp_init() to register patterns.
            # When fallback_random is set to True, the sdpa patterns will be traced during runtime.
            # If amp is turned on, the traced FP32 patterns will have prims.convert_element_type which
            # will be the same as the generated FP16 patterns.
            disable_amp = torch._C._is_any_autocast_enabled()
            context = (
                torch._C._DisableAutocast if disable_amp else contextlib.nullcontext
            )
            with V.set_fake_mode(fake_mode), compiled_autograd._disable(), context():
                return inference_compiler(unlifted_gm, example_inputs_)

        with (
            V.set_fake_mode(fake_mode),
            torch._guards.tracing(tracing_context),
            compiled_autograd._disable(),
            functorch_config.patch(unlift_effect_tokens=True),
        ):
            try:
                return aot_autograd(
                    fw_compiler=fw_compiler,
                    bw_compiler=bw_compiler,
                    inference_compiler=inference_compiler,
                    decompositions=decompositions,
                    partition_fn=partition_fn,
                    keep_inference_input_mutations=True,
                    cudagraphs=compiler_config_extra.cudagraphs,
                    boxed_forward_device_index=compiler_config_extra.forward_device,
                    ignore_shape_env=ignore_shape_env,
                )(model_, example_inputs_)
            except ShortenTraceback as e:
                # We will also shorten the traceback inside dynamo.
                # This is only useful if inductor is called directly with an FX graph.
                raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1


def graph_returns_tuple(gm: GraphModule) -> bool:
    """True if a FX graph returns a tuple"""
    if not isinstance(gm, GraphModule):
        return True  # can't check this, assume true
    (rv,) = output_node(gm).args
    if isinstance(rv, (list, tuple)):
        return True
    if (
        isinstance(rv, torch.fx.node.Node)
        and hasattr(rv.target, "_schema")
        and len(rv.target._schema.returns) > 1
        and all(str(ret.type) == "Tensor" for ret in rv.target._schema.returns)
    ):
        # for graphs whose result is one node with multiple outputs
        return True
    return False


def make_graph_return_tuple(
    gm: GraphModule,
    inputs: Sequence[InputType],
    compile_gm: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Mutate gm so it returns a tuple.  This is only needed for graphs
    not created by torchdynamo that return non-tuples.
    """
    node = output_node(gm)
    (rv,) = node.args
    rv, spec = pytree.tree_flatten(rv)
    with gm.graph.inserting_before(node):
        gm.graph.output(rv)
    gm.graph.erase_node(node)
    assert graph_returns_tuple(gm)

    compiled_fn = compile_gm(gm, inputs)

    @functools.wraps(compiled_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return pytree.tree_unflatten(compiled_fn(*args, **kwargs), spec)

    return wrapper


def handle_dynamo_export_graph(
    gm: GraphModule,
    inputs: Sequence[InputType],
    compile_gm: Callable[..., Any],
) -> Callable[..., Any]:
    """
    `torch._dynamo.export` embeds pytrees in the FX graph codegen object,
    convert that to a normal FX graph so inductor can compile it.
    """
    codegen = gm.graph._codegen
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.recompile()

    compiled_fn = compile_gm(gm, codegen.process_inputs(*inputs))

    @functools.wraps(compiled_fn)  # type: ignore[misc]
    def wrapper(*args: Any) -> Any:
        return codegen.process_outputs(compiled_fn(*codegen.process_inputs(*args)))

    return wrapper


def _check_triton_bf16_support(graph: GraphLowering) -> None:
    def warn_and_skip(device: Optional[torch.device]) -> Never:
        from torch._dynamo.exc import SkipFrame

        assert device is not None

        device_interface = get_interface_for_device(device.type)
        device_props = device_interface.get_device_properties(device)
        warnings.warn(
            f"{device_props.name} does not support bfloat16 compilation natively, skipping"
        )
        raise SkipFrame("BF16 is not supported")

    for node in itertools.chain(graph.graph_inputs.values(), graph.graph_outputs):
        if not isinstance(node, IRNode):
            continue
        device_type = get_device_type(node)
        if (
            not device_type
            or not is_gpu(device_type)
            or node.get_dtype() != torch.bfloat16
        ):
            continue
        # Print warning and skip frame if attempting to compile for bfloat16
        # on device without hardware support for dtype
        device_interface = get_interface_for_device(device_type)
        if device_interface.is_bf16_supported(including_emulation=False):
            return
        warn_and_skip(node.get_device())


def _aoti_flatten_inputs(
    gm: torch.fx.GraphModule,
    args: Union[list[Any], tuple[Any, ...]],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    options: Optional[dict[str, Any]] = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Flatten the inputs to the graph module and return the flat inputs and options.
    Add "aot_inductor.serialized_in_spec" and "aot_inductor.serialized_out_spec" to the options.
    """
    # pyrefly: ignore [missing-module-attribute]
    from .compile_fx import graph_returns_tuple

    assert graph_returns_tuple(gm), (
        "Graph output must be a tuple(). This is so that we can avoid "
        "pytree processing of the outputs. Please change the module to "
        "have tuple outputs."
    )

    # We will serialize the pytree info into the .so as constant strings
    in_spec = None
    out_spec = None
    if isinstance(gm.graph._codegen, torch.fx.graph._PyTreeCodeGen):
        codegen = gm.graph._codegen
        gm.graph._codegen = torch.fx.graph.CodeGen()
        gm.recompile()

        if codegen.pytree_info.in_spec is not None:
            in_spec = codegen.pytree_info.in_spec
        if codegen.pytree_info.out_spec is not None:
            out_spec = codegen.pytree_info.out_spec

    else:
        if hasattr(gm, "_in_spec"):
            in_spec = gm._in_spec
        if hasattr(gm, "_out_spec"):
            out_spec = gm._out_spec

    serialized_in_spec = pytree.treespec_dumps(in_spec) if in_spec is not None else ""
    serialized_out_spec = (
        pytree.treespec_dumps(out_spec) if out_spec is not None else ""
    )

    flat_args_with_path, received_spec = pytree.tree_flatten_with_path(
        (args, kwargs or {})
    )

    if any(isinstance(x[1], torch.ScriptObject) for x in flat_args_with_path):
        from torch._dynamo.exc import UserError, UserErrorType

        raise UserError(
            UserErrorType.INVALID_INPUT,
            "TorchBind objects found in inputs. TorchBind object inputs are not supported in AOTInductor. "
            "TorchBind objects can only be attributes.",
        )

    # Replace non-tensor (constant) inputs with Nones, since these are not being
    # used anyways by the graph
    flat_example_inputs = [
        x[1] if isinstance(x[1], torch.Tensor) else None for x in flat_args_with_path
    ]

    if in_spec is not None and received_spec != in_spec:
        raise ValueError(  # noqa: B904
            "Trying to flatten user inputs with exported input tree spec: \n"
            f"{in_spec}\n"
            "but actually got inputs with tree spec of: \n"
            f"{received_spec}"
        )

    options = (
        {
            "aot_inductor.serialized_in_spec": serialized_in_spec,
            "aot_inductor.serialized_out_spec": serialized_out_spec,
        }
        if options is None
        else {
            **options,
            "aot_inductor.serialized_in_spec": serialized_in_spec,
            "aot_inductor.serialized_out_spec": serialized_out_spec,
        }
    )
    return flat_example_inputs, options
