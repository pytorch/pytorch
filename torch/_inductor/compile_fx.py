import dataclasses
import functools
import itertools
import logging
import sys
import warnings
from typing import Any, Dict, List, Optional

import functorch
from functorch.compile import min_cut_rematerialization_partition

import torch._dynamo.config as dynamo_config

import torch.fx

from torch._dynamo import logging as dynamo_logging, utils as dynamo_utils
from torch._dynamo.utils import fake_mode_from_tensors
from torch._functorch.aot_autograd import make_boxed_func
from torch._subclasses.fake_tensor import FakeTensor
from .._dynamo.backends.common import aot_autograd
from . import config, metrics, overrides, pattern_matcher
from .debug import DebugContext
from .decomposition import select_decomp_table
from .graph import GraphLowering
from .mkldnn import convert_outplace_to_inplace
from .utils import developer_warning, get_dtype_size, has_incompatible_cudagraph_ops
from .virtualized import V

log = logging.getLogger(__name__)
ALIGNMENT = 16


@dataclasses.dataclass
class BoxedBool:
    value: bool

    def __bool__(self):
        return self.value

    @staticmethod
    def disable(obj):
        if isinstance(obj, BoxedBool):
            obj.value = False
            return obj
        return False


# copy_ fails when trying to write to tensors with memory overlap,
# for expanded dimensions (a dimension which used to have size 1 -> ?)
# we can select one element from that dimension and write to it
# to achieve writing to all values of that dimension of the input tensor
def get_expanded_dims(t):
    return [i for i in range(t.ndim) if t.stride(i) == 0 and t.size(i) != 1]


def index_expanded_dims(t, expanded_dims):
    for expanded_dim in expanded_dims:
        t = torch.ops.aten.slice(t, expanded_dim, 0, 1)
    return t


def complex_memory_overlap(t):
    # if torch._debug_has_internal_overlap thinks this tensor potentially has
    # memory overlap internally, let's dig deeper to find out whether it's true.
    if torch._debug_has_internal_overlap(t) != 0:
        strides = t.stride()
        sizes = t.shape
        indices = list(range(len(strides)))
        indices = [x for _, x in sorted(zip(strides, indices))]
        for i in range(len(strides)):
            prev_stride = 1 if i == 0 else strides[indices[i - 1]]
            prev_size = 1 if i == 0 else sizes[indices[i - 1]]
            if strides[indices[i]] < prev_stride * prev_size:
                return True
    return False


@functools.lru_cache(None)
def _step_logger():
    return dynamo_logging.get_step_logger(log)


@functools.lru_cache(None)
def _warn_tf32_disabled():
    if (
        torch.cuda.is_available()
        and not torch.backends.cuda.matmul.allow_tf32
        and torch.cuda.get_device_capability() >= (8, 0)
    ):
        warnings.warn(
            "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. "
            "Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
        )


def is_tf32_warning_applicable(gm: torch.fx.GraphModule):
    aten = torch.ops.aten
    tf32_ops = {
        aten.mm.default,
        aten.addmm.default,
        aten.bmm.default,
        aten.baddbmm.default,
    }
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target in tf32_ops
            and isinstance(node.meta.get("val", None), torch.Tensor)
            and node.meta["val"].dtype == torch.float32
            and node.meta["val"].device.type == "cuda"
        ):
            return True
    return False


@DebugContext.wrap
def count_bytes_inner(gm, example_inputs, num_fixed=0, **kwargs):
    shape_env = _shape_env_from_inputs(example_inputs)

    graph = GraphLowering(gm, shape_env=shape_env, num_static_inputs=num_fixed)
    with V.set_graph_handler(graph):
        graph.run(*example_inputs)
        num_bytes, nodes_num_elem = graph.count_bytes()
        metrics.num_bytes_accessed += num_bytes
        metrics.nodes_num_elem += nodes_num_elem
    return make_boxed_func(gm.forward)


@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs=None,
    num_fixed=0,
    is_backward=False,
    graph_id=None,
):
    if is_tf32_warning_applicable(gm):
        _warn_tf32_disabled()

    if dynamo_utils.count_calls(gm.graph) == 0:
        return make_boxed_func(gm.forward)

    # lift the maximum depth of the Python interpreter stack
    # to adapt large/deep models
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    _step_logger()(
        logging.INFO,
        "torchinductor compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )
    V.debug.fx_graph(gm, example_inputs)

    if cudagraphs is None:
        cudagraphs = config.triton.cudagraphs

    shape_env = _shape_env_from_inputs(example_inputs)
    fake_mode = fake_mode_from_tensors(
        example_inputs
    ) or torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)

    with V.set_fake_mode(fake_mode):
        pattern_matcher.fx_passes(gm)
        V.debug.fx_graph_transformed(gm, example_inputs)

        graph = GraphLowering(
            gm,
            shape_env=shape_env,
            num_static_inputs=num_fixed,
            graph_id=graph_id,
        )
        with V.set_graph_handler(graph):
            graph.run(*example_inputs)
            compiled_fn = graph.compile_to_fn()

    if cudagraphs:
        complex_memory_overlap_inputs = any(
            complex_memory_overlap(t) for t in example_inputs
        )

        if (
            set(graph.device_types) == {"cuda"}
            and not graph.mutated_inputs
            and not has_incompatible_cudagraph_ops(gm)
            and not complex_memory_overlap_inputs
        ):
            compiled_fn = cudagraphify(
                compiled_fn, example_inputs, static_input_idxs=range(num_fixed)
            )
        else:
            BoxedBool.disable(cudagraphs)

            if len(set(graph.device_types)) > 1:
                developer_warning("skipping cudagraphs due to multiple devices")
            elif set(graph.device_types) == {"cuda"}:
                if graph.mutated_inputs:
                    developer_warning("skipping cudagraphs due to input mutation")
                elif complex_memory_overlap_inputs:
                    developer_warning(
                        "skipping cudagraphs due to complex input striding"
                    )

    result = align_inputs(compiled_fn, example_inputs, range(num_fixed))
    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    # aot autograd needs to know to pass in inputs as a list
    result._boxed_call = True
    return result


def clone_preserve_strides(x):
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
    )
    buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
    return torch.as_strided(buffer, x.size(), x.stride())


def align_inputs(model, inputs, static_input_idxs=()):
    def is_aligned(storage_offset, dtype):
        return (storage_offset * get_dtype_size(dtype)) % ALIGNMENT == 0

    check_inputs = [
        i
        for i in range(len(inputs))
        if (
            i not in static_input_idxs
            or not is_aligned(inputs[i].storage_offset(), inputs[i].dtype)
        )
        and inputs[i].device.type == "cuda"
    ]

    if len(check_inputs) == 0:
        return model

    def run(new_inputs):
        for i in check_inputs:
            if new_inputs[i].data_ptr() % ALIGNMENT:
                new_inputs[i] = clone_preserve_strides(new_inputs[i])
        return model(new_inputs)

    return run


@dynamo_utils.dynamo_timed
def cudagraphify(model, inputs, static_input_idxs=()):
    # if using fake tensors, defer cudagraphs until we get real inputs at runtime
    if not any(isinstance(inp, FakeTensor) for inp in inputs):
        return cudagraphify_impl(model, inputs, static_input_idxs)

    compiled_fn = None

    def run(new_inputs):
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                compiled_fn = cudagraphify_impl(model, new_inputs, static_input_idxs)

        return compiled_fn(new_inputs)

    return run


def remove_unaligned_input_idxs(inputs, static_input_idxs):
    """
    We require all inputs to be aligned, so introduce a copy for any
    that aren't.
    """
    aligned_static_input_idxs = {
        idx for idx in static_input_idxs if (inputs[idx].data_ptr() % ALIGNMENT) == 0
    }
    if len(aligned_static_input_idxs) != len(static_input_idxs):
        return aligned_static_input_idxs
    return static_input_idxs


def cudagraphify_impl(model, inputs, static_input_idxs=()):
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)

    def static_input(x):
        """
        Copy and input while preserving strides
        """
        # TODO(jansel): figure out why this version doesn't work:
        # return torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
        needed_size = (
            sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
        )
        buffer = torch.zeros(needed_size, dtype=x.dtype, device=x.device)
        return torch.as_strided(buffer, x.size(), x.stride())

    assert isinstance(inputs, (list, tuple))
    static_inputs = [
        static_input(x) if idx not in static_input_idxs else x.detach()
        for idx, x in enumerate(inputs)
    ]

    inps_expanded_dims = [
        get_expanded_dims(x) if idx not in static_input_idxs else []
        for idx, x in enumerate(inputs)
    ]

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
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(list(static_inputs))
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    if config.size_asserts:

        def run(new_inputs):
            assert len(static_inputs) == len(new_inputs)
            for idx, (dst, src, expanded_dims) in enumerate(
                zip(static_inputs, new_inputs, inps_expanded_dims)
            ):
                if idx in static_input_idxs:
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    # TODO - could make one single op of multiple slices
                    # and avoid dispatch.
                    # Could also pre-index the `dst` tensors
                    dst = index_expanded_dims(dst, expanded_dims)
                    src = index_expanded_dims(src, expanded_dims)
                    dst.copy_(src)
            new_inputs.clear()
            graph.replay()
            return static_outputs

    else:
        copy_indices = [
            idx for idx in range(len(static_inputs)) if idx not in static_input_idxs
        ]

        def run(new_inputs):
            for idx in copy_indices:
                src = index_expanded_dims(static_inputs[idx], inps_expanded_dims[idx])
                dst = index_expanded_dims(new_inputs[idx], inps_expanded_dims[idx])
                dst.copy_(src)
            new_inputs.clear()
            graph.replay()
            return static_outputs

    return run


def count_tangents(fx_g: torch.fx.GraphModule):
    """
    Infers which inputs are static for a backwards graph
    """

    def is_not_gradout(x):
        return "tangents" not in x.name

    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            if is_not_gradout(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1

    assert static_arg_idxs == list(range(len(static_arg_idxs)))
    return len(static_arg_idxs)


_graph_counter = itertools.count(0)


def compile_fx(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    inner_compile=compile_fx_inner,
    config_patches: Optional[Dict[str, Any]] = None,
):
    """Main entrypoint to a compile given FX graph"""
    if config_patches:
        with config.patch(config_patches):
            return compile_fx(
                model_,
                example_inputs_,
                # need extra layer of patching as backwards is compiled out of scope
                inner_compile=config.patch(config_patches)(inner_compile),
            )

    assert not config._raise_error_for_testing

    functorch.compile.config.use_functionalize = True
    functorch.compile.config.use_fake_tensor = True

    with overrides.patch_functions():
        model_ = overrides.replace_fx(model_)
        model_ = overrides.fuse_fx(model_, example_inputs_)
    num_example_inputs = len(example_inputs_)
    cudagraphs = BoxedBool(
        config.triton.cudagraphs and not dynamo_config.dynamic_shapes
    )

    graph_id = next(_graph_counter)

    @dynamo_utils.dynamo_timed
    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = len(example_inputs) - num_example_inputs
        # Why convert outplace op to inplace? Inductor can support inplace operations well and for custom
        # inplace ops which are lowered as ExternKernel, it is beneficial to performance when the inplace
        # implementation is used if available.
        model = convert_outplace_to_inplace(model)
        return inner_compile(
            model,
            example_inputs,
            num_fixed=fixed,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
        )

    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = count_tangents(model)
        return inner_compile(
            model,
            example_inputs,
            num_fixed=fixed,
            cudagraphs=cudagraphs,
            is_backward=True,
            graph_id=graph_id,
        )

    with overrides.patch_functions():

        # TODO: can add logging before/after the call to create_aot_dispatcher_function
        # in torch._functorch/aot_autograd.py::aot_module_simplified::aot_function_simplified::new_func
        # once torchdynamo is merged into pytorch
        return aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
            keep_inference_input_mutations=True,
        )(model_, example_inputs_)


def _shape_env_from_inputs(inputs):
    shape_env = None
    fake_mode = fake_mode_from_tensors(inputs)

    # TODO(voz): It would be nice to enable this assert, but there are lots of tests that
    # pass in real inputs for now.
    # if len(inputs) > 0:
    # assert fake_mode is not None, breakpoint()

    if fake_mode is not None:
        return fake_mode.shape_env

    # TODO(voz): Should we always have one anyway?
    return None
