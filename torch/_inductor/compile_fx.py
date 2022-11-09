import dataclasses
import functools
import itertools
import logging
import sys
from typing import List

import functorch
from functorch.compile import min_cut_rematerialization_partition

import torch.fx
from torch._functorch.aot_autograd import make_boxed_func
from torch._subclasses.fake_tensor import FakeTensor

from . import config, overrides
from .debug import DebugContext
from .decomposition import select_decomp_table
from .graph import GraphLowering
from .utils import (
    dynamo_logging,
    dynamo_optimizations,
    dynamo_utils,
    has_incompatible_cudagraph_ops,
)
from .virtualized import V

log = logging.getLogger(__name__)
ALIGNMENT = 16

aot_autograd = dynamo_optimizations.backends.aot_autograd
normalize_ir = dynamo_optimizations.normalize.normalize_ir
is_aot_autograd_safe_to_run = dynamo_optimizations.training.is_aot_autograd_safe_to_run
count_calls = dynamo_utils.count_calls


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
    shape_env = None
    for inp in example_inputs:
        if isinstance(inp, FakeTensor) and inp.fake_mode.shape_env is not None:
            shape_env = inp.fake_mode.shape_env

    graph = GraphLowering(gm, shape_env=shape_env, num_static_inputs=num_fixed)
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
                log.warning("skipping cudagraphs due to multiple devices")
            elif set(graph.device_types) == {"cuda"}:
                if graph.mutated_inputs:
                    log.warning("skipping cudagraphs due to input mutation")
                elif complex_memory_overlap_inputs:
                    log.warning("skipping cudagraphs due to complex input striding")

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
    check_inputs = [
        i
        for i in range(len(inputs))
        if (i not in static_input_idxs or (inputs[i].data_ptr() % ALIGNMENT) != 0)
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


def compile_fx(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor]):
    """Main entrypoint to a compile given FX graph"""

    if not is_aot_autograd_safe_to_run(model_, example_inputs_):
        log.warning("Aot Autograd is not safe to run, so falling back to eager")
        return model_

    functorch.compile.config.use_functionalize = True
    functorch.compile.config.use_fake_tensor = True

    with overrides.patch_functions():
        model_ = normalize_ir(model_, example_inputs_)
        model_ = overrides.replace_fx(model_)
        model_ = overrides.fuse_fx(model_, example_inputs_)
    num_example_inputs = len(example_inputs_)
    cudagraphs = BoxedBool(config.triton.cudagraphs and not config.dynamic_shapes)

    graph_id = next(_graph_counter)

    @dynamo_utils.dynamo_timed
    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = len(example_inputs) - num_example_inputs
        return compile_fx_inner(
            model,
            example_inputs,
            num_fixed=fixed,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
        )

    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = count_tangents(model)
        return compile_fx_inner(
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
            model_,
            example_inputs_,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        )
