# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import itertools
import logging
import operator
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch._inductor as inductor
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp import register_decomposition
from torch._dynamo.utils import counters
from torch._inductor import comms
from torch._inductor.virtualized import ops
from torch._logging import trace_structured
from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch.utils._ordered_set import OrderedSet

from .. import config, ir, pattern_matcher
from ..codegen.common import custom_backend_passes
from ..comms import remove_fsdp2_unsharded_param_graph_input_usage
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage
from ..lowering import lowerings as L
from ..pattern_matcher import (
    _return_true,
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    filter_nodes,
    fwd_only,
    get_arg_value,
    get_mutation_region_id,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    ListOf,
    Match,
    MultiOutputPattern,
    MULTIPLE,
    PatternMatcherPass as PatternMatcherPassBase,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)
from ..utils import (
    decode_device,
    get_all_devices,
    get_gpu_type,
    has_uses_tagged_as,
    is_gpu,
    OPTIMUS_EXCLUDE_POST_GRAD,
)
from ..virtualized import V
from .b2b_gemm import B2B_GEMM_PASS
from .ddp_fusion import fuse_ddp_communication
from .group_batch_fusion import group_batch_fusion_passes, POST_GRAD_FUSIONS
from .micro_pipeline_tp import micro_pipeline_tp_pass
from .pre_grad import is_same_dict, save_inductor_dict
from .reinplace import reinplace_inplaceable_ops
from .split_cat import POST_GRAD_PATTERNS


_T = TypeVar("_T")
_P = ParamSpec("_P")

PatternMatcherPass = functools.partial(
    PatternMatcherPassBase, subsystem="post_grad_passes"
)

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]


def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="post_grad_passes",
    )

    if not torch._dynamo.config.skip_fsdp_hooks:
        remove_fsdp2_unsharded_param_graph_input_usage(gm.graph)

    if config.dce:
        # has some issues with mutation in inference mode
        gm.graph.eliminate_dead_code()

    if is_inference and config.reorder_for_locality:
        GraphTransformObserver(gm, "reorder_for_locality").apply_graph_pass(
            reorder_for_locality
        )

    fake_tensor_updater = FakeTensorUpdater(gm.graph)

    if post_grad_custom_pre_pass := config.post_grad_custom_pre_pass:
        GraphTransformObserver(gm, "post_grad_custom_pre_pass").apply_graph_pass(
            post_grad_custom_pre_pass
        )

    if torch._C._has_mkldnn:
        if (
            config.cpp.enable_grouped_gemm_template
            and config.max_autotune
            and "CPP" in config.max_autotune_gemm_backends
        ):
            from .mkldnn_fusion import grouped_gemm_pass

            grouped_gemm_pass(gm.graph)

        if config.cpp.enable_concat_linear:
            from .quantization import concat_linear_woq_int4

            # Concat linear optimization for WOQ int4
            concat_linear_woq_int4(gm)

    if config.pattern_matcher:
        lazy_init()
        GraphTransformObserver(gm, "post_grad_custom_pre_pass").apply_graph_pass(
            functools.partial(group_batch_fusion_passes, pre_grad=False)
        )
        GraphTransformObserver(gm, "remove_noop_ops").apply_graph_pass(remove_noop_ops)
        GraphTransformObserver(gm, "remove_assert_ops").apply_graph_pass(
            remove_assert_ops
        )
        for i, patterns in enumerate(pass_patterns):
            GraphTransformObserver(gm, f"pass_pattern_{i}").apply_graph_pass(
                patterns.apply
            )
        for pass_name in config.post_grad_fusion_options:
            # skip all patterns for group batch fusions or quantization patterns
            if pass_name in POST_GRAD_FUSIONS or pass_name in OPTIMUS_EXCLUDE_POST_GRAD:
                continue
            pattern_matcher_pass = POST_GRAD_PATTERNS[pass_name]
            inductor_before_change = save_inductor_dict(
                [pattern_matcher_pass.pass_name]
            )
            GraphTransformObserver(gm, pass_name).apply_graph_pass(
                pattern_matcher_pass.apply
            )
            if not is_same_dict(counters["inductor"], inductor_before_change):
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": f"{pattern_matcher_pass.pass_name}_post_grad",
                        "encoding": "string",
                    },
                    payload_fn=lambda: gm.print_readable(
                        print_output=False, include_stride=True, include_device=True
                    ),
                )
        if config.b2b_gemm_pass:
            B2B_GEMM_PASS.apply(gm.graph)  # type: ignore[arg-type]

    if config._micro_pipeline_tp:
        micro_pipeline_tp_pass(gm.graph)

    if config._fuse_ddp_communication:
        GraphTransformObserver(gm, "fuse_ddp_communication").apply_graph_pass(
            lambda graph: fuse_ddp_communication(
                graph,
                config._fuse_ddp_communication_passes,
                config._fuse_ddp_bucket_size,
            )
        )

    if post_grad_custom_post_pass := config.post_grad_custom_post_pass:
        GraphTransformObserver(gm, "post_grad_custom_post_pass").apply_graph_pass(
            post_grad_custom_post_pass
        )

    GraphTransformObserver(gm, "stable_sort").apply_graph_pass(stable_topological_sort)

    GraphTransformObserver(gm, "move_constructors_to_cuda").apply_graph_pass(
        move_constructors_to_gpu
    )

    fake_tensor_updater.incremental_update()

    for device, custom_backend_pass in custom_backend_passes.items():
        if custom_backend_pass is not None:
            gm_devices = [d.type for d in get_all_devices(gm)]
            if device in gm_devices:
                pass_name = "custom_backend_passes_" + device
                GraphTransformObserver(gm, pass_name).apply_gm_pass(custom_backend_pass)

    collectives_bucketing: bool = False

    if config.bucket_reduce_scatters_fx != "none":
        from torch._inductor.fx_passes.bucketing import bucket_reduce_scatter
        from torch._inductor.fx_passes.fsdp import bucket_fsdp_reduce_scatter

        p = (
            bucket_fsdp_reduce_scatter
            if "fsdp" in config.bucket_reduce_scatters_fx
            else bucket_reduce_scatter
        )
        GraphTransformObserver(gm, "bucket_reduce_scatters").apply_graph_pass(
            lambda graph: p(
                graph.owning_module,
                config.bucket_reduce_scatters_fx_bucket_size_determinator,
                config.bucket_reduce_scatters_fx,  # type: ignore[arg-type]
            )
        )
        collectives_bucketing = True

    # Fx all_gather bucketing introduces mutation op
    # Keeping it in the end to keep invariant of functional graph for previous passes.
    if config.bucket_all_gathers_fx != "none":
        from torch._inductor.fx_passes.bucketing import bucket_all_gather
        from torch._inductor.fx_passes.fsdp import bucket_fsdp_all_gather

        p = (
            bucket_fsdp_all_gather  # type: ignore[assignment]
            if "fsdp" in config.bucket_all_gathers_fx
            else bucket_all_gather
        )
        GraphTransformObserver(gm, "bucket_all_gathers").apply_graph_pass(
            lambda graph: p(
                graph.owning_module,
                config.bucket_all_gathers_fx_bucket_size_determinator,
                config.bucket_all_gathers_fx,  # type: ignore[arg-type]
            )
        )
        collectives_bucketing = True

    if collectives_bucketing:
        # Fx collectives bucketing passes require topological sort for the cases:
        # when bucketed collectives have users before the last collective in the bucket
        # AND when inputs of bucketed collective have ancestors after the first collective in the bucket.
        #
        # In this case we can not manually pick the place for bucketed collective insertion.
        # But we are guaranteed by the bucketing (independent collectives in the bucket),
        # that it is possible to reorder nodes to satisfy all ordering requirements.
        #
        # --- before bucketing ---
        # in0 = ...
        # wait_ag0 = ag(in0)
        # user0(wait_ag0)
        # ...
        # pre_in1 = ...
        # in1 = transform(pre_in1)
        # wait_ag1 = ag(in1)
        # user1(wait_ag1)
        #
        # --- after bucketing ---
        #
        # in0 = ...
        # user(wait_ag0) <--- wait_ag0 is defined only after bucketed collective.
        #
        # pre_in1 = ...
        # in1 = transform(pre_in1)
        # ag_bucket(in0+in1)
        # wait_bucket
        # wait_ag0 = wait_bucket[0]
        # wait_ag1 = wait_bucket[1]
        # user1(wait_ag1)
        stable_topological_sort(gm.graph)

    # Apply overlap scheduling if enabled
    if config.aten_distributed_optimizations.enable_overlap_scheduling:
        from torch._inductor.config import aten_distributed_optimizations as dist_opts
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        # by default, insert overlap deps within inductor
        kwargs: dict[str, object] = {"insert_overlap_deps": True}

        config_keys = (
            "collective_bucketing",
            "max_compute_pre_fetch",
            "custom_runtime_estimation",
            "insert_overlap_deps",
            "collective_estimator",
        )
        for key in config_keys:
            if (val := getattr(dist_opts, key)) is not None:
                kwargs[key] = val

        GraphTransformObserver(gm, "overlap_scheduling").apply_graph_pass(
            lambda graph: schedule_overlap_bucketing(graph.owning_module, **kwargs)  # type: ignore[arg-type]
        )

    # Keep these last, since they introduce mutation. Look at
    # ./fx_passes/README.md for a discussion of mutation invariants.
    GraphTransformObserver(gm, "reinplace_inplaceable_ops").apply_graph_pass(
        functools.partial(reinplace_inplaceable_ops, fake_tensor_updater),
    )
    GraphTransformObserver(
        gm, "decompose_triton_kernel_wrapper_functional"
    ).apply_graph_pass(decompose_triton_kernel_wrapper_functional)
    GraphTransformObserver(gm, "decompose_auto_functionalized").apply_graph_pass(
        decompose_auto_functionalized
    )
    if not torch._dynamo.config.skip_fsdp_hooks:
        GraphTransformObserver(gm, "reinplace_fsdp_all_gather").apply_graph_pass(
            comms.reinplace_fsdp_all_gather
        )
    GraphTransformObserver(gm, "decompose_scan_to_while_loop").apply_gm_pass(
        decompose_scan_to_while_loop
    )
    GraphTransformObserver(gm, "decompose_map_to_while_loop").apply_gm_pass(
        decompose_map_to_while_loop
    )

    gm.recompile()
    gm.graph.lint()


def prepare_softmax_pattern(x, dim):
    xmax = x.amax(dim=dim, keepdim=True)
    xsub = x - xmax
    xexp = xsub.exp()
    xsum = xexp.sum(dim=dim, keepdim=True)
    return xmax, xsum, xsub, xexp


def prepare_softmax_replacement(x, dim):
    """
    Return xsub since otherwise log-softmax can not be matched
    due to a use of this intermediate node. Same reason to return
    xsub.exp() for softmax.
    """
    from torch._inductor.inductor_prims import prepare_softmax_online

    xmax, xsum = prepare_softmax_online(x, dim)
    xsub = x - xmax
    return xmax, xsum, xsub, xsub.exp()


def prepare_softmax_extra_check(match):
    """
    We only have triton online softmax kernels currently.
    """
    return (
        config.online_softmax
        and match.kwargs["x"].meta["val"].device.type == "cuda"
        and config.cuda_backend == "triton"
    )


def decompose_map_to_while_loop(gm: torch.fx.GraphModule):
    """This is similar to decompose_scan_to_while_loop."""
    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.map_impl),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        assert len(kwargs) == 0, (
            "kwargs of map are not merged into args before entering decompose_map_to_while_loop_pass"
        )
        subgraph, fx_xs, fx_additional_inputs = args
        sub_gm: torch.fx.GraphModule = getattr(gm, subgraph.target)
        cur_node = match.nodes[0]
        mapped_outputs = cur_node.meta["val"]

        def lower_to_while_loop(*args, **kwargs):
            assert len(kwargs) == 0
            xs, additional_inputs = pytree.tree_unflatten(args, tree_spec)
            assert isinstance(xs, (tuple, list)) and isinstance(
                additional_inputs, (tuple, list)
            ), (xs, additional_inputs)
            map_length = xs[0].size(0)
            loop_idx = torch.zeros([], dtype=torch.int64, device=torch.device("cpu"))

            # Similar to NOTE [Pre-allocate scan's output buffer]
            bound_symbols = {
                arg.node.expr: arg
                for arg in pytree.tree_leaves((args, map_length))
                if isinstance(arg, torch.SymInt)
            }
            out_buffers = [
                torch.empty_strided(
                    resolve_shape_to_proxy(out.size(), bound_symbols),
                    resolve_shape_to_proxy(out.stride(), bound_symbols),
                    device=out.device,
                    dtype=out.dtype,
                    layout=out.layout,
                    requires_grad=out.requires_grad,
                )
                for out in mapped_outputs
            ]

            while_loop_operands = (loop_idx, out_buffers, xs)
            while_loop_flat_operands, operands_spec = pytree.tree_flatten(
                while_loop_operands
            )
            while_loop_additional_inputs = additional_inputs
            _, operands_and_additional_inputs_spec = pytree.tree_flatten(
                (*while_loop_operands, additional_inputs)
            )

            def cond_fn(*flat_args):
                loop_idx, _, _, _ = pytree.tree_unflatten(
                    flat_args,
                    operands_and_additional_inputs_spec,
                )
                return loop_idx < map_length

            def body_fn(*flat_args):
                loop_idx, out_bufs, xs, additional_inputs = pytree.tree_unflatten(
                    flat_args,
                    operands_and_additional_inputs_spec,
                )

                idx_int = loop_idx.item()
                torch.ops.aten._assert_scalar.default(idx_int >= 0, "")
                torch.ops.aten._assert_scalar.default(idx_int < map_length, "")
                sub_xs = [torch.ops.aten.select.int(x, 0, idx_int) for x in xs]
                outs = sub_gm(*sub_xs, *additional_inputs)

                for out, buffer in zip(outs, out_bufs):
                    buffer_slice = torch.ops.aten.select.int(buffer, 0, idx_int)
                    buffer_slice.copy_(out)
                return loop_idx + 1, *out_bufs, *xs

            _, final_out, _ = pytree.tree_unflatten(
                torch.ops.higher_order.while_loop(
                    cond_fn,
                    body_fn,
                    tuple(while_loop_flat_operands),
                    tuple(while_loop_additional_inputs),
                ),
                operands_spec,
            )
            return (final_out,)

        lower_to_while_loop_args, tree_spec = pytree.tree_flatten(
            (fx_xs, fx_additional_inputs)
        )
        match.replace_by_example(
            lower_to_while_loop, lower_to_while_loop_args, run_functional_passes=False
        )

    graph_pass.apply(gm)

    for _node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.map_impl
    ):
        raise AssertionError("map is not lowered to while_loop")


def resolve_shape_to_proxy(
    shape: list[int | torch.SymInt], bound_symbols: dict[Any, Any]
):
    """
    Given a list of symints/ints, this function returns a calculated expression of bound_symbols' values.
    When we trace this function, we'll get a graph with call_function nodes that describes how the shape expr is
    computed from bound_symbols' values.

    Suppose shape = (s1*s2, s1+s2) and bound_symbols = {s1: arg0, s2: arg1}, the result will be
    (arg0 * arg1, arg0 + arg1).
    """
    from torch.utils._sympy.interp import sympy_interp
    from torch.utils._sympy.reference import PythonReferenceAnalysis

    ret = []
    for s in shape:
        if isinstance(s, torch.SymInt):
            ret.append(
                sympy_interp(
                    PythonReferenceAnalysis,
                    bound_symbols,
                    s.node.expr,
                ),
            )
        else:
            assert isinstance(s, int)
            ret.append(s)
    return ret


def decompose_scan_to_while_loop(gm: torch.fx.GraphModule):
    """
    NOTE [decompose scan to while_loop]
    This pass decomposes `scan` to  `while_loop` by replacing the scan fx_node with a while_loop hop.

    Suppose we have a function f:

        def f():
            init = torch.zeros([])
            xs = torch.arange(4)
            ys = []
            for i in range(xs.size(0)):
                init = xs[i] + init
                ys.append(init)

            # Return the final carry and stack the intermediates
            return init, torch.stack(ys)

    We could rewrite it with a scan with the benefits of reducing compilation time/binary size, reducing
    memory usage, supporting loops over unbacked shapes and cudagraph etc.

        def g():
            def step_fn(init: torch.Tensor, x: torch.Tensor):
                next_init = x + init
                return next_init, next_init

            init = torch.zeros([])
            xs = torch.arange(4)
            final_carry, ys = torch._higher_order.scan(step_fn, init, xs)
            return final_carry, ys

    This pass will rewrite scan into:

        def k():
            init = torch.zeros([])
            xs = torch.arange(4)

            # we create a loop_idx and loop through xs.shape[0]
            loop_idx = torch.zeros([])
            ys = torch.empty_strided(_shape_stride_of_ys)
            def cond_fn(loop_idx, ys, init, xs):
                return loop_idx < xs.shape[0]

            # we pre-allocate the output buffer ys and inplace
            # copy the y of each intermediate into a slice.
            # NOTE [Pre-allocate scan's output buffer].
            def body_fn(loop_idx, ys, init, xs):
                int_idx = loop_idx.item()
                next_init, y = step_fn(init, xs[int_idx])
                ys[int_idx].copy_(y)
                return loop_idx + 1, ys, next_init, xs

            final_carry, _, _, ys = torch._higher_order.while_loop(cond_fn, body_fn, (loop_idx, ys, init, xs))
            return final_carry, ys
    """

    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.scan),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._higher_order_ops.scan import _extract_carry_and_out

        assert len(kwargs) == 0, (
            "kwargs of scan are not merged into args before entering decompose_scan_to_while_loop_pass"
        )

        combine_subgraph, fx_init, fx_xs, fx_additional_inputs = args
        assert combine_subgraph.op == "get_attr", "first arg is not combine_subgraph"
        sub_gm: torch.fx.GraphModule = getattr(gm, combine_subgraph.target)
        cur_node = match.nodes[0]
        num_init_leaves = len(fx_init)
        _, ys_outputs = _extract_carry_and_out(cur_node.meta["val"], num_init_leaves)

        def lower_to_while_loop(*args, **kwargs):
            """
            The traced graph of this function will be used to replace the original scan fx_node.
            """
            assert len(kwargs) == 0

            # Step 1: construct necessary inputs to while_loop based on scan's input.
            (
                init,
                xs,
                additional_inputs,
            ) = pytree.tree_unflatten(args, tree_spec)
            scan_length = xs[0].size(0)
            loop_idx = torch.zeros([], dtype=torch.int64, device=torch.device("cpu"))

            # NOTE [Pre-allocate scan's output buffer]
            # In order to pre-allocate the output buffer for ys, we rely on the meta of scan's fx_node.
            # However, the meta consists of concrete symints, we need to bind those symints with
            # proxies in order to trace the torch.empty_strided call correctly.
            #
            # Also note that basic free symbols of tensor's shapes are guaranteed to be lifted as subgraph inputs
            # in dynamo so we can always re-construct the sym expression from placeholders.
            # See Note [Auto lift basic free symbols when create_graph_input] for how this is done.
            bound_symbols = {
                arg.node.expr: arg
                for arg in pytree.tree_leaves((args, scan_length))
                if isinstance(arg, torch.SymInt)
            }
            ys_outs = [
                torch.empty_strided(
                    resolve_shape_to_proxy(ys_out.size(), bound_symbols),
                    resolve_shape_to_proxy(ys_out.stride(), bound_symbols),
                    device=ys_out.device,
                    dtype=ys_out.dtype,
                    layout=ys_out.layout,
                    requires_grad=ys_out.requires_grad,
                )
                for ys_out in ys_outputs
            ]

            while_loop_operands = (loop_idx, ys_outs, init, xs)
            flat_operands, operands_spec = pytree.tree_flatten(while_loop_operands)
            _, operands_and_additional_inputs_spec = pytree.tree_flatten(
                (*while_loop_operands, additional_inputs)
            )

            # Step 2: create the cond_fn and body_fn for while_loop
            def cond_fn(*flat_args):
                loop_idx, _, _, _, _ = pytree.tree_unflatten(
                    flat_args, operands_and_additional_inputs_spec
                )  # type: ignore[has-type]
                return loop_idx < scan_length  # type: ignore[has-type]

            def body_fn(*flat_args):
                loop_idx, ys_outs, carry, xs, additional_inputs = pytree.tree_unflatten(
                    flat_args,
                    operands_and_additional_inputs_spec,  # type: ignore[has-type]
                )

                idx_int = loop_idx.item()
                torch.ops.aten._assert_scalar.default(idx_int >= 0, "")
                torch.ops.aten._assert_scalar.default(idx_int < scan_length, "")
                sub_xs = [torch.ops.aten.select.int(x, 0, idx_int) for x in xs]
                next_carry, ys = _extract_carry_and_out(
                    sub_gm(*(list(carry) + sub_xs + list(additional_inputs))),
                    num_init_leaves,
                )
                for y, y_out in zip(ys, ys_outs):
                    y_out_slice = torch.ops.aten.select.int(y_out, 0, idx_int)
                    y_out_slice.copy_(y)
                return loop_idx + 1, *ys_outs, *next_carry, *xs

            # Step 3: call the while_loop operator
            _, ys_outs, last_carry, _ = pytree.tree_unflatten(
                torch.ops.higher_order.while_loop(
                    cond_fn,
                    body_fn,
                    tuple(flat_operands),
                    tuple(additional_inputs),
                ),
                operands_spec,
            )
            return list(last_carry) + list(ys_outs)

        lower_to_while_loop_args, tree_spec = pytree.tree_flatten(
            (
                fx_init,
                fx_xs,
                fx_additional_inputs,
            )
        )
        match.replace_by_example(
            lower_to_while_loop,
            lower_to_while_loop_args,
            run_functional_passes=False,
        )

    graph_pass.apply(gm)

    for _node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.scan
    ):
        raise AssertionError("scan is not lowered to while_loop")


@init_once_fakemode
def lazy_init():
    if torch._C._has_mkldnn:
        from . import decompose_mem_bound_mm  # noqa: F401
        from .mkldnn_fusion import _mkldnn_fusion_init

        _mkldnn_fusion_init()

    # Put this patterns in post-grad pass rather than joint-graph
    # pass since otherwise there will be perf/peak-memory regression:
    # https://github.com/pytorch/pytorch/issues/148141
    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        prepare_softmax_pattern,
        # pyrefly: ignore [bad-argument-type]
        prepare_softmax_replacement,
        [torch.empty(4, 8)],
        scalar_workaround=dict(dim=-1),
        # pyrefly: ignore [bad-argument-type]
        trace_fn=fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_dicts=pass_patterns[1],
        extra_check=prepare_softmax_extra_check,
    )


def reorder_for_locality(graph: torch.fx.Graph):
    if torch.distributed.is_available():

        def check():
            # This is a wait node, and `other_node`` is some collective node.
            # Eager semantics allow waits to be issued in a different order than
            # the collectives. Reordering this wait node might reorder collectives
            # which cause hangs. Once we have SPMD mode, we can safely reorder them.
            # However, increasing the locality between a collective and its wait node
            # is generally worse for performance.
            return node.target != torch.ops._c10d_functional.wait_tensor.default
    else:

        def check():
            return True

    def visit(other_node):
        if (
            other_node.op == "call_function"
            and other_node.target != operator.getitem
            and all((n in seen_nodes) for n in other_node.users)
            and get_mutation_region_id(graph, node)
            == get_mutation_region_id(graph, other_node)
            and check()
        ):
            # move node's producers right before it
            node.prepend(other_node)

    seen_nodes = OrderedSet[torch.fx.Node]()

    # only reorder nodes before the first copy_ in the graph.
    # copy_ will appear at the end of functionalized graphs when there is mutation on inputs,
    # and this reordering doesn't work well with mutation
    first_copy = next(
        iter(graph.find_nodes(op="call_function", target=torch.ops.aten.copy_.default)),
        None,
    )
    past_mutating_epilogue = first_copy is None

    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        if not past_mutating_epilogue:
            past_mutating_epilogue = node is first_copy
            continue

        torch.fx.map_arg((node.args, node.kwargs), visit)


def register_lowering_pattern(
    pattern, extra_check=_return_true, pass_number=1
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register an aten to inductor IR replacement pattern
    """
    return pattern_matcher.register_lowering_pattern(
        pattern,
        extra_check,
        # pyrefly: ignore [bad-argument-type]
        pass_dict=pass_patterns[pass_number],
    )


################################################################################
# Actual patterns below this point.
# Priority of patterns is:
#   - later output nodes first
#   - order patterns are defined in
################################################################################


def is_valid_mm_plus_mm(match: Match):
    if not (config.max_autotune or config.max_autotune_gemm):
        return False

    *_b1, m1, k1 = match.kwargs["mat1"].meta.get("tensor_meta").shape
    *_b2, k2, n1 = match.kwargs["mat2"].meta.get("tensor_meta").shape
    if k1 != k2:
        return False

    *_b1, m2, k3 = match.kwargs["mat3"].meta.get("tensor_meta").shape
    *_b2, k4, n2 = match.kwargs["mat4"].meta.get("tensor_meta").shape
    if k3 != k4:
        return False

    if m1 != m2 or n1 != n2:
        return False

    return True


def scatter_upon_const_tensor_extra_check(m):
    if not config.optimize_scatter_upon_const_tensor:
        return False
    full_shape = m.kwargs["shape"]
    selector = m.kwargs["selector"]
    dim = m.kwargs["dim"]
    if dim < 0:
        dim += len(full_shape)

    selector_ft = selector.meta["val"]
    assert selector_ft.dim() == len(full_shape)

    for idx, select_sz, full_sz in zip(
        itertools.count(), selector_ft.shape, full_shape
    ):
        if idx == dim:
            continue

        # TODO: the pattern can be updated to support the case that index tensor
        # is shorter. But that will need a more complex condition expression
        # especially for multi-dimensional tensors.
        # Skip it for now.
        if isinstance(full_sz, fx.Node):
            full_sz = full_sz.meta["val"]
        if select_sz < full_sz:
            return False

    # Actually we can support small size larger than 1. It would be a bit
    # tedius. E.g., we load all the index values (not many) and compare
    # them with the position in tensor to decide what value to return.
    return selector_ft.size(dim) == 1


@register_lowering_pattern(
    CallFunction(
        aten.scatter.value,
        CallFunction(
            aten.full,
            KeywordArg("shape"),
            KeywordArg("background_val"),
            dtype=KeywordArg("dtype"),
        ),
        KeywordArg("dim"),
        KeywordArg("selector"),
        KeywordArg("val"),  # scalar value
    ),
    extra_check=scatter_upon_const_tensor_extra_check,
)
def scatter_upon_const_tensor(
    match: Match, shape, background_val, dtype, dim, selector, val
):
    """
    Match the pattern of full+scatter into a pointwise.

    TODO: Right now the scatter value must be a scalar. But we could support it
    when it is a tensor as well.
    """
    from torch._inductor import metrics

    # Check if inputs are tensors instead of inductor IR nodes
    if isinstance(selector, torch.Tensor):
        # Return a fake tensor with the proper shape that this operator is intended to return
        device = selector.device if hasattr(selector, "device") else torch.device("cpu")
        return torch.empty(shape, dtype=dtype, device=device)

    # pyrefly: ignore [bad-assignment]
    metrics.num_matches_for_scatter_upon_const_tensor += 1

    selector_loader = selector.make_loader()

    def inner_fn(idx):
        selector_idx = list(idx)
        selector_idx[dim] = 0

        selector = selector_loader(selector_idx)
        return ops.where(
            selector == ops.index_expr(idx[dim], torch.int64),
            ops.constant(val, dtype),
            ops.constant(background_val, dtype),
        )

    return ir.Pointwise.create(
        device=selector.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=shape,
    )


@register_lowering_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, KeywordArg("mat1"), KeywordArg("mat2")),
        CallFunction(aten.mm, KeywordArg("mat3"), KeywordArg("mat4")),
    ),
    extra_check=is_valid_mm_plus_mm,
)
def mm_plus_mm(match: Match, mat1, mat2, mat3, mat4):
    return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)


@register_graph_pattern(
    CallFunction(
        aten.cumsum.default,
        CallFunction(
            torch.ops.aten.full.default,
            KeywordArg("shape"),
            KeywordArg("fill_value"),
            dtype=KeywordArg("dtype"),
            layout=Ignored(),
            device=KeywordArg("device"),
            pin_memory=False,
            _users=MULTIPLE,
        ),
        KeywordArg("dim"),
        _users=MULTIPLE,
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[1],
)
def pointless_cumsum_replacement(match: Match, shape, fill_value, device, dtype, dim):
    """Based on a pattern in OPTForCausalLM"""

    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        # cumsum promotes all integral types to int64
        dtype = torch.int64

    def repl(*shape):
        dim_size = shape[dim]
        idx = torch.arange(1, dim_size + 1, device=device, dtype=dtype)

        inter_shape = [1] * len(shape)
        inter_shape[dim] = dim_size
        return (idx * fill_value).view(inter_shape).expand(shape)

    # only replace the output node, not all nodes
    match.nodes = [match.output_node()]
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, list(shape))


_cat_1 = CallFunction(aten.cat, Arg(), 1, _users=2)


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        [
            _cat_1,
            CallFunction(
                aten.slice,
                _cat_1,
                1,
                0,
                KeywordArg("size"),
            ),
        ],
        1,
    )
)
def cat_slice_cat(match, cat_input, size, dim=1):
    """
    This is an example of a more complex pattern where cat_1 is used
    multiple times inside the pattern.  We fold 2 calls to cat into one.

    Matches:
        cat_1: f32[1024, 4077] = torch.ops.aten.cat.default([add_26, primals_217], 1)
        slice_1: f32[1024, 4077] = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
        slice_2: f32[1024, 19] = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
        cat_2: f32[1024, 4096] = torch.ops.aten.cat.default([cat_1, slice_2], 1)


    Rewrite to:
        slice_2 = torch.ops.aten.slice.Tensor(add_26, 1, 0, 19)
        cat_2 = torch.ops.aten.cat.default([add_26, primals_217, slice2], 1)
    """
    first, *rest = cat_input
    # Optimization is optional, because we can just not fold the cat
    # size should be within first.get_size()[dim] such that the optimization is valid.
    # For negative `end`, we currently fallback to not optimizing.
    if size >= 0 and V.graph.sizevars.statically_known_leq(size, first.get_size()[dim]):
        # fold 2 cats into 1 cat
        return L[aten.cat](
            [
                first,
                *rest,
                L[aten.slice](first, dim, 0, size),
            ],
            dim,
        )
    else:
        # don't expect to hit this case, just fall back
        tmp = L[aten.cat](cat_input, dim)
        return L[aten.cat](
            [
                tmp,
                L[aten.slice](tmp, dim, 0, size),
            ],
            dim,
        )


def is_valid_splitwithsizes_cat(match):
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    split_node, cat_node = split_nodes[0], cat_nodes[0]
    # The dim of split and cat should match for passthrough
    if get_arg_value(split_node, 2, "dim") != get_arg_value(cat_node, 1, "dim"):
        return False
    get_item_args = OrderedSet(
        get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes
    )
    assert None not in get_item_args
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    # All parts of split should be included in the cat
    if get_item_args != OrderedSet(range(len(split_sizes))):
        return False
    # The order of get_item_args should same with cat_node used.
    # For example, if the split_node like split_with_sizes(input, [2, 2, 3], 1),
    # the cat node should be like cat([get_item(0), get_item(1), get_item(2)], 1).
    cat_items_args_order = [
        get_arg_value(item_node, 1) for item_node in get_arg_value(cat_node, 0)
    ]
    if cat_items_args_order != list(range(len(split_sizes))):
        return False

    return True


def same_meta(node1: torch.fx.Node, node2: torch.fx.Node):
    """True if two nodes have the same metadata"""
    val1 = node1.meta.get("val")
    val2 = node2.meta.get("val")
    return (
        val1 is not None
        and val2 is not None
        and statically_known_true(sym_eq(val1.size(), val2.size()))
        and val1.layout == val2.layout
        and val1.dtype == val2.dtype
        and val1.device == val2.device
        and (
            val1.layout != torch.strided
            or statically_known_true(sym_eq(val1.stride(), val2.stride()))
        )
    )


noop_registry: dict[Any, Any] = {}


def register_noop_decomp(targets, nop_arg=0):
    def register_fun(cond):
        register_decomposition(targets, registry=noop_registry, unsafe=True)(
            (cond, nop_arg)  # type: ignore[arg-type]
        )
        return cond

    return register_fun


@register_noop_decomp(aten.slice)
def slice_noop(self, dim=0, start=None, end=None, step=1):
    if start is None or end is None:
        return False

    slice_dim_size = self.shape[dim]
    if (
        statically_known_true(sym_eq(start, 0))
        and (
            statically_known_true(end >= 2**63 - 1)
            or statically_known_true(end >= slice_dim_size)
        )
        and statically_known_true(sym_eq(step, 1))
    ):
        return True
    return False


@register_noop_decomp(aten.slice_scatter, 1)
def slice_scatter_noop(self, src, dim=0, start=None, end=None, step=1):
    if start is None:
        start = 0
    if end is None:
        end = 2**63 - 1
    slice_scatter_dim_size = self.shape[dim]
    if (
        self.shape == src.shape
        and start == 0
        and (
            statically_known_true(end >= 2**63 - 1)
            or statically_known_true(end >= slice_scatter_dim_size)
        )
        and step == 1
    ):
        return True
    return False


@register_noop_decomp(aten.repeat)
def repeat_noop(self, repeats):
    return all(r == 1 for r in repeats)


@register_noop_decomp(aten.constant_pad_nd)
def constant_pad_nd(x, padding, fill_value=0):
    return all(p == 0 for p in padding)


@register_noop_decomp(torch.ops.prims.convert_element_type)
def convert_element_type_noop(x, dtype: torch.dtype):
    return x.dtype == dtype


@register_noop_decomp(torch.ops.prims.device_put)
def device_put_noop(x, device, non_blocking=True):
    return x.device == decode_device(device)


@register_noop_decomp([aten.ceil, aten.floor, aten.round, aten.trunc])
def int_noop(x):
    return is_integer_dtype(x.dtype)


@register_noop_decomp([aten.pow])
def pow_noop(a, b):
    return isinstance(b, int) and b == 1


@register_noop_decomp([aten.cat], lambda args: args[0][0])
def cat_noop(inputs, dim=0):
    return len(inputs) == 1


@register_noop_decomp(aten.view.default)
def view_default_noop(arg, size):
    return statically_known_true(sym_eq(arg.shape, tuple(size)))


@register_noop_decomp(aten.view.dtype)
def view_dtype_noop(arg, dtype):
    return arg.dtype == dtype


# Note, we also always have a check for identical metadata, which is why these
# are safe
@register_noop_decomp([aten.copy], nop_arg=1)
@register_noop_decomp([aten.alias, aten.clone])
def true_noop(*args, **kwargs):
    return True


def remove_noop_ops(graph: torch.fx.Graph):
    """
    Removes both operations that are essentially aten.clone and operations that are essentially aten.alias from the graph.
    """
    inputs = OrderedSet[torch.fx.Node]()
    input_storages = OrderedSet[int | None]()
    output_storages = OrderedSet[int | None]()

    for node in graph.find_nodes(op="placeholder"):
        inputs.add(node)
        input_storages.add(get_node_storage(node))

    output_node = next(iter(reversed(graph.nodes)))
    assert output_node.op == "output"
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        # nested subgraphs can have singleton outputs
        outputs = (outputs,)
    for out in outputs:
        if isinstance(out, torch.fx.Node):
            output_storages.add(get_node_storage(out))

    for node in graph.nodes:
        if node.target in noop_registry:
            cond, src_index = noop_registry[node.target]
            if isinstance(src_index, int):
                src = node.args[src_index]
            else:
                src = src_index(node.args)
            if not isinstance(src, torch.fx.Node):
                continue
            # Don't introduce new aliasing between inputs and outputs.
            # See fx_passes/README.md for a discussion of why this is
            # necessary.
            node_storage = get_node_storage(node)
            src_storage = get_node_storage(src)
            node_is_view = node_storage == src_storage
            if (
                not node_is_view
                and node_storage in output_storages
                and (src_storage in input_storages or src_storage in output_storages)
            ):
                continue

            # Even if input and outputs are expected to alias,
            # don't make "node is src" True
            if (
                node_is_view
                and node in output_node.args
                and (src in inputs or src in output_node.args)
            ):
                continue

            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
                node.replace_all_uses_with(src)
                graph.erase_node(node)


def remove_assert_ops(graph: torch.fx.Graph):
    """
    Removes aten._assert_tensor_metadata.default op because
    1) it will be lowered to a no-op in inductor
    2) it can block fusion, such as unfuse_bias_add_to_pointwise fusion.

    This op could come from aten.to functionalization in export.

    For example, if we have a graph like below

    %addmm = aten.addmm.default(%linear_bias, %arg3_1, %permute)
    %_assert_tensor_metadata = aten._assert_tensor_metadata.default(%addmm, None, None, torch.float16)
    %convert_element_type_3 = prims.convert_element_type.default(%addmm, torch.float32)
    %pow_1 = aten.pow.Tensor_Scalar(%convert_element_type_3, 2)

    We still want to fuse add from addmm with pow, instead of fusing add with mm, according to unfuse_bias_add_to_pointwise fusion.

    However, aten._assert_tensor_metadata.default is not a pointwise op, and would fail the should_prefer_unfused_addmm check.

    We remove this op so it doesn't block fusion decisions. It's safe because this op is lowered to a no-op with @register_lowering.

    """
    for node in graph.find_nodes(
        op="call_function", target=torch.ops.aten._assert_tensor_metadata.default
    ):
        graph.erase_node(node)


def decompose_triton_kernel_wrapper_functional(graph):
    """Decomposes triton_kernel_wrapper_functional nodes into clones and the underlying
    mutation node.

    We assume that the reinplacing pass runs before this; the reinplacing pass
    tells us (via rewriting the arguments or .meta to those nodes) which
    Tensors we should clone and which Tensors are safe to reinplace.
    """
    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.triton_kernel_wrapper_functional),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_functional_dense,
        )

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            return (triton_kernel_wrapper_functional_dense(*args, **kwargs),)

        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    graph_pass.apply(graph)

    for _ in graph.find_nodes(
        op="call_function",
        target=torch.ops.higher_order.triton_kernel_wrapper_functional,
    ):
        raise AssertionError("triton_kernel_wrapper_functional was not removed")


def decompose_auto_functionalized(graph):
    """Decomposes auto_functionalized nodes into clones and the underlying
    mutation node.

    We assume that the reinplacing pass runs before this; the reinplacing pass
    tells us (via rewriting the arguments or .meta to those nodes) which
    Tensors we should clone and which Tensors are safe to reinplace.
    """
    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.auto_functionalized),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized_dense

        only_clone_these_tensors = tuple(
            match.nodes[0].meta.get("only_clone_these_tensors", [])
        )

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            assert len(args) == 1
            mode = args[0]
            return auto_functionalized_dense(mode, only_clone_these_tensors, **kwargs)

        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.auto_functionalized_v2),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._higher_order_ops.auto_functionalize import (
            auto_functionalized_v2_dense,
        )

        only_clone_these_bases = tuple(
            match.nodes[0].meta.get("only_clone_these_tensors", [])
        )

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        def _maybe_resolve_constant_get_attr(node):
            # Resolve getattr node to its value because they don't always have meta["val"]
            if (
                isinstance(node, torch.fx.Node)
                and node.op == "get_attr"
                and "val" not in node.meta
            ):
                const_attr = getattr(graph.owning_module, node.target)  # type: ignore[arg-type]
                assert isinstance(
                    const_attr, (torch.fx.GraphModule, pytree.TreeSpec)
                ), (type(const_attr), const_attr)
                return const_attr
            return node

        flat_args = [_maybe_resolve_constant_get_attr(arg) for arg in flat_args]

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            assert len(args) == 1
            mutable_op = args[0]
            return auto_functionalized_v2_dense(
                mutable_op, only_clone_these_bases, **kwargs
            )

        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    graph_pass.apply(graph)

    # Remove unused get_attr nodes and their corresponding attributes from the graph module.
    # When auto_functionalizing a hop, we need to clean up get_attr nodes for _constant_schema
    # and the auto_functionalized graph module that are no longer referenced.
    unused_get_attr_nodes = []
    removable_attrs: OrderedSet[torch.fx.node.Target] = OrderedSet()
    protected_attrs: OrderedSet[torch.fx.node.Target] = OrderedSet()

    # First pass: identify unused get_attr nodes and track attribute usage
    for node in graph.nodes:
        if node.op != "get_attr":
            continue

        if len(node.users) == 0:
            # Node is unused, mark for removal
            unused_get_attr_nodes.append(node)

            # Check if the attribute can be removed from the module
            if (
                hasattr(graph.owning_module, node.target)
                and isinstance(
                    getattr(graph.owning_module, node.target), torch.fx.GraphModule
                )
                and node.target not in protected_attrs
            ):
                removable_attrs.add(node.target)
        else:
            # Node is used, protect its attribute from removal
            if node.target in removable_attrs:
                removable_attrs.remove(node.target)
            protected_attrs.add(node.target)

    # Second pass: clean up unused nodes and attributes
    for node in unused_get_attr_nodes:
        graph.erase_node(node)

    for attr_name in removable_attrs:
        assert isinstance(attr_name, str)
        delattr(graph.owning_module, attr_name)

    graph.lint()

    for _ in graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized
    ):
        raise AssertionError("auto_functionalized was not removed")

    for _ in graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized_v2
    ):
        raise AssertionError("auto_functionalized_v2 was not removed")


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        ListOf(
            CallFunction(
                operator.getitem,
                CallFunction(
                    aten.split_with_sizes,
                    KeywordArg("input_"),
                    Ignored(),
                    Ignored(),
                    _users=MULTIPLE,
                ),
                Ignored(),
            ),
        ),
        Ignored(),
    ),
    pass_number=2,
    extra_check=is_valid_splitwithsizes_cat,
)
def splitwithsizes_cat_replace(match, input_):
    return input_


def is_valid_cat_splitwithsizes(match):
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    split_node, cat_node = split_nodes[0], cat_nodes[0]

    # the cat node has other users: can't eliminate
    if len(cat_node.users) > 1:
        return False

    # the dim of the cat and split should match
    dim = get_arg_value(split_node, 2, "dim")
    if dim != get_arg_value(cat_node, 1, "dim"):
        return False

    cat_inputs = list(get_arg_value(cat_node, 0))
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    # the number of input tensors in cat and the
    # length of the split sizes should match
    if len(cat_inputs) != len(split_sizes):
        return False

    for cat_input, split_size in zip(cat_inputs, split_sizes):
        # each cat input tensor's size along dim
        # should match the corresponding split size
        if "val" not in cat_input.meta:
            return False
        cat_input_size = cat_input.meta["val"].size(dim)
        if cat_input_size != split_size:
            return False

    return True


@register_lowering_pattern(
    CallFunction(
        aten.split_with_sizes,
        CallFunction(
            aten.cat,
            KeywordArg("input_"),
            Ignored(),
            _users=MULTIPLE,
        ),
        Ignored(),
        Ignored(),
    ),
    pass_number=2,
    extra_check=is_valid_cat_splitwithsizes,
)
def cat_splitwithsizes_replace(match, input_):
    return input_


def view_to_reshape(gm):
    """
    Replace view ops in the GraphModule to reshape ops.
    """
    subgraph_names: OrderedSet[str] = OrderedSet(
        x.target for x in gm.graph.find_nodes(op="get_attr")
    )

    for child_name, child_mod in gm.named_children():
        if child_name in subgraph_names and isinstance(child_mod, torch.fx.GraphModule):
            view_to_reshape(child_mod)

    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.view.default
    ):
        nd.target = torch.ops.aten.reshape.default


def should_prefer_unfused_addmm(match):
    inp = match.kwargs["inp"]
    if not is_gpu(inp.meta["val"].device.type):
        return False

    return has_uses_tagged_as(
        match.output_node(),
        (torch.Tag.pointwise, torch.Tag.reduction),
    )


@register_graph_pattern(
    CallFunction(
        aten.addmm,
        KeywordArg("inp"),
        Arg(),
        Arg(),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[2],
    extra_check=should_prefer_unfused_addmm,
)
def unfuse_bias_add_to_pointwise(match: Match, mat1, mat2, *, inp, alpha, beta):
    def repl(inp, x1, x2, alpha, beta):
        mm_result = x1 @ x2
        if alpha != 1:
            mm_result = alpha * mm_result
        if beta != 1:
            inp = beta * inp
        return inp + mm_result

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [inp, mat1, mat2, alpha, beta])


def is_valid_addmm_activation_fusion(match: Match) -> bool:
    if not is_gpu(match.kwargs["inp"].meta["val"].device.type):
        return False

    # Only beta == 1 implies activation epilogue in addmm
    if match.kwargs["beta"] not in (1, 1.0, 1+0j, 1-0j):
        return False

    # GELU epilogue fusion only for the "tanh" approximation
    if "approximate" in match.kwargs:
        # Somehow match.kwargs.setdefault("approximate", <default>) does not work
        gelu_approximation = match.kwargs["approximate"]
        if gelu_approximation != "tanh":
            return False

    return not has_uses_tagged_as(
        match.output_node(),
        (torch.Tag.pointwise, torch.Tag.reduction),
    )


@register_graph_pattern(
    CallFunction(
        aten.relu,
        CallFunction(aten.addmm, KeywordArg("inp"), Arg(), Arg(), beta=KeywordArg("beta"), alpha=KeywordArg("alpha")),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[1],
    extra_check=is_valid_addmm_activation_fusion,
)
def relu_addmm_fusion(match: Match, mat1, mat2, *, inp, beta, alpha):
    def replacement(inp, mat1, mat2, beta, alpha):
        return aten._addmm_activation(inp, mat1, mat2, beta=beta, alpha=alpha)

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [inp, mat1, mat2, beta, alpha])


@register_graph_pattern(
    CallFunction(
        aten.gelu,
        CallFunction(aten.addmm, KeywordArg("inp"), Arg(), Arg(), beta=KeywordArg("beta"), alpha=KeywordArg("alpha")),
        KeywordArg("approximate"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[1],
    extra_check=is_valid_addmm_activation_fusion,
)
def gelu_addmm_fusion(match: Match, mat1, mat2, *, inp, beta, alpha, approximate):
    def replacement(inp, mat1, mat2, beta, alpha, approximate):
        return aten._addmm_activation(inp, mat1, mat2, beta=beta, alpha=alpha, use_gelu=True)

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [inp, mat1, mat2, alpha, beta, approximate])


def is_valid_addmm_fusion(match):
    mat1, mat2 = match.args
    inp = match.kwargs["inp"]

    if not (
        isinstance(inp, torch.fx.Node) and isinstance(inp.meta["val"], torch.Tensor)
    ):
        return False  # Input is a number

    in_shape = inp.meta["val"].shape
    mm_shape = mat1.meta["val"].shape[0], mat2.meta["val"].shape[1]
    matched = is_expandable_to(in_shape, mm_shape)
    if not matched:
        return False  # Shape mismatch

    inp_dtype = inp.meta["val"].dtype

    # aten cublas integration assumes equal dtypes
    if inp_dtype != mat1.meta["val"].dtype or inp_dtype != mat2.meta["val"].dtype:
        return False

    return not should_prefer_unfused_addmm(match)


@register_graph_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, Arg(), Arg()),
        KeywordArg("inp"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[2],
    extra_check=is_valid_addmm_fusion,
)
@register_graph_pattern(
    CallFunction(
        aten.add,
        KeywordArg("inp"),
        CallFunction(aten.mm, Arg(), Arg()),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[2],
    extra_check=is_valid_addmm_fusion,
)
def addmm(match, mat1, mat2, *, inp):
    def repl(inp, mat1, mat2):
        return aten.addmm(inp, mat1, mat2)

    match.replace_by_example(repl, [inp, mat1, mat2])


def register_partial_reduction_pattern():
    "Reuse partial reductions in complete reductions"

    # post grad equivalents
    equiv_red = {
        aten.amax.default: aten.max.default,
        aten.amin.default: aten.min.default,
    }

    # TODO: to support other reductions like sum, would need to skip
    # lower precision reductions since partial output would need to be kept at fp32.
    for red_op in (aten.amax.default, aten.amin.default):
        inp = KeywordArg("input")
        partial_reduc = CallFunction(
            red_op, inp, KeywordArg("reduced_dims"), KeywordArg("keepdim")
        )
        full_reduc = CallFunction([red_op, equiv_red[red_op]], inp)

        @register_graph_pattern(
            MultiOutputPattern([partial_reduc, full_reduc]),
            # pyrefly: ignore [bad-argument-type]
            pass_dict=pass_patterns[2],
        )
        def reuse_partial(match, input, reduced_dims, keepdim):
            partial_red, full_red = match.output_nodes()

            # if they're small, reuse not worth it
            if not statically_known_true(input.meta["val"].numel() >= 4096):
                return True

            def replacement(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                partial = partial_red.target(inp, reduced_dims, keepdim)
                complete = full_red.target(partial)
                return (partial, complete)

            counters["inductor"]["partial_reduction_reuse"] += 1
            match.replace_by_example(replacement, [input])


register_partial_reduction_pattern()


def check_shape_cuda_and_fused_int_mm_mul_enabled(match):
    return (
        config.force_fuse_int_mm_with_mul
        and len(getattr(match.args[2].meta.get("val"), "shape", [])) == 2
        and getattr(match.args[2].meta.get("val"), "is_cuda", False)
    )


def is_index_put_and_requires_h2d_sync_for_gpu_value(node):
    from torch.fx.operator_schemas import normalize_function

    if node.target not in [
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put_.default,
    ]:
        return False
    # Inductor falls back to aten.index_put_.
    # index_put_ will will call nonzero() and perform a H2D sync if
    # any of its indices are bool/byte tensors
    # However, it will short-circuit this H2D sync and run mask_fill_
    # if the value we are putting is a cpu scalar.
    # Therefore, when inductor sees an index_put_ with byte tensor indices,
    # it should *not* convert the cpu scalar value into a gpu tensor.
    args_, _kwargs = normalize_function(node.target, node.args, node.kwargs)  # type: ignore[misc]
    any_byte_bool_indices = False
    indices = args_[1]
    for i in indices:
        if i is not None and i.meta["val"].dtype in [torch.bool, torch.int8]:
            any_byte_bool_indices = True

    val = args_[2].meta["val"]
    val_is_cpu_scalar = val.device.type == "cpu" and val.numel() == 1
    # If both these conditions hold, then converting the val
    # to a gpu tensor will incur a H2D sync when inductor calls aten.index_put_
    return any_byte_bool_indices and val_is_cpu_scalar


class ConstructorMoverPass:
    def __init__(
        self, target: str, allow_outputs: bool = False, allow_inputs: bool = False
    ) -> None:
        """
        Move constructors from cpu to the target_device.

        Sweeps through the module, looking for constructor nodes that can be moved
        to the target_device.

        A constructor node can be moved to the target_device iff all of its users
        can also be moved (tested by cannot_be_moved). Otherwise, all dependent
        constructor nodes won't be moved.

        - target: target device type
        - allow_outputs: allow outputs to be moved
        - allow_inputs: allow inputs to be moved
        """

        self.target = target
        self.allow_inputs = allow_inputs
        self.allow_outputs = allow_outputs

        assert isinstance(target, str), (
            "target should be a string representing the device type. "
            f"Got: {type(target).__name__}"
        )

    def allow_cpu_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node that returns a tensor on the target device may have
        cpu tensors as input.
        """
        return node.target in (
            torch.ops.aten.index.Tensor,
            torch.ops.aten.index_put.default,
            torch.ops.aten.index_put_.default,
            torch.ops.aten.copy.default,
            torch.ops.aten.copy_.default,
            torch.ops.aten.slice_scatter.default,
        )

    def is_on_target_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node is on the target device.
        """
        node_device = self.get_node_device(node)
        return node_device is not None and node_device.type == self.target

    def is_cpu_scalar_tensor(self, node: fx.Node) -> bool:
        """
        Returns whether a node is a cpu scalar tensor.
        """
        device = self.get_node_device(node)
        is_cpu = device is not None and device.type == "cpu"
        ten = node.meta.get("val")
        is_scalar = isinstance(ten, torch.Tensor) and len(ten.size()) == 0
        return is_cpu and is_scalar

    def all_inputs_are_cpu_scalar_or_on_target_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node's inputs are either cpu scalar tensors or
        on the target device.
        """
        inputs = (
            inp
            for inp in itertools.chain(node.args, node.kwargs.values())
            if isinstance(inp, fx.Node)
        )
        return all(
            self.is_cpu_scalar_tensor(inp) or self.is_on_target_device(inp)
            for inp in inputs
        )

    def cannot_be_moved(self, node: fx.Node) -> bool:
        """
        Returns whether a node can be moved to the target device.

        If this function returns False, it means that this node and all of its users
        won't be moved into the target device.
        """
        if node.target == "output":
            return not self.allow_outputs

        if not (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace in ("prims", "aten")
        ):
            return True

        if is_index_put_and_requires_h2d_sync_for_gpu_value(node):
            return True

        return False

    def get_node_device(self, node: fx.Node) -> torch.device | None:
        """
        Get the device of a node.
        """
        ten = node.meta.get("val")
        return None if not isinstance(ten, torch.Tensor) else ten.device

    def get_cpu_indeg_count(self, graph: fx.Graph) -> dict[fx.Node, int]:
        """
        Get the number of cpu inputs to a node
        """
        cpu_indeg: dict[fx.Node, int] = Counter()

        for node in graph.nodes:
            cpu_count = 0

            def add_cpu_inp(node):
                nonlocal cpu_count
                device = self.get_node_device(node)
                cpu_count += device is not None and device.type == "cpu"

            pytree.tree_map_only(fx.Node, add_cpu_inp, (node.args, node.kwargs))

            # pyrefly: ignore [redundant-condition]
            if cpu_count:
                cpu_indeg[node] = cpu_count

        return cpu_indeg

    def __call__(self, graph: fx.Graph) -> None:
        target_devices = OrderedSet[torch.device]()
        constructors = []
        cpu_placeholders: OrderedSet[fx.Node] = OrderedSet()

        for node in graph.nodes:
            device = self.get_node_device(node)
            if device and device.type == self.target:
                target_devices.add(device)

            if (
                self.allow_inputs
                and node.op == "placeholder"
                and self.is_cpu_scalar_tensor(node)
            ):
                cpu_placeholders.add(node)
                constructors.append(node)
                continue

            if not (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace in ("prims", "aten")
            ):
                continue

            if not torch._subclasses.fake_tensor._is_tensor_constructor(node.target):
                continue

            if node.kwargs.get("device") != torch.device("cpu"):
                continue

            constructors.append(node)

        # not handling multiple target devices initially
        if not constructors or len(target_devices) != 1:
            return

        movable_constructors = self.find_movable_constructors(graph, constructors)

        target_device = next(iter(target_devices))
        movable_cpu_placeholders = movable_constructors & cpu_placeholders
        if movable_cpu_placeholders:
            node = next(iter(reversed(movable_cpu_placeholders)))
            last_node = node
            unsqueezed_nodes = []
            for elem in movable_cpu_placeholders:
                with graph.inserting_after(last_node):
                    unsqueezed_nodes.append(
                        graph.call_function(torch.ops.aten.unsqueeze.default, (elem, 0))
                    )
                    last_node = unsqueezed_nodes[-1]
            with graph.inserting_after(last_node):
                cpu_concat = graph.call_function(
                    torch.ops.aten.cat.default, (unsqueezed_nodes,)
                )
                last_node = cpu_concat
            with graph.inserting_after(last_node):
                gpu_concat = graph.call_function(
                    torch.ops.prims.device_put.default,
                    (cpu_concat, target_device, True),
                )
                last_node = gpu_concat
            with graph.inserting_after(last_node):
                gpu_split = graph.call_function(
                    torch.ops.aten.unbind.int, (gpu_concat,)
                )
                last_node = gpu_split
            for idx, node in enumerate(movable_cpu_placeholders):
                with graph.inserting_after(last_node):
                    gpu_node = graph.call_function(operator.getitem, (gpu_split, idx))
                    node.replace_all_uses_with(
                        gpu_node,
                        lambda x: x
                        not in [cpu_concat, gpu_concat, gpu_split, gpu_node]
                        + unsqueezed_nodes
                        and x.target != torch.ops.aten.copy_.default,
                    )
                    last_node = gpu_node

                # noop elimination if there are other device_put for gpu_node to
                # target device. Alternatively, we could just move the other device_put
                # earlier in the graph, but that is not supported in fx graph yet.
                noop_device_puts = [
                    user
                    for user in gpu_node.users
                    if user.target is torch.ops.prims.device_put.default
                    and user.args[1] == target_device
                ]
                for noop in noop_device_puts:
                    noop.replace_all_uses_with(gpu_node)
                    graph.erase_node(noop)

        movable_constructors -= movable_cpu_placeholders
        for node in movable_constructors:
            kwargs = node.kwargs.copy()
            kwargs["device"] = target_device
            node.kwargs = kwargs

    def find_movable_constructors(
        self, graph: fx.Graph, constructors: list[fx.Node]
    ) -> OrderedSet[fx.Node]:
        """
        Starting from the cpu constructors, iterate through the graph and test that all of their
        downstream uses can safely be moved to cpu.
        """
        cpu_indeg: dict[fx.Node, int] = self.get_cpu_indeg_count(graph)

        # which constructors cannot be moved to gpu
        cannot_move_to_gpu = OrderedSet[fx.Node]()

        # For any node in the graph, which constructors does it have a dependency on
        constructor_dependencies: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(
            OrderedSet
        )

        # if a cpu node has a dependency on two different cpu constructors,
        # then if either constructor cannot be moved to gpu, the other cannot as well.
        # In this case any node with a dependency on one will have a dependency on the other
        equal_constructor_sets: dict[fx.Node, OrderedSet[fx.Node]] = {
            c: OrderedSet([c]) for c in constructors
        }

        def make_dependencies_equivalent(
            set1: OrderedSet[fx.Node], set2: OrderedSet[fx.Node]
        ) -> OrderedSet[fx.Node]:
            # could use union find but not worth complexity here
            set1.update(set2)
            for obj in set1:
                equal_constructor_sets[obj] = set1
            return set1

        queue: list[fx.Node] = list(constructors)

        for c in queue:
            constructor_dependencies[c].add(c)

        while queue:
            node = queue.pop()
            dependencies = constructor_dependencies[node]

            for user in node.users:
                if self.cannot_be_moved(user):
                    cannot_move_to_gpu.update(dependencies)
                    break

                # this node was used on a op which takes in multiple devices and output a gpu
                # tensor. we can convert its cpu input to gpu without making further changes
                if self.allow_cpu_device(user) and self.is_on_target_device(user):
                    del cpu_indeg[user]
                elif (
                    self.allow_inputs
                    and self.all_inputs_are_cpu_scalar_or_on_target_device(user)
                ):
                    # this node takes only cpu scalar tensors or gpu tensors as inputs
                    # and outputs a gpu tensor. we can convert its cpu scalar inputs to gpu
                    # without making further changes
                    del cpu_indeg[user]
                else:
                    # otherwise, we should continue look at its downstream uses
                    cpu_indeg[user] -= 1
                    if cpu_indeg[user] == 0:
                        del cpu_indeg[user]
                        queue.append(user)

                unioned_set = make_dependencies_equivalent(
                    dependencies, constructor_dependencies[user]
                )
                constructor_dependencies[user] = unioned_set

        for node in cpu_indeg:
            if constructor_dependencies[node]:
                cannot_move_to_gpu.update(constructor_dependencies[node])

        all_cannot_move_to_gpu = cannot_move_to_gpu.copy()
        for constructor in cannot_move_to_gpu:
            all_cannot_move_to_gpu.update(equal_constructor_sets[constructor])

        return OrderedSet(constructors) - all_cannot_move_to_gpu


def move_constructors_to_gpu(graph: fx.Graph) -> None:
    """
    Moves intermediary tensors which are constructed on the cpu to gpu when safe
    """

    # cudagraph does not support cpu tensors. In this pass, we update the graph
    # by explicitly moving cpu scalar tensors to gpu when profitable, relying on
    # graph partition to split off this data copy, and cudagraphifying
    # the remaining gpu ops.
    allow_inputs_outputs = bool(
        torch._inductor.config.triton.cudagraphs
        and torch._inductor.config.graph_partition
    )
    ConstructorMoverPass(
        get_gpu_type(),
        allow_inputs=allow_inputs_outputs,
        allow_outputs=allow_inputs_outputs,
    )(graph)
