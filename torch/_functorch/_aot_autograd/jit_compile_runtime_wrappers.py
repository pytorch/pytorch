# mypy: allow-untyped-defs
"""
Functions in this module do most of the "work" of AOTAutograd.
An aot_dispatch_* function:
- Takes in the input flat_fn, flat_args, and some metadata
- Runs a set of pre compile wrappers (e.g. argument deduping)
- Runs the actual compiler
- Wraps the returned callable in a set of post compile wrappers
- Returns the wrapped callable and metadata.
"""

import copy
import dataclasses
import itertools
import logging
import operator
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import detect_fake_mode, lazy_format_graph_code
from torch._guards import CompileContext, TracingContext
from torch._logging import getArtifactLogger, trace_structured
from torch._subclasses import FakeTensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from torch.fx.graph_module import GraphModule
from torch.fx.passes._tensorify_python_scalars import tensorify_python_scalars
from torch.multiprocessing.reductions import StorageWeakRef
from torchgen.utils import dataclass_repr

from .. import config
from .autograd_cache import (
    AOTAutogradCache,
    AOTAutogradCacheEntry,
    CompiledBackward,
    CompiledForward,
    should_use_remote_autograd_cache,
)
from .dispatch_and_compile_graph import (
    aot_dispatch_autograd_graph,
    aot_dispatch_base_graph,
)
from .logging_utils import track_graph_compiling
from .runtime_wrappers import (
    AOTDedupeWrapper,
    AOTDispatchAutograd,
    AOTDispatchSubclassWrapper,
    AOTSyntheticBaseWrapper,
    AutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    DebugAssertWrapper,
    EffectTokensWrapper,
    FakifiedOutWrapper,
    FunctionalizedRngRuntimeWrapper,
    make_runtime_safe,
    post_compile,
    pre_compile,
    RuntimeWrapper,
)
from .schemas import AOTConfig, MutationType, ViewAndMutationMeta
from .subclass_utils import compute_inner_mutated_inp_indices_from_subclass_meta
from .utils import (
    _get_symint_hints,
    contain_metadata_mutation_ops,
    get_cuda_generator_meta_val,
    make_boxed_func,
    strict_zip,
    unlift_tokens,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

zip = strict_zip

log = logging.getLogger(__name__)
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")

aten = torch.ops.aten

# Returns a Callable and a ViewAndMutationMeta.
# Currently, only export needs the ViewAndMutationMeta after this function.
DispatchReturn = tuple[Callable, ViewAndMutationMeta]


def _create_wrappers_for_dispatch(needs_autograd: bool) -> list[CompilerWrapper]:
    """
    Wrappers that run on every dispatch function
    """
    return [AOTDedupeWrapper(), AOTSyntheticBaseWrapper(trace_joint=needs_autograd)]


# Export's dispatching logic is unique in a few ways: it only needs the "graph"
# bits of aot_autograd, and doesn't need to do any specific wrapping.
def aot_dispatch_export(
    flat_fn: Callable,
    flat_args: list[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
    needs_autograd: bool,
) -> DispatchReturn:
    wrappers = _create_wrappers_for_dispatch(needs_autograd)
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers,
        flat_fn,
        flat_args,
        aot_config,
        fw_metadata=fw_metadata,
    )
    if needs_autograd and not aot_config.pre_dispatch:
        graph, _, _ = aot_dispatch_autograd_graph(
            flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
        )
    else:
        graph, _, _ = aot_dispatch_base_graph(
            flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
        )

    # NB: the wrappers that run in pre_compile for export are
    # either a no-op, because they're not needed, or will raise a runtime error,
    # since they don't support export.
    # We still run these wrappers to make sure that they're not needed pre compile,
    # but we technically don't need to run them post compile at all here.
    compiled_fn, fw_metadata = post_compile(
        wrappers, graph, aot_config, runtime_metadata=fw_metadata
    )

    # Therefore, since no wrapperes run, we don't get back a callable - we get back the raw fx graph
    # (either a joint or an inference-only graph)
    assert isinstance(compiled_fn, torch.fx.GraphModule)
    return compiled_fn, fw_metadata


def aot_dispatch_base(
    flat_fn,
    flat_args: list[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> DispatchReturn:
    """
    Handles functions that don't need autograd. Runs wrappers and compiles with fw_compiler.
    """
    wrappers = _create_wrappers_for_dispatch(needs_autograd=False)
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers, flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )
    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(  # type: ignore[misc]
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )
    # Save the forward_graph_str right after aot_dispatch_base_graph,
    # to save in the cache
    aot_forward_graph_str = None
    if aot_config.cache_info is not None:
        aot_forward_graph_str = fw_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        )

    fakified_out_wrapper = FakifiedOutWrapper()
    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = fakified_out_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )
    functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper()
    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = functionalized_rng_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )

    if aot_config.enable_log:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "torch._functorch.config",
                "encoding": "string",
            },
            payload_fn=lambda: torch._functorch.config.get_config_copy(),
        )

    disable_amp = torch._C._is_any_autocast_enabled()
    context = torch._C._DisableAutocast if disable_amp else nullcontext

    with context(), track_graph_compiling(aot_config, "inference"):
        compiler = (
            aot_config.inference_compiler
            if aot_config.inference_compiler is not None
            else aot_config.fw_compiler
        )

        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.fw_metadata = (
                fw_metadata
                if maybe_subclass_meta is None
                else maybe_subclass_meta.fw_metadata
            )

        with TracingContext.report_output_strides() as fwd_output_strides:
            fake_mode = detect_fake_mode()
            if fake_mode is not None and fake_mode.shape_env is not None:
                assert isinstance(fw_module, GraphModule)
                tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
            compiled_fw = compiler(fw_module, updated_flat_args)

        if fakified_out_wrapper.needs_post_compile:
            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    # However, RuntimeWrapper does not expect the rng offsets in the
    # output. So, we have to create another wrapper and take out the offset. As
    # a result, we have to account for not boxed_call compilers as well.
    if not hasattr(compiled_fw, "_boxed_call"):
        compiled_fw = make_boxed_func(compiled_fw)

    # Create a wrapper to set up the rng functionalize and fakified out bits
    compiled_fw = functionalized_rng_wrapper.post_compile(
        compiled_fw, aot_config, runtime_metadata=fw_metadata
    )
    cache_info = aot_config.cache_info
    if cache_info is not None:
        if fw_key := getattr(compiled_fw, "_fx_graph_cache_key", None):
            time_taken_ns = time.time_ns() - cache_info.start_time_ns
            entry = AOTAutogradCacheEntry(
                compiled_fw=CompiledForward(fw_key),
                compiled_bw=None,
                aot_joint_graph_str=None,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=None,
                runtime_metadata=fw_metadata,
                dispatch_wrappers=wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=None,
                indices_of_inps_to_detach=[],
                forward_time_taken_ns=time_taken_ns,
                backward_time_taken_ns=0,
            )
            AOTAutogradCache.save(
                cache_info.cache_key, entry, remote=should_use_remote_autograd_cache()
            )

    compiled_fw = fakified_out_wrapper.post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    compiled_fw = EffectTokensWrapper().post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    # Why do we need to pass in num_fw_outs_saved_for_bw?
    # See Note: [Partitioner handling for Subclasses, Part 2]
    compiled_fw = AOTDispatchSubclassWrapper(
        trace_joint=False,
        # TODO: once we use pre_compile this will be flat_fn at the top of this function
        fw_only=None,
        maybe_subclass_meta=maybe_subclass_meta,
        num_fw_outs_saved_for_bw=None,
    ).post_compile(
        compiled_fw,
        aot_config,  # not used
        runtime_metadata=fw_metadata,
    )

    if not hasattr(compiled_fw, "_boxed_call"):
        compiled_fw = make_boxed_func(compiled_fw)

    compiled_fn = RuntimeWrapper(
        indices_of_inps_to_detach=[],
        trace_joint=False,
        disable_amp=disable_amp,
    ).post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    compiled_fn = post_compile(
        wrappers, compiled_fn, aot_config, runtime_metadata=fw_metadata
    )
    return compiled_fn


def collect_fw_donated_buffer_idxs(
    fw_ins: list[Optional[FakeTensor]],
    user_fw_outs: list[Optional[FakeTensor]],
    bw_outs: list[Optional[FakeTensor]],
    saved_tensors: list[FakeTensor],
) -> list[int]:
    """
    Checks if the saved tensors are donated buffers, which means a saved tensor is not
    an alias of any tensors in fw_ins, user_fw_outs, and bw_outs.
    """

    storage_refs = set()
    for t in itertools.chain(fw_ins, user_fw_outs, bw_outs):
        # Only access storage if a tensor has storage (not sparse)
        if t is not None and isinstance(t, FakeTensor) and not is_sparse_any(t):
            storage_refs.add(StorageWeakRef(t.untyped_storage()))

    num_saved_tensor = len(saved_tensors)
    donated_buffer_idxs = []
    for i in range(num_saved_tensor):
        t = saved_tensors[i]
        if (
            t is not None
            and not is_sparse_any(t)
            and StorageWeakRef(t.untyped_storage()) not in storage_refs
        ):
            donated_buffer_idxs.append(i)

    return donated_buffer_idxs


def collect_bw_donated_buffer_idxs(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    fw_metadata: ViewAndMutationMeta,
) -> list[int]:
    """
    Collects backward donated buffer indexes from fw_module and bw_module.
    """

    # [Note: Metadata mutation in proxy tracing]
    # node.meta["val"] is a snapshot of the tensor value when tracing a graph,
    # instead of the final state after the graph has run. node.meta["val"] is
    # not updated even if later there is a metadata mutation op.
    # See: https://github.com/pytorch/pytorch/pull/141308#issuecomment-2495798947
    #
    # Currently, metadata mutation op happens only for sacrificial parameter
    # specifically the `set_` op. This motivates banning metadata mutation from
    # proxy tracing.
    #
    # Since node.meta["val"] is used to detect donated buffer, we return an empty
    # list if there exists metadata mutation op.
    if contain_metadata_mutation_ops(fw_module) or contain_metadata_mutation_ops(
        bw_module
    ):
        return []

    fw_ins = fw_module.graph.find_nodes(op="placeholder")
    bw_outs = next(reversed(bw_module.graph.find_nodes(op="output"))).args[0]
    fw_outs = next(reversed(fw_module.graph.find_nodes(op="output"))).args[0]

    fw_ins = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in fw_ins
    ]
    fw_outs = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in fw_outs
    ]
    bw_outs = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in bw_outs
    ]

    user_fw_outs = fw_outs[: fw_metadata.num_forward]
    saved_tensors = fw_outs[fw_metadata.tensors_saved_for_backwards_slice]

    fw_donated_buffer = collect_fw_donated_buffer_idxs(
        fw_ins,
        user_fw_outs,
        bw_outs,
        saved_tensors,
    )

    assert fw_metadata.num_symints_saved_for_bw is not None
    return [fw_metadata.num_symints_saved_for_bw + i for i in fw_donated_buffer]


@dataclasses.dataclass
class InvokeSubgraphHopGraphs:
    """
    A data structure to hold all the information needed to partition the
    `joint_hop_gm` and joint graph and the restitch the `new_fw_hop_gm` and
    `new_bw_hop_gm` into the bigger `joint_gm`.
    """

    # To avoid re-partitioning subgraphs
    partitioning_done: bool = False
    old_num_fw_outputs: Optional[int] = None
    old_num_fw_inputs: Optional[int] = None

    new_fw_hop_gm: Optional[torch.fx.GraphModule] = None
    new_bw_hop_gm: Optional[torch.fx.GraphModule] = None
    new_num_sym_nodes: Optional[int] = None
    new_num_saved_nodes: Optional[int] = None


def run_joint_graph_passes_on_hops(
    joint_gm: torch.fx.GraphModule,
    joint_inputs: Any,
    aot_config: AOTConfig,
) -> torch.fx.GraphModule:
    """
    This pass runs the joint graph passes on the HOP graph. In torch.compile, we
    typically have many passes which work on the joint graph and then end with a
    partitioner.


    The partitioner part is quite mechanical to handle. HOP have their own
    forward and backward graph. The process can be broken into following steps

    1) Get a `joint_hop_gm` from the `fw_hop_gm` and `bw_hop_gm`
    2) Run joint graph passes on the `joint_hop_gm` to get `new_fw_hop_gm` and `new_bw_hop_gm`
    3) Stitch the `new_fw_hop_gm` and `new_bw_hop_gm` back into the `joint_gm`.

    The terminology used in the code is
    `joint_graph/joint_gm` : Refers to the main graph. This may contain many HOPs which have their own `hop_graph`
    `fw_hop_graph/fw_hop_gm` : Refers to the forward graph associated with a HOP.
    `bw_hop_graph/bw_hop_gm` : Refers to the backward graph associated with a HOP.
    `joint_hop_graph/joint_hop_gm` : Refers to the subgraph associated with the HOP like invoke_subgraph.
    `new_fw_hop_graph/new_fw_hop_gm` : Refers to the forward graph after partitioning is applied to `joint_hop_gm`.
    `new_bw_hop_graph/new_bw_hop_gm` : Refers to the backward graph after partitioning is applied to `joint_hop_gm`.

    NB: This pass works for invoke_subgraph today because we took extra care in
    the Autograd.Dispatch key of invoke_subgraph to vastly simplify Step 1.
    """
    from torch._higher_order_ops import invoke_subgraph

    def num_outputs(mod):
        return len(mod.graph.find_nodes(op="output")[0].args[0])

    def num_inputs(mod):
        return len(mod.graph.find_nodes(op="placeholder"))

    def prepare_for_partitioner(mod, num_primals, num_fw_outputs):
        # min-cut partitioner requires the placeholders to have primals and
        # tangents string in the node.name. The signature of the joint graph is
        # (*primals, *tangents)

        # We also have to update the output signature which is right now
        # (*grads, *fw_outs) and we have to change to (*fw_outs, *grads) for the
        # partitioner to work.
        new_graph = torch.fx.Graph()
        env = {}

        primals_counter = itertools.count(0)
        tangents_counter = itertools.count(0)

        for idx, node in enumerate(mod.graph.nodes):
            if node.op == "placeholder":
                if idx < num_primals:
                    env[node] = new_graph.placeholder(
                        f"primals_{next(primals_counter)}"
                    )
                else:
                    env[node] = new_graph.placeholder(
                        f"tangents_{next(tangents_counter)}"
                    )
                env[node].meta = copy.copy(node.meta)
            elif node.op == "output":
                # Reverse the (*grads, *fw_outs) to (*fw_outs, *grads)
                # The reason for having the reversed signature in the first
                # place is to simplify step 3.
                old_outputs = node.args[0]
                new_outputs = (
                    *old_outputs[-num_fw_outputs:],
                    *old_outputs[:-num_fw_outputs],
                )
                new_outputs = [env[n] if n else None for n in new_outputs]
                new_graph.output(tuple(new_outputs))
            else:
                env[node] = new_graph.node_copy(node, lambda n: env[n])
                env[node].meta = copy.copy(node.meta)

        new_graph.lint()

        out = torch.fx.GraphModule(joint_gm, new_graph)
        return out

    new_hop_graphs: dict[str, InvokeSubgraphHopGraphs] = defaultdict(
        lambda: InvokeSubgraphHopGraphs()
    )

    # Step 1 - Get a `joint_hop_gm` from the `fw_hop_gm` and `bw_hop_gm` This is
    # easy to do for `invoke_subgraph` HOP. During the Autograd dispatch key
    # tracing, we have put the joint_hop_graph in the backward hop graph itself.
    # So to recover the joint_hop_gm, we just have to look at the backward
    # HOP graphs.
    # So we will merge step 1 and step 2 in this next section

    # Save the fw and bwd hop nodes. We will later in-place modify the graph
    # using these nodes.
    fw_hop_nodes = []
    bw_hop_nodes = []

    for node in joint_gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is invoke_subgraph
            and isinstance(node.args[1], str)
        ):
            identifier = (
                node.args[1].replace("___forward", "").replace("___backward", "")
            )

            # NB: This is done in a separate if else condition because we early
            # return if the partitioning_done is True.
            if node.args[1].startswith("___forward"):
                fw_hop_nodes.append(node)
            elif node.args[1].startswith("___backward"):
                bw_hop_nodes.append(node)

            # If partitioning already done for this identifier, skip. This saves
            # redundant joint graph passes for same subgraphs.
            if new_hop_graphs[identifier].partitioning_done:
                continue

            hop_gm = getattr(joint_gm, node.args[0].target)
            assert isinstance(hop_gm, torch.fx.GraphModule)

            if node.args[1].startswith("___forward"):
                # Collect some information from the forward hop graph
                new_hop_graphs[identifier].old_num_fw_inputs = num_inputs(hop_gm)
                new_hop_graphs[identifier].old_num_fw_outputs = num_outputs(hop_gm)
            elif node.args[1].startswith("___backward"):
                num_fw_inputs = new_hop_graphs[identifier].old_num_fw_inputs
                assert num_fw_inputs is not None
                num_fw_outputs = new_hop_graphs[identifier].old_num_fw_outputs
                assert num_fw_outputs is not None

                # Step 1) - Get the `joint_hop_gm`. As mentioned earlier, the
                # backward graph is the joint graph.
                joint_hop_gm = hop_gm

                # Prepare the graph for the partitioner
                joint_hop_gm = prepare_for_partitioner(
                    joint_hop_gm, num_fw_inputs, num_fw_outputs
                )

                # Step 2) and 3) - Run joint graph passes and partitioner
                new_fw_hop_gm, new_bw_hop_gm = aot_config.partition_fn(
                    joint_hop_gm, [], num_fwd_outputs=num_fw_outputs
                )

                # Save the new forward and backward graph modules
                new_hop_graphs[identifier].new_fw_hop_gm = new_fw_hop_gm
                new_hop_graphs[identifier].new_bw_hop_gm = new_bw_hop_gm

                # Save the number of symints and saved tensors
                new_fw_out_nodes = new_fw_hop_gm.graph.find_nodes(op="output")[0].args[
                    0
                ]
                extra_outputs = new_fw_out_nodes[num_fw_outputs:]
                symint_outputs = [n for n in extra_outputs if is_sym_node(n)]

                new_hop_graphs[identifier].new_num_sym_nodes = len(symint_outputs)
                new_hop_graphs[identifier].new_num_saved_nodes = len(
                    extra_outputs
                ) - len(symint_outputs)

                new_hop_graphs[identifier].partitioning_done = True

    if not new_hop_graphs:
        return joint_gm

    # Step 3) Restitch the new fw and bw graphs back into the main graph.
    #
    # This is a very mechanical process. There are a quite a few pieces that we
    # need to connect together to make it work. Lets try to understand the
    # problem statement first.
    #
    # For the forward graph, the signature of the old_fw_hop_gm is
    #   inputs - (*primals)
    #   outputs - (*fw_outs)
    # Now the signature of the new_fw_hop_gm is
    #   inputs - (*primals)     -- This is same
    #   outputs - (*fw_outs, *saved_tensors)    - This is different
    # At a high level, this is an easy transformation, in the new graph we just
    # have to replace the old_fw_hop_gm with the new_fw_hop_gm. Everything else
    # falls into place, because the input signature (i.e. args) is same. And
    # even though output signature is different, fw_outs are still at the same
    # indexes as before. So the forward of the `joint_gm` works nicely.
    #
    # Now, lets look at the backward hop graph. Old signature
    #   inputs - (*primals, *tangents)
    #   outputs - (*grad_outs, *fw_outs)
    # New signature
    #   inputs - (*saved_tensors, *tangents) -- Different
    #   outputs - (*grad_outs)  -- Different
    # Here both input and output signature change. The output signature handling
    # is quite easy because the grads_out are sitting at the right place, so we
    # dont have to do anything.
    #
    # For the input signature, we have to collect the saved tensors from the
    # corresponding forward graph output. We collect all saved_tensors when we
    # see the forward graph, and save it into a map and then later use it during
    # the backward.

    # The stack of fw_nodes for invoke_subgraph HOP. There is an implicit
    # assumption about the graph structure, i.e., if we have hop1, hop2, hop3,
    # ... in the forward part of the joint graph, we will have .., hop3, hop2,
    # hop1 order for the backward. This structure allows us to just use a stack
    # to collect all the information that we need to pass from the forward hop
    # node to the corresponding backward node.
    old_fw_hop_nodes_stack = []
    # Collect the saved tensor nodes that we need to pass as inputs to the
    # backward hop.
    fw_node_to_saved_tensors_map = {}

    already_added_new_hop_mods = set()

    def add_new_hop_gm(new_subgraph_mod, name):
        new_subgraph_attr_name = f"{name}_post_graph"
        if new_subgraph_attr_name in already_added_new_hop_mods:
            return new_subgraph_attr_name

        joint_gm.register_module(new_subgraph_attr_name, new_subgraph_mod)
        already_added_new_hop_mods.add(new_subgraph_attr_name)
        return new_subgraph_attr_name

    def propagate_meta_info(new_hop_gm, new_call_function_node, old_call_function_node):
        # Copy all the fields from the old call_function node. And then override
        # the `val` meta field with the outputs of new_hop_gm.
        new_call_function_node.meta = copy.copy(old_call_function_node.meta)

        output = new_hop_gm.graph.find_nodes(op="output")[0]
        out_example_vals = [n.meta["val"] if n else None for n in output.args[0]]
        new_call_function_node.meta["val"] = tuple(out_example_vals)

    for fw_node in fw_hop_nodes:
        # Insert the new_fw_hop_gm. This is straightforward. Get the
        # new_fw_hop_gm, insert the hop_gm as a get_attr fw_node, and then
        # add a call_function fw_node. Additionally, also use getitem
        # call_functions to collect the saved_tensor nodes

        identifier = (
            fw_node.args[1].replace("___forward", "").replace("___backward", "")
        )
        new_fw_hop_gm = new_hop_graphs[identifier].new_fw_hop_gm
        assert new_fw_hop_gm is not None

        old_num_fw_outputs = new_hop_graphs[identifier].old_num_fw_outputs
        new_num_sym_nodes = new_hop_graphs[identifier].new_num_sym_nodes
        new_num_saved_nodes = new_hop_graphs[identifier].new_num_saved_nodes
        assert old_num_fw_outputs is not None
        assert new_num_sym_nodes is not None
        assert new_num_saved_nodes is not None
        total_outputs = old_num_fw_outputs + new_num_saved_nodes + new_num_sym_nodes

        extra_fw_outputs = []

        # Insert the new_fw_hop_gm into the joint_gm
        with joint_gm.graph.inserting_after(fw_node):
            new_fw_mod_attr_name = add_new_hop_gm(new_fw_hop_gm, fw_node.args[1])
            new_fw_mod_attr = joint_gm.graph.get_attr(new_fw_mod_attr_name)

        # new_hop_fw_gm output signature is (*fw_outs, *saved_tensors)
        with joint_gm.graph.inserting_after(new_fw_mod_attr):
            new_fw_node = joint_gm.graph.call_function(
                the_function=invoke_subgraph,
                args=(
                    new_fw_mod_attr,
                    new_fw_mod_attr_name,
                    fw_node.args[2],
                ),
            )
            propagate_meta_info(new_fw_hop_gm, new_fw_node, fw_node)

        # old_num_fw_outputs = (*fw_outs)
        # new_num_fw_outputs = (*fw_outs, *saved_tensors, *sym_nodes)
        with joint_gm.graph.inserting_after(new_fw_node):
            for fw_out_idx in range(old_num_fw_outputs, total_outputs):
                saved_tensor_node = joint_gm.graph.call_function(
                    the_function=operator.getitem, args=(new_fw_node, fw_out_idx)
                )
                saved_tensor_node.meta = copy.copy(new_fw_node.meta)
                saved_tensor_node.meta["val"] = new_fw_node.meta["val"][fw_out_idx]
                extra_fw_outputs.append(saved_tensor_node)

        fw_node.replace_all_uses_with(new_fw_node)
        joint_gm.graph.erase_node(fw_node)

        # Save the saved_tensors info for the fw_node. This will be used
        # to form the inputs for the backward hop.
        old_fw_hop_nodes_stack.append(fw_node)
        fw_node_to_saved_tensors_map[fw_node] = (identifier, extra_fw_outputs)

    for bw_node in bw_hop_nodes:
        # Get the saved_tensors from the forward graph and find the new
        # tangents, and replace the old bw hop with the new bw hop.
        identifier = (
            bw_node.args[1].replace("___forward", "").replace("___backward", "")
        )
        new_bw_hop_gm = new_hop_graphs[identifier].new_bw_hop_gm
        assert new_bw_hop_gm is not None

        # Prepare the operands for the bwd graph
        # Old bw graph signature : (*primals, *tangents)
        # New signature will be : (*sym_nodes, *saved_tensors, *tangents)
        # We have already collected the saved_tensors in the forward hop processing.

        assert len(old_fw_hop_nodes_stack)
        fw_hop_node = old_fw_hop_nodes_stack.pop()
        (
            fw_hop_node_identifier,
            extra_fw_outputs,
        ) = fw_node_to_saved_tensors_map[fw_hop_node]
        assert fw_hop_node_identifier == identifier

        # extra_fw_outputs are in the order (*saved_nodes, *sym_nodes).
        # Partitioner has this quirk where the backward wants sym_nodes
        # first. So extract the sym and saved nodes.
        num_sym_nodes = new_hop_graphs[fw_hop_node_identifier].new_num_sym_nodes
        num_saved_nodes = new_hop_graphs[fw_hop_node_identifier].new_num_saved_nodes
        assert num_sym_nodes is not None
        assert num_saved_nodes is not None
        saved_tensor_nodes = extra_fw_outputs[:num_saved_nodes]
        sym_nodes = extra_fw_outputs[num_saved_nodes:]

        num_primals = new_hop_graphs[identifier].old_num_fw_inputs
        assert num_primals is not None
        tangents = list(bw_node.args[2][num_primals:])
        operands = sym_nodes + saved_tensor_nodes + tangents

        # Insert the new_bw_hop_gm into the joint_gm
        with joint_gm.graph.inserting_after(bw_node):
            new_bw_mod_attr_name = add_new_hop_gm(new_bw_hop_gm, bw_node.args[1])
            new_bw_mod_attr = joint_gm.graph.get_attr(new_bw_mod_attr_name)

        with joint_gm.graph.inserting_after(new_bw_mod_attr):
            new_bw_node = joint_gm.graph.call_function(
                the_function=invoke_subgraph,
                args=(
                    new_bw_mod_attr,
                    new_bw_mod_attr_name,
                    tuple(operands),
                ),
            )
            propagate_meta_info(new_bw_hop_gm, new_bw_node, bw_node)

        bw_node.replace_all_uses_with(new_bw_node)
        joint_gm.graph.erase_node(bw_node)

    joint_gm.graph.eliminate_dead_code()
    joint_gm.graph.lint()
    joint_gm.recompile()
    return joint_gm


def aot_dispatch_autograd(
    flat_fn,
    flat_args: list[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> DispatchReturn:
    """
    Autograd logic. Generates a joint graph, partitions it, manipulates the input with various wrappers,
    and returns a wrapped torch.autograd.Function with a forward and backward.
    """
    wrappers = _create_wrappers_for_dispatch(needs_autograd=True)
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers,
        flat_fn,
        flat_args,
        aot_config,
        fw_metadata=fw_metadata,
    )

    fw_metadata.deterministic = torch.are_deterministic_algorithms_enabled()
    fx_g, joint_inputs, maybe_subclass_meta = aot_dispatch_autograd_graph(
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    # Copied from aot_dispatch_autograd_graph.
    disable_amp = torch._C._is_any_autocast_enabled()
    joint_graph_str = None
    if aot_config.enable_log:
        aot_joint_log.info(
            "%s",
            lazy_format_graph_code(
                "Joint graph",
                fx_g,
                aot_config.aot_id,
                include_stride=True,
                include_device=True,
                colored=True,
            ),
        )
        joint_graph_str = fx_g.print_readable(
            print_output=False, include_stride=True, include_device=True
        )
        trace_structured(
            "aot_joint_graph",
            payload_fn=lambda: joint_graph_str,
        )

    with torch.no_grad():
        inner_meta = (
            fw_metadata
            if maybe_subclass_meta is None
            else maybe_subclass_meta.fw_metadata
        )
        with track_graph_compiling(aot_config, "joint"):
            # See Note: [Partitioner handling for Subclasses, Part 1]
            # See Note: [Recomputing subclass mutation handling]
            mutated_inp_runtime_indices = (
                compute_inner_mutated_inp_indices_from_subclass_meta(
                    fw_metadata, inner_meta
                )
            )
            num_tokens = len(fw_metadata.tokens)
            num_mutated_inp_runtime_indices = len(mutated_inp_runtime_indices)
            num_inner_fwd_outputs = (
                num_mutated_inp_runtime_indices
                + inner_meta.num_outputs
                + inner_meta.num_intermediate_bases
                + inner_meta.num_outputs_rng_offset
                + num_tokens  # See Note [Side-Effectful Tokens in AOTAutograd]
            )
            fake_mode = detect_fake_mode()
            fx_g = run_joint_graph_passes_on_hops(fx_g, joint_inputs, aot_config)

            # TODO(anijain2305) - Add tensorify_python_scalars to the HOP graph passes.
            if fake_mode is not None and fake_mode.shape_env is not None:
                tensorify_python_scalars(fx_g, fake_mode.shape_env, fake_mode)

            fw_module, bw_module = aot_config.partition_fn(
                fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
            )
            rng_states = [
                n
                for n in fw_module.graph.find_nodes(op="placeholder")
                if "fwd_rng_state" in n.name
            ]
            fw_metadata.num_graphsafe_rng_states = len(rng_states)
            if rng_states:
                fw_metadata.graphsafe_rng_state_index = (
                    rng_states[0].meta["val"].device.index
                )

            # See Note [Side-Effectful Tokens in AOTAutograd]
            if config.unlift_effect_tokens and (
                num_tokens > 0 or fw_metadata.num_backward_tokens > 0
            ):
                unlift_tokens(fw_module, fw_metadata, aot_config, bw_module)

                num_inner_fwd_outputs -= num_tokens
                joint_inputs = (
                    joint_inputs[0][num_tokens:],
                    joint_inputs[1],
                )

            fw_outs = next(iter(fw_module.graph.find_nodes(op="output"))).args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)
            symint_outs_saved_for_bw = [
                n for n in fw_outs_saved_for_bw if is_sym_node(n)
            ]
            fw_metadata.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            inner_meta.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            num_symints_saved_for_bw = len(symint_outs_saved_for_bw)

            if torch._functorch.config.donated_buffer:
                fw_metadata.bw_donated_idxs = collect_bw_donated_buffer_idxs(
                    fw_module,
                    bw_module,
                    inner_meta,
                )
                inner_meta.bw_donated_idxs = fw_metadata.bw_donated_idxs

        if aot_config.enable_log:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "torch._functorch.config",
                    "encoding": "string",
                },
                payload_fn=lambda: torch._functorch.config.get_config_copy(),
            )
            aot_graphs_log.info(
                "aot_config id: %s, fw_metadata=%s, inner_meta=%s",
                str(aot_config.aot_id),
                str(fw_metadata),
                str(inner_meta),
            )

        # Note [Detaching inputs that never need gradients]
        # See https://github.com/pytorch/pytorch/issues/97745
        # Suppose we have a function like this that we want to compile:
        #
        # def f(x, y):
        #     return torch.mul(x, y.detach())
        #
        # What gradients should we compute for x and y?
        # By default, AOTAutograd will compute a gradient for **every** input that requires gradients,
        # and so we'll compute:
        #    x_grad_input = y
        #    y_grad_input = None
        # Does this preserve the semantics of eager mode?
        # Unfortunately, no.
        # Doing the above will cause autograd to **continue** to backprop the autograd tape
        # that was generated from constructing y.
        #
        # This is **different** from what would have happened in eager mode.
        # In eager mode, if we backprop through the output of this function, autograd will only traverse
        # the bit of the autograd tape corresponding to "x".
        # In particular, if a user had previously backpropped through y's autograd tape,
        # And then they try to backprop through the output of the above function,
        # then we'll hit the dreaded "Trying to backward through the graph a second time" error.
        #
        # You might think: If autograd sees that a gradient is None, shouldn't it stop early,
        # instead of continuing the backprop through the ancestors of that node in the graph?
        #
        # Autograd has two passes:
        # (1) a first pass that traverses the autograd graph and figures out which nodes need to be executed
        # (2) a second pass that actually goes ahead and executes each node when it becomes ready,
        #     propagating gradients
        # By the time we're executing a node and we see that it produces a None, the set of nodes to execute
        # is already locked-in.
        #
        # The fix: instead, we can recognize statically that the graph we're compiling will never contribute
        # gradients to y, and prevent autograd from trying to traverse y's autograd tape at all.
        # We can do this by manually detach'ing y before sending it through the `CompiledFunction`.
        #
        # Note that this solution is not bulletproof.
        # It's possible to construct a case where eager may or may not have have tried to autograd through y,
        # depending on the actual grad_outputs that were passed in during the backward.
        # There is no easy fix for this: the simplest fix would be to run with `retain_graph=True`,
        # allowing autograd to re-use the graph.
        #
        # An example of this case is:
        # def f(x):
        #     return x.detach() * 2, x * 3
        # If we were to only backprop through outs[0], in eager, we would stop
        # If we backward only on the first output, we shouldn't send a grad through x.
        # But the custom autograd function doesn't know that: it will materialize zero grads for x * 3
        # and we will end up with a zero grad at x.
        # If we later backprop through the second output, this will also require backprop'ing through x.
        # Meaning we'll need to use `retain_graph=True` to be able to backprop through x the second time.
        _indices_of_inps_to_detach: list[int] = []

        # reversed() since we expect output at end of graph
        bw_output = next(reversed(bw_module.graph.find_nodes(op="output")))
        bw_outs: Sequence[torch.fx.Node] = bw_output.args[0]  # type: ignore[assignment]

        # TODO: we should apply the below "detach inputs if their gradients are statically known to be None"
        # optimization even if we have subclass inputs/outputs (we do not handle this today).
        # Computing which our our inputs get None gradients is a bit more complicated,
        # if any of our inputs are subclasses. Why?
        # (a) we need to make sure that we call .detach() on the input subclasses, since autograd sees subclasses.
        # (b) The grad_outputs that we AOT computed in our backward graph are the desugared tensor tensors,
        #     so we need to figure out which subclass fw inputs they map to.
        if maybe_subclass_meta is None:
            num_backward_tokens: int = inner_meta.num_backward_tokens
            assert (
                len(bw_outs)
                == len(fw_metadata.input_info)
                + inner_meta.num_outputs_rng_offset
                + num_backward_tokens
            )
            bw_outs_no_rng_no_tokens = bw_outs
            if (inner_meta.num_outputs_rng_offset + num_backward_tokens) > 0:
                bw_outs_no_rng_no_tokens = bw_outs[
                    : -(inner_meta.num_outputs_rng_offset + num_backward_tokens)
                ]
            assert len(bw_outs_no_rng_no_tokens) == len(fw_metadata.input_info)

            for i, (bw_out) in enumerate(bw_outs_no_rng_no_tokens):
                # If our input experiences a metadata mutation inside the graph (e.g. set_()),
                # we *must* not detach, otherwise it will be the detach'd input that gets the metadata mutation
                metadata_mutation_in_graph = (
                    fw_metadata.input_info[i].mutation_type
                    == MutationType.MUTATED_IN_GRAPH
                    and fw_metadata.input_info[i].mutates_storage_metadata
                )
                is_non_leaf = (
                    fw_metadata.input_info[i].requires_grad
                    and not fw_metadata.input_info[i].is_leaf
                )
                if bw_out is None and not metadata_mutation_in_graph and is_non_leaf:
                    _indices_of_inps_to_detach.append(i)

        fw_module_str = None
        bw_module_str = None
        if aot_config.enable_log:
            aot_graphs_log.info(
                "%s",
                lazy_format_graph_code(
                    "Forward graph",
                    fw_module,
                    aot_config.aot_id,
                    include_stride=True,
                    include_device=True,
                    colored=True,
                ),
            )
            aot_graphs_log.info(
                "%s",
                lazy_format_graph_code(
                    "Backward graph",
                    bw_module,
                    aot_config.aot_id,
                    include_stride=True,
                    include_device=True,
                    colored=True,
                ),
            )
            fw_module_str = fw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            )
            bw_module_str = bw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            )

            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aot_forward_graph_fw_metadata",
                    "encoding": "string",
                },
                payload_fn=lambda: dataclass_repr(fw_metadata),
            )
            if maybe_subclass_meta is not None:
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "aot_forward_graph_fw_subclass_metadata",
                        "encoding": "string",
                    },
                    payload_fn=lambda: dataclass_repr(maybe_subclass_meta),
                )

            trace_structured(
                "aot_forward_graph",
                payload_fn=lambda: fw_module_str,
            )
            trace_structured(
                "aot_backward_graph",
                payload_fn=lambda: bw_module_str,
            )

        # AMP is already traced out in joint graph. we do not wish to reapply it accidentally
        # in the compiler.
        with track_graph_compiling(aot_config, "forward"), torch._C._DisableAutocast():
            # flat_args at this point might still be subclasses-
            # make sure to pass the unwrapped fake tensors into the compiler!
            adjusted_flat_args = joint_inputs[0]

            fakified_out_wrapper = FakifiedOutWrapper()
            (
                fw_module,
                adjusted_flat_args,
                fw_metadata,
            ) = fakified_out_wrapper.pre_compile(
                fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
            )

            functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper(
                return_new_outs=False
            )

            if rng_states:
                index = fw_metadata.graphsafe_rng_state_index
                assert index is not None
                rng_states = [
                    get_cuda_generator_meta_val(index)
                    for _ in range(fw_metadata.num_graphsafe_rng_states)
                ]
                adjusted_flat_args.extend(rng_states)  # type: ignore[arg-type]

            (
                fw_module,
                adjusted_flat_args,
                fw_metadata,
            ) = functionalized_rng_wrapper.pre_compile(
                fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
            )
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.fw_metadata = inner_meta

            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)

            if not hasattr(compiled_fw_func, "_boxed_call"):
                compiled_fw_func = make_boxed_func(compiled_fw_func)

            if fakified_out_wrapper.needs_post_compile:
                fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

            compiled_fw_func = EffectTokensWrapper().post_compile(
                compiled_fw_func,
                aot_config,
                runtime_metadata=fw_metadata,
            )

            compiled_fw_func = AOTDispatchSubclassWrapper(
                fw_only=None,
                trace_joint=False,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
            ).post_compile(
                compiled_fw_func,
                aot_config,  # not used
                runtime_metadata=fw_metadata,
            )

            compiled_fw_func = functionalized_rng_wrapper.post_compile(
                compiled_fw_func, aot_config, runtime_metadata=fw_metadata
            )
            compiled_fw_func = fakified_out_wrapper.post_compile(
                compiled_fw_func,
                aot_config,
                runtime_metadata=fw_metadata,
            )

        # NB: It's important to compile backwards ahead of time, as this may
        # add extra guards which we need to apply to the Dynamo cache at
        # forwards
        with track_graph_compiling(aot_config, "backward"), torch._C._DisableAutocast():
            placeholder_list = fx_placeholder_vals(bw_module)

            forward_saved_for_backwards_strides = None
            if fwd_output_strides is not None:
                forward_saved_for_backwards_strides = fwd_output_strides[
                    inner_meta.tensors_saved_for_backwards_slice
                ]

            # saved activations can have different stride to eager if
            # the compiler does layout optimization. We should restride the
            # tensor passed in for compiling the backward graph using the
            # saved tensor's stride.
            for i in range(len(placeholder_list)):
                ph_arg = placeholder_list[i]
                if not isinstance(ph_arg, torch.Tensor):
                    continue

                if forward_saved_for_backwards_strides is None:
                    continue

                real_stride = None
                # Per all_args calling convention
                j = i - num_symints_saved_for_bw
                if 0 <= j < len(forward_saved_for_backwards_strides):
                    real_stride = forward_saved_for_backwards_strides[j]
                if real_stride is None:
                    continue

                # Comparing ph_arg.stride() with real_stride directly may
                # cause dynamic dimensions in ph_arg being specialized to static
                # value. Using the hints to avoid that.
                if _get_symint_hints(ph_arg.stride()) != real_stride:
                    # Note that here we use the stride of the real tensor to
                    # restride a FakeTensor. This does not cause trouble
                    # for dynamic shape since this code path only get
                    # executed if layout optimization is enabled. And we
                    # disable layout optimization for dynamic shape right
                    # now.
                    #
                    # A solution that decide stride order based on real
                    # tensor's stride and then apply that stride order to
                    # the FakeTensor does not work smoothly since some
                    # tensor's layout is not 'dense'. E.g. mixnet_l has a
                    # tensor with size [8, 64, 112, 112] and strides
                    # (2408448, 1, 21504, 192). The solution mentioned will
                    # decide a stride of (802816, 1, 7168, 64) for this
                    # tensor which is wrong.
                    placeholder_list[i] = ph_arg.as_strided(ph_arg.size(), real_stride)

            compiled_bw_func = None
            if num_symints_saved_for_bw > 0:
                try:
                    compiled_bw_func = aot_config.bw_compiler(
                        bw_module, placeholder_list
                    )
                except Exception as e:
                    exc = e
                    trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": "eager_compile_backwards_failure",
                            "encoding": "string",
                        },
                        payload_fn=lambda: "\n".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        ),
                    )
                    log.warning(
                        "failed to eagerly compile backwards for dynamic, suppressing in case backwards not needed",
                        exc_info=True,
                    )
            # Compiled autograd will run the bw_module in the backward pass,
            # so recompilation need happen anyway if the backward pass is ever
            # called.
            #
            # The reason we do the GraphModule recompilation here is because
            # the lazy recompilation will cause issue in the backward pass
            # with compiled autograd.
            #
            # Do the _LazyGraphModule.force_recompile here rather than when
            # bw_module is first generated by the partitioner because the bw_module.recompile
            # may be called in some code path later and cause the _LazyGraphModule.forward
            # becomes the lazy version again. One example is when dynamic shape is enabled
            # upfront, the bw_compiler will be called above which can cause extra
            # graph module recompilation on bw_module.
            if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
                from torch.fx._lazy_graph_module import _LazyGraphModule

                _LazyGraphModule.force_recompile(bw_module)

    saved_context = TracingContext.try_get()
    saved_compile_context = CompileContext.try_get()

    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    assert len(backward_state_indices) <= 1

    lazy_backward_info = AutogradLazyBackwardCompileInfo(
        bw_module,
        placeholder_list,
        saved_context,
        saved_compile_context,
    )

    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    try_save_cache_entry: Optional[Callable] = None

    if aot_config.cache_info is not None:
        forward_time_taken_ns = time.time_ns() - aot_config.cache_info.start_time_ns

        # NB: aot_config here is technically not needed as an argument: we could just
        # close over aot_config.cache_info, since aot_config never changes.
        # But closing over random variables is confusing IMO, so I'm leaving it.
        def try_save_cache_entry(  # noqa: F811
            compiled_bw_func, _fw_metadata, aot_config
        ):
            fw_key = getattr(compiled_fw_func, "_fx_graph_cache_key", None)
            bw_key = getattr(compiled_bw_func, "_fx_graph_cache_key", None)
            cache_info = aot_config.cache_info
            if cache_info is not None and fw_key and bw_key:
                assert forward_time_taken_ns is not None
                # TODO: technically, AOTAutograd does a *little* bit of post processing work
                # in the backward that isn't measured here. But it's small enough that it's not worth
                # the complexity of threading a bunch of times through the code, so we
                # use the compiled_bw_func's inductor compile time instead.
                # It's possible this changes in the future, in which case we should
                # update backward_time_taken_ns to be more inclusive
                backward_time_taken_ns = getattr(compiled_bw_func, "_time_taken_ns", 0)

                aot_forward_graph_str: Optional[str] = fw_module_str
                aot_backward_graph_str: Optional[str] = bw_module_str
                aot_joint_graph_str: Optional[str] = joint_graph_str
                entry = AOTAutogradCacheEntry(
                    CompiledForward(fw_key),
                    CompiledBackward(
                        bw_key,
                        backward_state_indices,
                        num_symints_saved_for_bw,
                    ),
                    aot_joint_graph_str,
                    aot_forward_graph_str,
                    aot_backward_graph_str,
                    _fw_metadata,
                    wrappers,
                    maybe_subclass_meta,
                    num_fw_outs_saved_for_bw,
                    _indices_of_inps_to_detach,
                    forward_time_taken_ns,
                    backward_time_taken_ns,
                )
                remote = should_use_remote_autograd_cache()
                AOTAutogradCache.save(cache_info.cache_key, entry, remote)

        if compiled_bw_func is not None:
            # If we already compiled it we can just run it right now without waiting
            try_save_cache_entry(compiled_bw_func, fw_metadata, aot_config)
            try_save_cache_entry = None

    compiled_fn = AOTDispatchAutograd.post_compile(
        compiled_fw_func,
        compiled_bw_func,
        maybe_subclass_meta,
        num_symints_saved_for_bw,
        backward_state_indices,
        disable_amp,
        _indices_of_inps_to_detach,
        lazy_backward_info,
        aot_config,
        fw_metadata=fw_metadata,
        try_save_cache_entry=try_save_cache_entry,
    )

    if config.debug_assert:
        flat_requires_grad: list[Optional[bool]] = [
            a.requires_grad if isinstance(a, Tensor) else None for a in flat_args
        ]
        compiled_fn = DebugAssertWrapper(
            flat_requires_grad=flat_requires_grad
        ).post_compile(compiled_fn, aot_config, runtime_metadata=fw_metadata)

    compiled_fn = post_compile(
        wrappers,
        compiled_fn,
        aot_config,
        runtime_metadata=fw_metadata,
    )
    return compiled_fn
