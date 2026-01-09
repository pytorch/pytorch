# mypy: allow-untyped-defs
"""
This module dispatches the graphs to either the forward-only or joint compilation
pathways, taking into account the AOTConfig and the collected ViewAndMutationMetadata.
"""

import contextlib
import dataclasses
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import detect_fake_mode, lazy_format_graph_code
from torch._logging import getArtifactLogger, trace_structured
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torchgen.utils import dataclass_repr

from .. import config
from .descriptors import AOTInput, BackwardTokenAOTInput
from .functional_utils import (
    assert_functional_graph,
    propagate_input_mutation_stacktraces,
)
from .graph_capture_wrappers import (
    aot_dispatch_subclass,
    create_functionalized_fn,
    create_joint,
    fn_input_mutations_to_outputs,
    fn_prepped_for_autograd,
    handle_effect_tokens_fn,
)
from .schemas import AOTConfig, FxValue, SubclassMeta, TraceFn, ViewAndMutationMeta
from .streams import assign_backward_streams, insert_backward_syncs, sync_deallocations
from .utils import (
    call_and_expect_output_descs,
    copy_fwd_metadata_to_bw_nodes,
    fn_wrappers,
    register_buffer_assignment_hook,
    root_module_when_exporting_non_strict,
    simple_wraps,
    unlift_tokens,
)


aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


def _create_graph(
    f,
    args: list[torch.Tensor],
    args_descs: Optional[
        list[AOTInput]
    ] = None,  # keep compat with old clients; maybe we should split into two impls
    *,
    aot_config: AOTConfig,
) -> torch.fx.GraphModule:
    # FunctionalTensorMode must be enabled here.
    # See Note [Accessing .grad_fn on FunctionalTensor]
    out_descs = None

    if args_descs is None:
        inner_f = f
    else:

        @simple_wraps(f)
        def inner_f(*args):
            nonlocal out_descs
            assert out_descs is None
            out, out_descs = call_and_expect_output_descs(f, args)
            return out

    if aot_config.disable_functionalization:
        ctx = contextlib.nullcontext()
    else:
        ctx = FunctionalTensorMode(  # type: ignore[assignment]
            pre_dispatch=aot_config.pre_dispatch,
            export=aot_config.is_export,
            # Allow token discovery for joint fn tracing as tokens can be used in backward.
            _allow_token_discovery=True,
        )

    with (
        enable_python_dispatcher(),
        ctx,
    ):
        fx_g = make_fx(
            inner_f,
            decomposition_table=aot_config.decompositions,
            record_module_stack=True,
            pre_dispatch=aot_config.pre_dispatch,
            _disable_torch_fn_metadata_mode=aot_config._disable_torch_fn_metadata_mode,
        )(*args)

        if args_descs is not None:
            flat_args_descs, _ = pytree.tree_flatten(args_descs)
            flat_out_descs, _ = pytree.tree_flatten(out_descs)

            # Unfortunately, flat_args_descs is not guaranteed to match the
            # number of actual arguments that show up on the FX graph.
            # Specifically, allow_token_discovery=True means that we will
            # silently add extra token arguments to the backwards graph.
            #
            # Although there are a few ways to detect what these tokens are,
            # we are going to settle for something dodgy but simple to
            # implement: match tangents_token placeholders specifically,
            # as these are the only placeholders that are created by token
            # discovery (NB: there is NO other code that treats this name
            # as load bearing, so this is a bit naughty!)
            #
            # I originally wanted to detect tokens in exactly the same way
            # that they are detected at normal runtime, but to be honest
            # the normal runtime detection is pretty strange: it seems the
            # backward tokens are not reliably at the end of the argument list
            # but *precede* the RNG arguments (I don't understand why this is
            # the case).  And in unlift_tokens, token arguments are detected
            # by seeing if they feed into an effects call!  Dastardly.  Why
            # didn't we just introduce a new type.

            i = 0
            j = 0
            for n in fx_g.graph.nodes:
                if n.op == "placeholder":
                    if n.name.startswith("tangents_token"):
                        n.meta["desc"] = BackwardTokenAOTInput(j)
                        j += 1
                    else:
                        assert i < len(flat_args_descs), (
                            (fn_wrappers(inner_f)),
                            [n for n in fx_g.graph.nodes if n.op == "placeholder"],
                            flat_args_descs,
                        )
                        n.meta["desc"] = flat_args_descs[i]
                        i += 1
                elif n.op == "output":
                    n.meta["desc"] = flat_out_descs

    return fx_g


# TODO: Refactor the following code so detach() persists item_memo
def _detach_and_copy_item_memo(t):
    detached_t = t.detach()
    if hasattr(t, "item_memo"):
        detached_t.item_memo = t.item_memo
    return detached_t


def aot_dispatch_base_graph(
    flat_fn: TraceFn,
    flat_args: list[FxValue],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[torch.fx.GraphModule, list[FxValue], list[AOTInput], Optional[SubclassMeta]]:
    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
    # The cases that aot_dispatch_base doesn't need to handle include:
    # - outputs that are aliases of graph intermediates
    # - outputs that are aliases of graph inputs
    # While cases that it does need to handle include:
    # - input mutations (including when inputs are aliases of each other)
    # - input metadata mutations
    fn_to_trace = fn_input_mutations_to_outputs(
        flat_fn,
        flat_args_descs,
        fw_metadata,
        keep_data_input_mutations=aot_config.keep_inference_input_mutations,
    )

    if aot_config.disable_functionalization:
        updated_flat_args, updated_flat_args_descs = (
            flat_args,
            flat_args_descs,
        )
    else:
        fn_to_trace, updated_flat_args, updated_flat_args_descs = (
            create_functionalized_fn(
                fn_to_trace,
                flat_args,
                flat_args_descs,
                meta=fw_metadata,
                aot_config=aot_config,
                trace_joint=False,
            )
        )

    # TODO: replace with AOTDispatchSubclassWrapper once we refactor
    # fn_input_mutations_to_outputs and create_functionalized_fn
    # into CompilerWrappers.
    (
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        updated_flat_args_subclasses_desugared_descs,
        maybe_subclass_meta,
    ) = aot_dispatch_subclass(
        fn_to_trace,
        updated_flat_args,
        updated_flat_args_descs,
        is_joint_structure=False,
        meta=fw_metadata,
        fw_only=flat_fn,
    )

    if not aot_config.disable_functionalization:
        (
            fn_to_trace,
            updated_flat_args_subclasses_desugared,
            updated_flat_args_subclasses_desugared_descs,
        ) = handle_effect_tokens_fn(
            fn_to_trace,
            updated_flat_args_subclasses_desugared,
            updated_flat_args_subclasses_desugared_descs,
            meta=fw_metadata,
            trace_joint=False,
        )

    aot_graphs_log.debug(
        "aot_config id: %s, fw_metadata=%s,subclass_metadata=%s",
        str(aot_config.aot_id),
        str(fw_metadata),
        str(maybe_subclass_meta),
    )

    # We track buffer assignments when exporting in non-strict mode.
    # (In contrast, strict mode errors on any attribute assignment.)
    mod_when_exporting_non_strict = root_module_when_exporting_non_strict(flat_fn)
    if aot_config.is_export and mod_when_exporting_non_strict is not None:
        # For any buffer that is assigned, we want to associate it to the final proxy node
        # that it is assigned to. This node can then be added as a buffer mutation output.
        assigned_buffers: dict[str, str] = {}
        hook = register_buffer_assignment_hook(
            mod_when_exporting_non_strict, assigned_buffers
        )

    fake_mode = detect_fake_mode()
    if fake_mode:
        saved_updated_flat_args_subclasses_desugared = pytree.tree_map_only(
            torch.Tensor,
            _detach_and_copy_item_memo,
            updated_flat_args_subclasses_desugared,
        )
    else:
        saved_updated_flat_args_subclasses_desugared = pytree.tree_map_only(
            torch.Tensor, lambda t: t.detach(), updated_flat_args_subclasses_desugared
        )
    saved_updated_flat_args_subclasses_desugared_descs = (
        updated_flat_args_subclasses_desugared_descs
    )

    fw_module = _create_graph(
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        updated_flat_args_subclasses_desugared_descs,
        aot_config=aot_config,
    )

    if aot_config.is_export and mod_when_exporting_non_strict is not None:
        # We update metadata to consider any assigned buffers as buffer mutations.
        i = len(dict(mod_when_exporting_non_strict.named_parameters()))
        for name, _ in mod_when_exporting_non_strict.named_buffers():
            if name in assigned_buffers and not fw_metadata.input_info[i].mutates_data:  # type: ignore[possibly-undefined]
                fw_metadata.input_info[i] = dataclasses.replace(
                    fw_metadata.input_info[i], mutates_data=True
                )
                fw_metadata.num_mutated_inp_runtime_indices += 1
            i += 1

        # We add nodes corresponding to buffer assignments as output nodes in the graph.
        add_nodes = []
        output_node = list(fw_module.graph.nodes)[-1]
        for name in assigned_buffers.values():  # type: ignore[possibly-undefined]
            for node in fw_module.graph.nodes:
                if node.name == name:
                    add_nodes.append(node)
                    node.users[output_node] = None
        output_node.args = ((*add_nodes, *output_node.args[0]),)

        hook.remove()  # type: ignore[possibly-undefined]

    # As long as we opted to remove input mutations, then
    # there should be *NO* mutating ops in the graph at this point.
    if not aot_config.disable_functionalization:
        copy_count = assert_functional_graph(fw_module.graph)
        fw_module.graph.eliminate_dead_code()
        fw_module.recompile()
        copy_count2 = assert_functional_graph(fw_module.graph)
        propagate_input_mutation_stacktraces(fw_module.graph)
        assert copy_count == copy_count2
    else:
        fw_module.graph.eliminate_dead_code()

    # See Note [Side-Effectful Tokens in AOTAutograd]
    num_tokens = len(fw_metadata.tokens)
    if num_tokens != 0 and config.unlift_effect_tokens:
        unlift_tokens(fw_module, fw_metadata, aot_config)
        saved_updated_flat_args_subclasses_desugared = (
            saved_updated_flat_args_subclasses_desugared[num_tokens:]
        )
        saved_updated_flat_args_subclasses_desugared_descs = (
            saved_updated_flat_args_subclasses_desugared_descs[num_tokens:]
        )

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
                # For more expanded output (but can't default to this because it
                # affects tests):
                # expanded_def=True
            ),
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
            "aot_inference_graph",
            payload_fn=lambda: fw_module.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
                expanded_def=True,
            ),
        )

    # TODO: should factor this into a separate function for export that always only returns just the graph.
    if aot_config.is_export:
        assert maybe_subclass_meta is None, (
            "aot_export_module does not support tensor subclass inputs for now."
        )
    return (
        fw_module,
        saved_updated_flat_args_subclasses_desugared,
        saved_updated_flat_args_subclasses_desugared_descs,
        maybe_subclass_meta,
    )


# Has the precondition that there
# are no duplicate arguments in flat_args (e.g., the same Tensor
# object never shows up twice.  However, two tensor inputs MAY alias
# the same storage, so long as they have separate TensorImpls.)
def aot_dispatch_autograd_graph(
    flat_fn: TraceFn,
    flat_args: list[Any],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[
    torch.fx.GraphModule,
    tuple[list[Any], list[Any]],
    tuple[list[AOTInput], list[AOTInput]],
    Optional[SubclassMeta],
]:
    # NB: flat_fn here is the original user function (as far as
    # aot_module_simplified is concerned)

    # traced_tangents corresponds to the set of outputs in the traced forward that should get grad_outputs in the traced backward.
    # It includes outputs of the original forward, *and* any updated inputs due to input mutations.
    # However, it does *not* include any outputs that are aliases of inputs or intermediates, or any metadata-only input mutations.
    joint_inputs = (flat_args, fw_metadata.traced_tangents)
    joint_inputs_descs = (flat_args_descs, fw_metadata.traced_tangents_descs)

    fn_prepared_for_autograd = fn_prepped_for_autograd(
        flat_fn,
        flat_args_descs,
        fw_metadata,
        aot_config,
    )
    joint_fn_to_trace = create_joint(
        fn_prepared_for_autograd, flat_args_descs, aot_config=aot_config
    )
    joint_fn_handle = joint_fn_to_trace.handle

    if aot_config.disable_functionalization:
        updated_joint_inputs, updated_joint_inputs_descs = (
            joint_inputs,
            joint_inputs_descs,
        )
    else:
        joint_fn_to_trace, updated_joint_inputs, updated_joint_inputs_descs = (
            create_functionalized_fn(
                joint_fn_to_trace,
                joint_inputs,
                joint_inputs_descs,
                meta=fw_metadata,
                aot_config=aot_config,
                trace_joint=True,
                joint_fn_handle=joint_fn_handle,
            )
        )

    # TODO: replace with AOTDispatchSubclassWrapper once we refactor
    # fn_input_mutations_to_outputs and create_functionalized_fn
    # into CompilerWrappers.
    subclass_tracing_info = aot_dispatch_subclass(
        joint_fn_to_trace,
        updated_joint_inputs,
        updated_joint_inputs_descs,
        is_joint_structure=True,
        meta=fw_metadata,
        fw_only=flat_fn,
    )

    joint_fn_to_trace = subclass_tracing_info.plain_tensor_trace_fn
    updated_joint_inputs = subclass_tracing_info.plain_tensor_args
    updated_joint_inputs_descs = subclass_tracing_info.plain_tensor_args_descs

    if not aot_config.disable_functionalization:
        (joint_fn_to_trace, updated_joint_inputs, updated_joint_inputs_descs) = (
            handle_effect_tokens_fn(
                joint_fn_to_trace,
                updated_joint_inputs,
                updated_joint_inputs_descs,
                meta=fw_metadata,
                trace_joint=True,
            )
        )

    # When we call _create_graph, this may mutate the metadata of joint
    # inputs.  But callers are expecting to get the original joint inputs.  So
    # we make aliases of all the inputs to make sure we have a copy that
    # doesn't get modified.
    #
    # This destroys requires_grad/grad_fn information.  However, backends
    # beneath AOTAutograd are indifferent to this information, so it doesn't
    # matter.

    fake_mode = detect_fake_mode()
    if fake_mode:
        saved_updated_joint_inputs = pytree.tree_map_only(
            torch.Tensor, _detach_and_copy_item_memo, updated_joint_inputs
        )
    else:
        saved_updated_joint_inputs = pytree.tree_map_only(
            torch.Tensor, lambda t: t.detach(), updated_joint_inputs
        )
    maybe_subclass_meta = subclass_tracing_info.maybe_subclass_meta

    fx_g = _create_graph(
        joint_fn_to_trace,
        updated_joint_inputs,
        updated_joint_inputs_descs,
        aot_config=aot_config,
    )

    # Redundant with the check above, but worth having in case tracing introduced
    # a fake tensor. Unlikely.
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(fx_g)

    # Have to copy before eliminate_dead_code otherwise the
    # fw node match might be erased
    copy_fwd_metadata_to_bw_nodes(fx_g)

    # After copying metadata, assign streams to gradient accumulation nodes
    assign_backward_streams(fx_g)

    # Insert syncs for newly assigned backward streams
    insert_backward_syncs(fx_g)

    # Sync deallocations for tensors where the stream w/ their last usage
    # is distinct from their allocation strea
    sync_deallocations(fx_g)

    fx_g.graph.eliminate_dead_code()
    if not aot_config.disable_functionalization:
        # There should be *NO* mutating ops in the graph at this point.
        assert_functional_graph(fx_g.graph)

    fx_g.recompile()

    # TODO: in AOTAutograd, we create metadata like _indices_of_inps_to_detach to detect
    # when we need to manually detach() some inputs in the forward.
    # Higher order ops might eventually need to do the same.
    if aot_config.is_export:
        assert maybe_subclass_meta is None, (
            "aot_export_module does not support tensor subclass inputs for now."
        )
    return (
        fx_g,
        saved_updated_joint_inputs,
        updated_joint_inputs_descs,
        maybe_subclass_meta,
    )
