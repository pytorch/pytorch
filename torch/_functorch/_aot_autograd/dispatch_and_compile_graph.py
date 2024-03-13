"""
This module dispatches the graphs to either the forward-only or joint compilation
pathways, taking into account the AOTConfig and the collected ViewAndMutationMetadata.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import lazy_format_graph_code
from torch._logging import getArtifactLogger, trace_structured
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from .functional_utils import (
    assert_functional_graph,
    propagate_input_mutation_stacktraces,
)
from .schemas import AOTConfig, SubclassMeta, ViewAndMutationMeta
from .traced_function_transforms import (
    aot_dispatch_subclass,
    create_functionalized_fn,
    create_joint,
    fn_input_mutations_to_outputs,
    fn_prepped_for_autograd,
)

aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


def _create_graph(f, args, *, aot_config: AOTConfig) -> torch.fx.GraphModule:
    # FunctionalTensorMode must be enabled here.
    # See Note [Accessing .grad_fn on FunctionalTensor]
    with enable_python_dispatcher(), FunctionalTensorMode(
        pre_dispatch=aot_config.pre_dispatch, export=aot_config.is_export
    ):
        fx_g = make_fx(
            f,
            decomposition_table=aot_config.decompositions,
            record_module_stack=True,
            pre_dispatch=aot_config.pre_dispatch,
        )(*args)

    return fx_g


def aot_dispatch_base_graph(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> Union[Callable, Tuple[Callable, List[Any], Optional[SubclassMeta]]]:
    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
    # The cases that aot_dispatch_base doesn't need to handle include:
    # - outputs that are aliases of graph intermediates
    # - outputs that are aliases of graph inputs
    # While cases that it does need to handle include:
    # - input mutations (including when inputs are aliases of each other)
    # - input metadata mutations
    fn_to_trace = fn_input_mutations_to_outputs(
        flat_fn,
        fw_metadata,
        keep_data_input_mutations=aot_config.keep_inference_input_mutations,
    )

    fn_to_trace, updated_flat_args = create_functionalized_fn(
        fn_to_trace,
        flat_args,
        meta=fw_metadata,
        aot_config=aot_config,
        trace_joint=False,
    )

    (
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        maybe_subclass_meta,
    ) = aot_dispatch_subclass(
        fn_to_trace,
        updated_flat_args,
        is_joint_structure=False,
        meta=fw_metadata,
        fw_only=flat_fn,
    )

    fw_module = _create_graph(
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        aot_config=aot_config,
    )

    # As long as we opted to remove input mutations, then
    # there should be *NO* mutating ops in the graph at this point.
    copy_count = assert_functional_graph(fw_module.graph)

    fw_module.graph.eliminate_dead_code()
    fw_module.recompile()

    copy_count2 = assert_functional_graph(fw_module.graph)
    propagate_input_mutation_stacktraces(fw_module.graph)

    assert copy_count == copy_count2

    if aot_config.enable_log:
        aot_graphs_log.info(
            "%s", lazy_format_graph_code("Forward graph", fw_module, aot_config.aot_id)
        )
        trace_structured(
            "aot_forward_graph",
            payload_fn=lambda: fw_module.print_readable(print_output=False),
        )

    # TODO: should factor this into a separate function for export that always only returns just the graph.
    if aot_config.is_export:
        assert (
            maybe_subclass_meta is None
        ), "aot_export_module does not support tensor subclass inputs for now."
        return fw_module
    return fw_module, list(updated_flat_args_subclasses_desugared), maybe_subclass_meta


# Has the precondition that there
# are no duplicate arguments in flat_args (e.g., the same Tensor
# object never shows up twice.  However, two tensor inputs MAY alias
# the same storage, so long as they have separate TensorImpls.)
def aot_dispatch_autograd_graph(
    flat_fn,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> Union[Callable, Tuple[Callable, List[Any], Optional[SubclassMeta]]]:
    # traced_tangents corresponds to the set of outputs in the traced forward that should get grad_outputs in the traced backward.
    # It includes outputs of the original forward, *and* any updated inputs due to input mutations.
    # However, it does *not* include any outputs that are aliases of inputs or intermediates, or any metadata-only input mutations.
    traced_tangents = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        fw_metadata.traced_tangents,
    )

    joint_inputs = (flat_args, traced_tangents)

    fn_prepared_for_autograd = fn_prepped_for_autograd(
        flat_fn,
        fw_metadata,
    )
    joint_fn_to_trace = create_joint(fn_prepared_for_autograd, aot_config=aot_config)

    joint_fn_to_trace, updated_joint_inputs = create_functionalized_fn(
        joint_fn_to_trace,
        joint_inputs,
        meta=fw_metadata,
        aot_config=aot_config,
        trace_joint=True,
    )

    subclass_tracing_info = aot_dispatch_subclass(
        joint_fn_to_trace,
        updated_joint_inputs,
        is_joint_structure=True,
        meta=fw_metadata,
        fw_only=flat_fn,
    )

    joint_fn_to_trace = subclass_tracing_info.plain_tensor_trace_fn
    updated_joint_inputs = subclass_tracing_info.plain_tensor_args
    maybe_subclass_meta = subclass_tracing_info.maybe_subclass_meta

    fx_g = _create_graph(joint_fn_to_trace, updated_joint_inputs, aot_config=aot_config)

    # There should be *NO* mutating ops in the graph at this point.
    assert_functional_graph(fx_g.graph)

    # Redundant with the check above, but worth having in case tracing introduced
    # a fake tensor. Unlikely.
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(fx_g)
    fx_g.graph.eliminate_dead_code()
    fx_g.recompile()
    # TODO: in AOTAutograd, we create metadata like _indices_of_inps_to_detach to detect
    # when we need to manually detach() some inputs in the forward.
    # Higher order ops might eventually need to do the same.
    if aot_config.is_export:
        assert (
            maybe_subclass_meta is None
        ), "aot_export_module does not support tensor subclass inputs for now."
        return fx_g
    return fx_g, updated_joint_inputs, maybe_subclass_meta
