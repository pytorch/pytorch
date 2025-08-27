# mypy: allow-untyped-defs
from __future__ import annotations

import itertools
import logging
import weakref
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code
from torch._functorch.aot_autograd import MutationType
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.constant_folding import constant_fold, replace_node_with_constant
from torch._inductor.freezing_utils import enter_freezing, record_has_frozen_params
from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.fx_passes.post_grad import view_to_reshape

from . import config


aten = torch.ops.aten
prims = torch.ops.prims

log = logging.getLogger(__name__)


def replace_params_with_constants(
    gm: torch.fx.GraphModule,
    flat_params: list[Any],
    fw_metadata: torch._functorch.aot_autograd.ViewAndMutationMeta,
) -> list[int]:
    """
    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.
    Returns a list of indices representing the input parameters that were not converted to constants.
    """
    params = gm.graph.find_nodes(op="placeholder")
    fake_inp_nodes = params[: len(params)]
    preserved_arg_indices = []
    aliased_input_args = [
        out_info.base_idx
        for out_info in fw_metadata.output_info
        if out_info.base_idx is not None
    ]

    # TODO (tmanlaibaatar) figure out why this is different
    # from mutated_inp_runtime_indices
    mutated_inps = [
        i
        for i, m in enumerate(fw_metadata.input_info)
        if m.mutation_type
        in (MutationType.MUTATED_IN_GRAPH, MutationType.MUTATED_OUT_GRAPH)
    ]

    static_indices_new = []
    static_indices_offset = 0
    for i, (real_input, node) in enumerate(zip(flat_params, fake_inp_nodes)):
        if i in mutated_inps or i in aliased_input_args:
            preserved_arg_indices.append(i)
            if i in fw_metadata.static_input_indices:
                new_static_index = i - static_indices_offset
                static_indices_new.append(new_static_index)
        else:
            replace_node_with_constant(gm, node, real_input)
            static_indices_offset += 1
    # add on non param inputs
    preserved_arg_indices.extend(range(len(flat_params), len(params)))
    # is this necessary ?
    fw_metadata.static_input_indices = static_indices_new
    gm.recompile()
    return preserved_arg_indices


def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: list[torch._subclasses.FakeTensor],
) -> tuple[torch.fx.GraphModule, list[int]]:
    """
    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.

    Returns:
        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices
        of the inputs that were preserved (not turned into constants).
    """
    with enter_freezing():
        return _freeze(dynamo_gm, aot_autograd_gm, example_inputs)


def _freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: list[torch._subclasses.FakeTensor],
) -> tuple[torch.fx.GraphModule, list[int]]:
    # We have convert conv's weight to channels last which may meet error for .view
    # when doing fake_tensor_prop. So we need to convert view to reshape first.
    # See the details in fx_codegen_and_compile of compile_fx.py.
    view_to_reshape(aot_autograd_gm)

    if tracing_context := torch._guards.TracingContext.try_get():
        fw_metadata = tracing_context.fw_metadata
        assert tracing_context.params_flat_unwrap_subclasses is not None
        params_flat = tracing_context.params_flat_unwrap_subclasses
        assert fw_metadata is not None and params_flat is not None

        preserved_arg_indices = replace_params_with_constants(
            aot_autograd_gm, params_flat, fw_metadata
        )
    else:
        inputs = aot_autograd_gm.graph.find_nodes(op="placeholder")
        preserved_arg_indices = list(range(len(inputs)))

    # TODO - further restrict cse ? right now needed to dedup aliasing ops
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    aot_autograd_gm.recompile()

    aot_example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
    freezing_passes(aot_autograd_gm, aot_example_inputs)

    constant_fold(aot_autograd_gm)
    # invalidate nn Modules
    if config.freezing_discard_parameters:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)

    log.debug(
        "%s", lazy_format_graph_code("FROZEN GRAPH", aot_autograd_gm, colored=True)
    )

    record_has_frozen_params(aot_autograd_gm)
    return aot_autograd_gm, preserved_arg_indices


class ErasedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        return super().__new__(cls, elem.to(device="meta"))

    def __init__(self, elem, name: Optional[str], mod) -> None:
        self.erased_name = name
        self.owning_mod_ref = weakref.ref(mod)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        erased_tensors = [
            e
            for e in pytree.arg_tree_leaves(*args, **kwargs)
            if isinstance(e, ErasedTensor)
        ]
        assert len(erased_tensors) > 0
        e = erased_tensors[0]

        raise RuntimeError(
            f"Trying to run Pytorch Eager Module after Dynamo Freezing. "
            "The original parameters have been discarded for memory efficiency. "
            f"Found in op {func} for erased parameter {e.erased_name} of {e.owning_mod_ref()}"
        )


def invalidate_eager_modules():
    with torch.utils._python_dispatch._disable_current_modes():
        for (
            mod
        ) in torch._guards.TracingContext.get().module_context.nn_modules.values():
            if not isinstance(mod, torch.nn.Module):
                continue

            for attr_name, tensor in list(
                itertools.chain(
                    mod.named_parameters(recurse=False),
                    mod.named_buffers(recurse=False),
                )
            ):
                with torch._dispatch.python.no_python_dispatcher():
                    e_t = ErasedTensor(tensor, attr_name, mod)
                if isinstance(tensor, torch.nn.Parameter):
                    e_t.requires_grad_(True)
                    e_t._is_param = True
                setattr(mod, attr_name, e_t)


def discard_traced_gm_params(mod: torch.fx.GraphModule):
    with torch.utils._python_dispatch._disable_current_modes():
        for attr_name, tensor in list(
            itertools.chain(
                mod.named_parameters(recurse=False), mod.named_buffers(recurse=False)
            )
        ):
            with torch._dispatch.python.no_python_dispatcher():
                e_t = ErasedTensor(tensor, attr_name, mod)
            if isinstance(tensor, torch.nn.Parameter):
                e_t.requires_grad_(True)
                e_t._is_param = True
            setattr(mod, attr_name, e_t)


def enforce_output_layout(gm: torch.fx.GraphModule):
    """
    Make sure the output node's layout does not change due to compiler optimizations
    by adding aten.as_strided nodes with the expected strides.

    Only used for inference so we can assume all graph outputs are model outputs.
    """
    *_, output_node = gm.graph.nodes
    out_list = output_node.args[0]
    with gm.graph.inserting_before(output_node):
        for n in out_list:
            if not isinstance(
                n.meta["val"], torch.Tensor
            ) or not torch._prims_common.is_non_overlapping_and_dense(n.meta["val"]):
                continue

            # add a node to enforce eager layout
            ft = n.meta["val"]
            new_node = gm.graph.call_function(
                prims.inductor_force_stride_order.default, (n, ft.stride())
            )

            # can not call
            # n.replace_all_uses_with(new_node)
            # since it will replace the usage of n in new_node itself.
            output_node.replace_input_with(n, new_node)

    gm.graph.lint()
    gm.recompile()


def enforce_as_strided_input_layout(gm: torch.fx.GraphModule):
    """
    Make sure the as_strided node's input's layout does not change due to compiler
    optimizations, because the as_strided strides info depends on input tensor stride info.
    """

    as_strided_ops = [
        torch.ops.aten.as_strided.default,
        torch.ops.aten.as_strided_.default,
        torch.ops.aten.as_strided_scatter.default,
    ]
    strided_nodes = [n for n in gm.graph.nodes if n.target in as_strided_ops]
    for n in strided_nodes:
        with gm.graph.inserting_before(n):
            # add a node to enforce eager layout
            ft = n.args[0].meta["val"]
            new_node = gm.graph.call_function(
                prims.inductor_force_stride_order.default, (n.args[0], ft.stride())
            )
            n.replace_input_with(n.args[0], new_node)

    gm.graph.lint()
    gm.recompile()


def convert_conv_weights_to_channels_last(gm: torch.fx.GraphModule):
    """
    Convert 4d convolution weight tensor to channels last format.

    This pass is performed before freezing so the added nodes can be constant
    folded by freezing.
    """
    with dynamo_timed("convert_conv_weights_to_channels_last"):
        convs = [n for n in gm.graph.nodes if n.target == aten.convolution.default]
        for conv in convs:
            weight_node = conv.args[1]
            if len(weight_node.meta["val"].size()) != 4 or weight_node.meta[
                "val"
            ].is_contiguous(memory_format=torch.channels_last):
                # not a 4d tensor or already channels last, skip
                continue

            with gm.graph.inserting_before(conv):
                new_node = gm.graph.call_function(
                    aten.clone.default,
                    (weight_node,),
                    {"memory_format": torch.channels_last},
                )
                conv.replace_input_with(weight_node, new_node)

        enforce_as_strided_input_layout(gm)
        enforce_output_layout(gm)
