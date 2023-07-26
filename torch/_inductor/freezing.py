import itertools

import weakref
from typing import List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import dynamo_timed
from torch._functorch.compile_utils import fx_graph_cse

from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.fx_passes.post_grad import view_to_reshape

from . import config

aten = torch.ops.aten

aten = torch.ops.aten
prims = torch.ops.prims


def replace_node_with_constant(gm, node, constant):
    g = gm.graph

    if not hasattr(gm, "_frozen_param_count"):
        gm._frozen_param_count = 0

    i = gm._frozen_param_count

    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


def replace_params_with_constants(gm, flat_params, fw_metadata) -> List[int]:
    """
    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.
    Returns a list of indices representing the input parameters that were not converted to constants.
    """
    params = [node for node in gm.graph.nodes if node.op == "placeholder"]
    fake_inp_nodes = params[: len(params)]
    g = gm.graph
    preserved_arg_indices = []
    aliased_input_args = [
        out_info.base_idx
        for out_info in fw_metadata.output_info
        if out_info.base_idx is not None
    ]
    for i, (real_input, node) in enumerate(zip(flat_params, fake_inp_nodes)):
        if i in fw_metadata.mutated_inp_indices or aliased_input_args:
            preserved_arg_indices.append(i)
            continue
        replace_node_with_constant(gm, node, real_input)
    # add on non param inputs
    preserved_arg_indices.extend(range(len(flat_params), len(params)))
    # is this necessary ?
    gm.recompile()
    return preserved_arg_indices


class ConstantFolder(torch.fx.Interpreter):
    def __init__(self, gm, skip_constructors=False):
        super().__init__(gm)
        self.node_replacements = {}
        self.unknown_value = object()
        self.skip_constructors = skip_constructors

    def run_node(self, node):
        aten = torch.ops.aten
        args, kwargs = self.fetch_args_kwargs_from_env(node)

        if node.target == "output":
            return super().run_node(node)

        flattened_inputs = pytree.tree_flatten((args, kwargs))[0]
        if self.unknown_value in flattened_inputs:
            return self.unknown_value

        # TODO - fix errors with this
        if (
            node.op == "call_function"
            and node.target == aten._efficientzerotensor.default
        ):
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        # TODO - more complicated strategy
        if (
            self.skip_constructors
            and node.op != "get_attr"
            and not any(isinstance(e, torch.Tensor) for e in flattened_inputs)
        ):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return self.unknown_value

        out = super().run_node(node)

        # TODO - remove constant from node_replacement when it has no uses
        if node.op != "get_attr" and isinstance(out, torch.Tensor):
            self.node_replacements[node] = out

        return out

    def run(self):
        env = {}
        for n in self.module.graph.nodes:
            if n.op == "placeholder":
                env[n] = self.unknown_value
        return super().run(initial_env=env)


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm):
    cf = ConstantFolder(gm, skip_constructors=True)
    cf.run()

    for node, constant in cf.node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 0:
            delattr(gm, node.target)
            erased_params.append(node)

    for node in erased_params:
        gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: List[torch._subclasses.FakeTensor],
) -> Tuple[torch.fx.GraphModule, List[int]]:
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
    # We have convert conv's weight to channels last which may meet error for .view
    # when doing fake_tensor_prop. So we need to convert view to reshape first.
    # See the details in fx_codegen_and_compile of compile_fx.py.
    view_to_reshape(aot_autograd_gm)

    fw_metadata = torch._guards.TracingContext.get().fw_metadata
    params_flat = torch._guards.TracingContext.get().params_flat
    assert fw_metadata is not None and params_flat is not None

    preserved_arg_indices = replace_params_with_constants(
        aot_autograd_gm, params_flat, fw_metadata
    )

    # TODO - further restrict cse ? right now needed to dedup aliasing ops
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    aot_autograd_gm.recompile()

    aot_example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
    freezing_passes(aot_autograd_gm, aot_example_inputs)

    # TODO - apply legalization in pattern matcher
    torch.fx.passes.tools_common.legalize_graph(aot_autograd_gm)
    constant_fold(aot_autograd_gm)
    # invalidate nn Modules
    if config.freezing_discard_parameters:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)
    return aot_autograd_gm, preserved_arg_indices


class ErasedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        return super().__new__(cls, elem.to(device="meta"))

    def __init__(self, elem, name: Optional[str], mod):
        self.erased_name = name
        self.owning_mod_ref = weakref.ref(mod)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        erased_tensors = [
            e
            for e in pytree.tree_flatten((args, kwargs))[0]
            if isinstance(e, ErasedTensor)
        ]
        assert len(erased_tensors) > 0
        e = erased_tensors[0]

        raise RuntimeError(
            f"Trying to Run Pytorch Eager Module After Dynamo Freezing. "
            "The original parameters have been discarded for memeory efficiency. "
            f"Found in op {func} for erased parameter {e.erased_name} of {e.owning_mod_ref()}"
        )


@torch.utils._python_dispatch._disable_current_modes()
def invalidate_eager_modules():
    for mod in torch._guards.TracingContext.get().module_context.nn_modules.values():
        if not isinstance(mod, torch.nn.Module):
            continue

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


@torch.utils._python_dispatch._disable_current_modes()
def discard_traced_gm_params(mod):
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


def enforce_output_layout(gm):
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


def enforce_as_strided_input_layout(gm):
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


@dynamo_timed
def convert_conv_weights_to_channels_last(gm):
    """
    Convert 4d convolution weight tensor to channels last format.

    This pass is performed before freezing so the added nodes can be constant
    folded by freezing.
    """
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
