import torch
from torch.fx import GraphModule
import torch._dynamo as torchdynamo
from torch.nn.utils.fusion import fuse_conv_bn_weights

from torch.ao.quantization.fx.prepare import (
    is_activation_post_process_node,
)
from .qconfig_mapping import QConfigMapping
from .backend_config import BackendConfig
from .fx import prepare
from .quantize_fx import _convert_to_reference_decomposed_fx

from typing import Tuple, Any, Dict
from collections import OrderedDict
import operator

# TODO: longer term, don't need to retrace or parse the string
# we should have node.meta["nn_module_stack"] that store the dict
def _infer_nn_stack_trace_and_append_on_mode(m, gm, args_as_list):
    trace_func, guards = torchdynamo.export(
        m,
        args_as_list,
        aten_graph=True,
        tracing_mode="real"
    )
    reset_metadata = {}
    for node in trace_func.graph.nodes:
        nn_module_stack = {}
        if (stack_trace := node.meta.get("stack_trace")) is not None:
            for line in stack_trace.split("\n"):
                if line.startswith("Module stack:"):
                    mod_trace = eval(line.replace("Module stack:", ""))  # pyre-ignore
                    nn_module_stack = {"nn_module_stack": mod_trace}
        reset_metadata[node.name] = nn_module_stack

    for n in gm.graph.nodes:
        if (meta := reset_metadata.get(n.name)):
            n.meta.update(meta)

# TODO: longer term, this should happen in the dynamo stack as well
def _get_renamed_nn_module_stack(nn_module_stack):
    # initialize with top level parent scope
    nn_module_stack_renamed = OrderedDict([("", None)])
    if nn_module_stack:
        # Rename module_key, e.g. "self_layer1_1__conv1" to "self.layer1.1._conv1", for easier downstream parsing
        prev_key = ""
        for key, value in nn_module_stack.items():
            if not prev_key:
                if key.startswith("self_"):
                    new_key = key[5:]
                    prev_key = new_key
            else:
                new_key = prev_key + "." + key[len(prev_key) + 6 :]
            nn_module_stack_renamed[new_key] = value
            prev_key = new_key
    return nn_module_stack_renamed

def _get_tensor_constant_from_node(node, m):
    if node is None:
        return None
    assert node.op == "get_attr"
    return getattr(m, node.target)

# fuse conv bn weights, inplace modification of the graph_module and graph
def _fuse_conv_bn_(m: GraphModule) -> None:
    for n in m.graph.nodes:
        if n.op != "call_function" or n.target != torch.ops.aten.native_batch_norm.default:
            continue
        bn_op = n
        n = bn_op.args[0]
        if n.op != "call_function" or n.target != torch.ops.aten.convolution.default:
            continue
        conv_op = n

        # conv weight
        conv_w = _get_tensor_constant_from_node(conv_op.args[1], m)
        # conv bias
        conv_b = _get_tensor_constant_from_node(conv_op.args[2], m)
        transpose = conv_op.args[6]

        # bn weight
        bn_w = _get_tensor_constant_from_node(bn_op.args[1], m)
        # bn bias
        bn_b = _get_tensor_constant_from_node(bn_op.args[2], m)
        # bn running mean
        bn_rm = _get_tensor_constant_from_node(bn_op.args[3], m)
        # bn running variance
        bn_rv = _get_tensor_constant_from_node(bn_op.args[4], m)
        bn_eps = bn_op.args[7]

        fused_weight, fused_bias = fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False)

        # update the weight and bias for conv
        conv_args = list(conv_op.args)
        # calling data since the fused_weight and fused_bias are nn.Parameter
        weight_attr_name = conv_args[1].target
        setattr(m, weight_attr_name, fused_weight)
        if conv_args[2] is not None:
            bias_attr_name = conv_args[2].target
        else:
            bias_attr_name = weight_attr_name + "_bias"
            with m.graph.inserting_before(conv_op):
                get_bias_node = m.graph.get_attr(bias_attr_name)
            conv_args[2] = get_bias_node
        setattr(m, bias_attr_name, fused_bias)
        conv_op.args = tuple(conv_args)

        # native_batch_norm has 3 outputs, we expect getitem calls on the output
        # and we want to replace the uses of getitem 0 with the output of conv
        #
        # Before:
        # conv -> bn - (first output) -> users1
        #          \ - (second output) -> users2
        #          \ - (third output) -> users3
        # After:
        # conv -> (first output) -> users1
        #       bn -
        #          \ - (second output) -> users2
        #          \ - (third output) -> users3
        # if users2 and users3 are empty then bn will be removed through dead code elimination

        for user in bn_op.users:
            if user.op != "call_function" or user.target != operator.getitem or user.args[1] != 0:
                continue
            user.replace_all_uses_with(conv_op)
    m.graph.eliminate_dead_code()
    m.recompile()

def _rearrange_weight_observer_for_addmm(
    model: GraphModule,
) -> None:
    """
    before:
         weight - t - observer \
          input - observer - addmm
    after:
         weight - observer - t \
           input - observer - addmm
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.target != torch.ops.aten.addmm.default:
            continue
        addmm = node
        maybe_weight_obs = addmm.args[2]
        if not is_activation_post_process_node(maybe_weight_obs, named_modules):
            continue
        transpose_node = maybe_weight_obs.args[0]
        if transpose_node.target != torch.ops.aten.t.default:
            continue
        # swap the order of transpose and observation

        maybe_weight_obs.replace_input_with(transpose_node, transpose_node.args[0])
        # remove the transpose node
        with model.graph.inserting_after(maybe_weight_obs):
            args = list(transpose_node.args)
            args[0] = maybe_weight_obs
            new_transpose_node = model.graph.create_node(
                "call_function",
                torch.ops.aten.t.default,
                tuple(args),
                transpose_node.kwargs
            )
        addmm.replace_input_with(maybe_weight_obs, new_transpose_node)

    model.graph.eliminate_dead_code()
    model.graph.lint()

def prepare_pt2e(
    model: torch.fx.GraphModule,
    qconfig_mapping: QConfigMapping,
    example_inputs: Tuple[Any, ...],
    backend_config: BackendConfig,
):
    # _infer_nn_stack_trace_and_append_on_mode(model, model, example_inputs)
    # TODO: move this information to fx node itself
    node_name_to_scope: Dict[str, Tuple[str, type]] = {}
    for n in model.graph.nodes:
        renamed_stack = _get_renamed_nn_module_stack(n.meta.get("nn_module_stack", None))
        current_scope = list(renamed_stack.items())[-1]
        node_name_to_scope[n.name] = current_scope

    # TODO: check qconfig_mapping to make sure conv and bn are both configured
    # to be quantized before fusion
    # TODO: (maybe) rewrite this with subgraph_rewriter
    _fuse_conv_bn_(model)
    model = prepare(
        model,
        qconfig_mapping,
        False,  # is_qat
        node_name_to_scope,
        example_inputs,
        backend_config=backend_config
    )

    # TODO: remove hack when we have better support for pattern matching
    # move around the observer for addmm
    _rearrange_weight_observer_for_addmm(model)
    return model

def convert_pt2e(
    model: GraphModule
):
    return _convert_to_reference_decomposed_fx(model)
