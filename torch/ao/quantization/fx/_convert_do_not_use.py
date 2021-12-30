from typing import Any, Dict, List, Optional
import torch
from torch.fx import (
    GraphModule,
)
from torch.fx.graph import (
    Graph,
    Node,
)
from ..qconfig import QConfigAny
from ..utils import (
    activation_is_int8_quantized,
    weight_is_statically_quantized,
    get_qparam_dict,
    _parent_name,
)
from .backend_config.utils import get_quantized_reference_module_mapping

from .graph_module import (
    QuantizedGraphModule,
    is_observed_standalone_module,
)
from ._equalize import update_obs_for_equalization, convert_eq_obs
from .utils import (
    get_custom_module_class_keys,
    get_quantize_node_info,
    create_getattr_from_value,
)

from torch.ao.quantization.quantize import (
    _remove_qconfig,
    is_activation_post_process,
)

from .convert import restore_state

# these are tuples so that they can work with isinstance(module, tuple_of_classes)
FUSED_MODULE_CLASSES = (
    torch.nn.intrinsic.LinearReLU,
    torch.nn.intrinsic.ConvReLU1d,
    torch.nn.intrinsic.ConvReLU2d,
    torch.nn.intrinsic.ConvReLU3d,
)

QAT_MODULE_CLASSES = (
    torch.nn.qat.Linear,
    torch.nn.qat.Conv2d,
    torch.nn.qat.Conv3d,
    torch.nn.intrinsic.qat.LinearReLU,
    torch.nn.intrinsic.qat.ConvBn2d,
    torch.nn.intrinsic.qat.ConvBnReLU2d,
    torch.nn.intrinsic.qat.ConvReLU2d,
    torch.nn.intrinsic.qat.ConvBn3d,
    torch.nn.intrinsic.qat.ConvBnReLU3d,
    torch.nn.intrinsic.qat.ConvReLU3d
)

def insert_dequantize_node(
        node: Node,
        graph: Graph):
    """ Inserts dequantize node for `node` in `graph`
    """
    with graph.inserting_after(node):
        dequantize_node = graph.call_method("dequantize", (node,))
        for user_node in dict(node.users):
            if user_node is not dequantize_node:
                user_node.replace_input_with(node, dequantize_node)

def _convert_do_not_use(
        model: GraphModule, is_reference: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None,
        is_standalone_module: bool = False,
        _remove_qconfig_flag: bool = True,
        backend_config_dict: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
    """
    We will convert an observed model (a module with observer calls) to a reference
    quantized model, the rule is simple:
    1. for each observer module call in the graph, we'll convert it to calls to
       quantize and dequantize functions based on the observer instance
    2. for weighted operations like linear/conv, we need to convert them to reference
       quantized module, this requires us to know whether the dtype configured for the
       weight is supported in the backend, this is done in prepare step and the result
       is stored in observed_node_names, we can decide whether we need to swap the
       module based on this set

    standalone_module means it a submodule that is not inlined in
    parent module, and will be quantized separately as one unit.

    Returns a quantized standalone module, whether input/output is quantized is
    specified by prepare_custom_config_dict, with
    input_quantized_idxs, output_quantized_idxs, please
    see docs for prepare_fx for details
    """
    if convert_custom_config_dict is None:
        convert_custom_config_dict = {}
    patterns, node_name_to_scope, prepare_custom_config_dict, observed_node_names = restore_state(model)
    qconfig_map: Dict[str, QConfigAny] = model._qconfig_map  # type: ignore[assignment]

    assert is_reference, "_convert_do_not_use only supports reference option"

    # mapping from fully qualified module name to module instance
    # for example,
    # {
    #   '': Model(...),
    #   'linear': Linear(...),
    #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
    # }
    # We use remove_duplicate=False here because torch.cat uses
    # the same activation_post_process module instance but different names
    modules = dict(model.named_modules(remove_duplicate=False))

    custom_module_classes = get_custom_module_class_keys(
        convert_custom_config_dict,
        "observed_to_quantized_custom_module_class")

    if model._equalization_qconfig_map is not None:
        # If we want to do equalization then do the following:
        # Calculate the equalization scale, update the observers with the scaled
        # inputs, and scale the weight
        weight_eq_obs_dict = update_obs_for_equalization(model, modules)
        convert_eq_obs(model, modules, weight_eq_obs_dict)

    graph_inputs: List[str] = []
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            graph_inputs.append(node.name)

    def replace_observer_with_quantize_dequantize_node(graph: Graph, node: Node, modules: Dict[str, torch.nn.Module]) -> None:
        """ Replace activation_post_process module call node with quantize and
        dequantize node

        Before:
        ... -> observer_0(x) -> ...
        After:
        ... -> torch.quantize_per_tensor(x, ...) -> x.dequantize() -> ...
        """
        assert modules is not None
        assert isinstance(node.target, str)
        observer_module = modules[node.target]
        root_module = modules[""]
        if observer_module.dtype == torch.float32:
            # remove the node for now
            # TODO: support dynamic quant
            with graph.inserting_before(node):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
        elif observer_module.dtype in [torch.quint8, torch.qint8, torch.float16]:
            node_type, quantize_op, qparams = get_quantize_node_info(observer_module)
            # replace observer node with quant - dequant node
            with graph.inserting_before(node):
                input_node = node.args[0]
                inputs = [input_node]
                for key, value in qparams.items():
                    if key in ['_scale_', '_zero_point_']:
                        # For scale and zero_point values we register them as buffers in the root module.
                        # TODO: maybe need more complex attr name here
                        qparam_node = create_getattr_from_value(root_module, graph, key, value)
                        inputs.append(qparam_node)
                    else:
                        # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                        inputs.append(value)

                quantized_node = graph.create_node(node_type, quantize_op, tuple(inputs), {})
                dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
                node.replace_all_uses_with(dequantized_node)
                graph.erase_node(node)


    # additional state to override inputs to be quantized, if specified
    # by the user
    placeholder_node_seen_cnt = 0
    output_node_seen_cnt = 0
    input_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "input_quantized_idxs", [])
    output_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "output_quantized_idxs", [])

    if backend_config_dict is None:
        backend_config_dict = {}
    quantized_reference_module_mapping = get_quantized_reference_module_mapping(backend_config_dict)
    # convert tuples so that it can work with isinstance(module, tuple_of_classes)
    weighted_module_classes = tuple(quantized_reference_module_mapping.keys())

    for node in list(model.graph.nodes):
        if node.op == 'placeholder':
            cur_placeholder_node_idx = placeholder_node_seen_cnt
            placeholder_node_seen_cnt += 1
            if cur_placeholder_node_idx in input_quantized_idxs:
                # Inputs are assumed to be quantized if the user specifid the
                # input_quantized_idxs override.
                # we need to dequantize the inputs since all operators took
                # floating point inputs in reference quantized models
                insert_dequantize_node(node, model.graph)
        elif node.op == "output":
            cur_output_node_idx = output_node_seen_cnt
            output_node_seen_cnt += 1
            if cur_output_node_idx in output_quantized_idxs:
                # Result are kept quantized if the user specified the
                # output_quantized_idxs override.
                # Remove the dequantize operator in the end
                maybe_dequantize_node = node.args[0]
                if isinstance(maybe_dequantize_node, Node) and \
                   maybe_dequantize_node.op == "call_method" and \
                   maybe_dequantize_node.target == "dequantize":
                    quantize_node = maybe_dequantize_node.args[0]
                    maybe_dequantize_node.replace_all_uses_with(quantize_node)
                    model.graph.erase_node(maybe_dequantize_node)
        elif node.op == "call_module":
            if is_activation_post_process(modules[node.target]):
                replace_observer_with_quantize_dequantize_node(model.graph, node, modules)
            elif is_observed_standalone_module(modules[node.target]):
                # TODO: move this to a separate function
                convert = torch.ao.quantization._quantize_fx_do_not_use._convert_do_not_use  # type: ignore[attr-defined]
                # We know that observed standalone module is a GraphModule since
                # it's produced by us
                observed_standalone_module : GraphModule = modules[str(node.target)]  # type: ignore[assignment]
                sm_input_quantized_idxs = \
                    observed_standalone_module \
                    ._standalone_module_input_quantized_idxs\
                    .tolist()  # type: ignore[operator]
                # remove the dequantize nodes for inputs
                args = list(node.args)
                for idx in range(len(args)):
                    if idx in sm_input_quantized_idxs:
                        arg = args[idx]
                        if arg.op == "call_method" and arg.target == "dequantize":
                            quantize_node = arg.args[0]
                            node.replace_input_with(arg, quantize_node)
                            if len(arg.users) == 0:
                                model.graph.erase_node(arg)
                # add dequantize node for output
                sm_output_quantized_idxs = \
                    observed_standalone_module \
                    ._standalone_module_output_quantized_idxs \
                    .tolist()  # type: ignore[operator]
                if len(sm_output_quantized_idxs) > 0:
                    assert sm_output_quantized_idxs[0] == 0, "Currently only quantized"
                    "output idxs = [0] is supported"

                    # if it's non-empty, then it means the output is kept in quantized form
                    # we'll just add a dequantize node after this node
                    insert_dequantize_node(node, model.graph)

                # TODO: allow convert_custom_config_dict to override backend_config_dict
                # for standalone module
                quantized_standalone_module = convert(
                    observed_standalone_module,
                    is_reference=True,
                    backend_config_dict=backend_config_dict)
                parent_name, name = _parent_name(node.target)
                # update the modules dict
                setattr(modules[parent_name], name, quantized_standalone_module)
                modules[str(node.target)] = quantized_standalone_module
            elif type(modules[node.target]) in set(
                    weighted_module_classes).union(QAT_MODULE_CLASSES).union(FUSED_MODULE_CLASSES):
                # TODO: refactor this part to a function
                original_module = modules[node.target]
                qconfig = original_module.qconfig

                is_observed = node.name in observed_node_names
                is_activation_quantized = activation_is_int8_quantized(qconfig)
                is_weight_quantized = weight_is_statically_quantized(qconfig)
                # TODO: rename weight_is_statically_quantized to weight_is_int8_quantized
                if qconfig is None or \
                   not is_observed or \
                   not is_weight_quantized or \
                   not is_activation_quantized:
                    continue

                float_module = original_module
                fused_module = None
                if isinstance(
                        original_module,
                        QAT_MODULE_CLASSES):
                    # case 1. converting qat module to
                    # a float module, we need to attch
                    # weight fake_quant to the module,
                    # weight fake_quant is assumed to be run during
                    # QAT so we don't need to run it again here
                    float_module = original_module.to_float()  # type: ignore[operator]
                    # change qat conv to conv
                    parent_name, name = _parent_name(node.target)
                    setattr(modules[parent_name], name, float_module)
                    if isinstance(float_module, torch.nn.intrinsic._FusedModule):
                        fused_module = float_module
                        float_module = fused_module[0]
                    weight_post_process = original_module.weight_fake_quant
                else:
                    # case 2. converting a float module/fused float module
                    # to float module, we need to attach
                    # weight observer to the conv module and run it
                    # with conv weight
                    if isinstance(original_module, torch.nn.intrinsic._FusedModule):
                        fused_module = original_module
                        float_module = fused_module[0]  # type: ignore[index]
                    assert qconfig is not None
                    weight_post_process = qconfig.weight()
                    # run weight observer
                    weight_post_process(float_module.weight)  # type: ignore[operator]
                weight_qparams = get_qparam_dict(weight_post_process)
                # TODO: may need to change the mapping when we support dynamic quantization
                ref_qmodule_cls = quantized_reference_module_mapping.get(type(float_module), None)
                assert ref_qmodule_cls is not None, f"No reference quantized module class configured for {type(float_module)}"
                ref_qmodule = ref_qmodule_cls.from_float(float_module, weight_qparams)  # type: ignore[attr-defined]
                if fused_module is not None:
                    fused_module[0] = ref_qmodule
                else:
                    parent_name, name = _parent_name(node.target)
                    setattr(modules[parent_name], name, ref_qmodule)

    # removes qconfig and activation_post_process modules
    if _remove_qconfig_flag:
        _remove_qconfig(model)
    preserved_attributes = set(convert_custom_config_dict.get("preserved_attributes", []))
    model = QuantizedGraphModule(model, model.graph, preserved_attributes)
    return model
