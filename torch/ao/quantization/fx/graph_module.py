# mypy: allow-untyped-defs
import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Union, Dict, Any, Set

__all__ = [
    "FusedGraphModule",
    "ObservedGraphModule",
    "ObservedStandaloneGraphModule",
    "QuantizedGraphModule",
]

class FusedGraphModule(GraphModule):
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {attr: getattr(root, attr) for attr in self.preserved_attr_names if hasattr(root, attr)}
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return FusedGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))

class ObservedGraphModule(GraphModule):

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        self.preserved_attr_names = {
            '_activation_post_process_map',
            '_activation_post_process_indexes',
            '_patterns',
            '_node_name_to_qconfig',
            '_prepare_custom_config',
            '_equalization_node_name_to_qconfig',
            '_node_name_to_scope',
            '_qconfig_mapping',
            '_is_qat',
            '_observed_node_names'}.union(preserved_attr_names)
        preserved_attrs = {attr: getattr(root, attr) for attr in self.preserved_attr_names if hasattr(root, attr)}
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))

def _is_observed_module(module: Any) -> bool:
    return hasattr(module, "meta") and "_observed_graph_module_attrs" in module.meta

def _get_observed_graph_module_attr(model: Union[torch.nn.Module, GraphModule], attr_name: str) -> Any:
    if hasattr(model, "meta") and "_observed_graph_module_attrs" in model.meta:  # type: ignore[operator, index]
        return getattr(model.meta["_observed_graph_module_attrs"], attr_name)  # type: ignore[index]
    return None

class ObservedStandaloneGraphModule(ObservedGraphModule):
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        preserved_attr_names = preserved_attr_names.union({
            "_standalone_module_input_quantized_idxs",
            "_standalone_module_output_quantized_idxs"})
        super().__init__(root, graph, preserved_attr_names)

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedStandaloneGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))

def _is_observed_standalone_module(module: Any) -> bool:
    return _is_observed_module(module) and module.meta["_observed_graph_module_attrs"].is_observed_standalone_module

def _save_packed_weight(self, destination, prefix, keep_vars):
    for attr_name in dir(self):
        if "_packed_weight" in attr_name and \
           isinstance(getattr(self, attr_name), torch._C.ScriptObject):  # type: ignore[attr-defined]
            packed_weight = getattr(self, attr_name)
            destination[prefix + attr_name] = packed_weight

class QuantizedGraphModule(GraphModule):
    """ This class is created to make sure PackedParams
    (e.g. LinearPackedParams, Conv2dPackedParams) to appear in state_dict
    so that we can serialize and deserialize quantized graph module with
    torch.save(m.state_dict()) and m.load_state_dict(state_dict)
    """
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {attr: getattr(root, attr) for attr in self.preserved_attr_names if hasattr(root, attr)}
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])
        self._register_state_dict_hook(_save_packed_weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        attrs_to_pop = []
        for attr_name in state_dict:
            if attr_name.startswith("_packed_weight") and isinstance(state_dict[attr_name], torch._C.ScriptObject):  # type: ignore[attr-defined] # noqa: B950
                setattr(self, attr_name, state_dict[attr_name])
                attrs_to_pop.append(attr_name)

        # pop the packed param attributesn
        for attr_name in attrs_to_pop:
            state_dict.pop(attr_name)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return QuantizedGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))
