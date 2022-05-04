import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Union, Dict, Any, Set

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
        self.preserved_attr_names = set([
            '_activation_post_process_map',
            '_activation_post_process_indexes',
            '_patterns',
            '_qconfig_map',
            '_prepare_custom_config_dict',
            '_equalization_qconfig_map',
            '_node_name_to_scope',
            '_qconfig_dict',
            '_is_qat',
            '_observed_node_names']).union(preserved_attr_names)
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

def is_observed_module(module: Any) -> bool:
    return isinstance(module, ObservedGraphModule)

class ObservedStandaloneGraphModule(ObservedGraphModule):
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        preserved_attr_names = preserved_attr_names.union(set([
            "_standalone_module_input_quantized_idxs",
            "_standalone_module_output_quantized_idxs"]))
        super().__init__(root, graph, preserved_attr_names)

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedStandaloneGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))

def is_observed_standalone_module(module: Any) -> bool:
    return isinstance(module, ObservedStandaloneGraphModule)

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
