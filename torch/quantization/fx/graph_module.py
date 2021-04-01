import torch
import copy
from torch.fx import GraphModule  # type: ignore
from torch.fx.graph import Graph
from typing import Union, Dict, Any, List

class ObservedGraphModule(GraphModule):

    def get_preserved_attr_names(self) -> List[str]:
        return ['_activation_post_process_map',
                '_activation_post_process_indexes',
                '_patterns',
                '_qconfig_map',
                '_prepare_custom_config_dict',
                '_node_name_to_scope']

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph):
        preserved_attrs = dict()
        for attr in self.get_preserved_attr_names():
            preserved_attrs[attr] = getattr(root, attr)
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedGraphModule(fake_mod, self.graph)

def is_observed_module(module: Any) -> bool:
    return isinstance(module, ObservedGraphModule)

class ObservedStandaloneGraphModule(ObservedGraphModule):
    def get_preserved_attr_names(self) -> List[str] :
        return super().get_preserved_attr_names() + [
            "_standalone_module_input_quantized_idxs",
            "_standalone_module_output_quantized_idxs"
        ]

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedStandaloneGraphModule(fake_mod, self.graph)

def is_observed_standalone_module(module: Any) -> bool:
    return isinstance(module, ObservedStandaloneGraphModule)


def _save_packed_weight(self, destination, prefix, keep_vars):
    for attr_name in dir(self):
        if "_packed_weight" in attr_name and \
           isinstance(getattr(self, attr_name), torch._C.ScriptObject):  # type: ignore
            packed_weight = getattr(self, attr_name)
            destination[prefix + attr_name] = packed_weight

class QuantizedGraphModule(GraphModule):
    """ This class is created to make sure PackedParams
    (e.g. LinearPackedParams, Conv2dPackedParams) to appear in state_dict
    so that we can serialize and deserialize quantized graph module with
    torch.save(m.state_dict()) and m.load_state_dict(state_dict)
    """
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph):
        super().__init__(root, graph)
        self._register_state_dict_hook(_save_packed_weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        attrs_to_pop = []
        for attr_name in state_dict:
            if attr_name.startswith("_packed_weight") and isinstance(state_dict[attr_name], torch._C.ScriptObject):  # type: ignore
                setattr(self, attr_name, state_dict[attr_name])
                attrs_to_pop.append(attr_name)

        # pop the packed param attributesn
        for attr_name in attrs_to_pop:
            state_dict.pop(attr_name)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
