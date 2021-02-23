import torch
import copy
from torch.fx import GraphModule  # type: ignore
from torch.fx.graph import Graph
from typing import Union, Dict, Any, List

class ObservedGraphModule(GraphModule):

    def get_preserved_attr_names(self) -> List[str]:
        return ['_activation_post_process_map',
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

def mark_observed_module(module: GraphModule) -> GraphModule:
    return ObservedGraphModule(module, module.graph)

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

def mark_observed_standalone_module(module: GraphModule) -> GraphModule:
    return ObservedStandaloneGraphModule(module, module.graph)

def is_observed_standalone_module(module: Any) -> bool:
    return isinstance(module, ObservedStandaloneGraphModule)
