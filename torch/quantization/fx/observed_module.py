import torch
import copy
from torch.fx import GraphModule  # type: ignore

class ObservedGraphModule(GraphModule):

    def get_preserved_attr_names(self):
        return ['_activation_post_process_map',
                '_patterns',
                '_qconfig_map']

    def __init__(self, root, graph):
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

def mark_observed_module(module):
    return ObservedGraphModule(module, module.graph)

def is_observed_module(module):
    return isinstance(module, ObservedGraphModule)

class ObservedStandaloneGraphModule(ObservedGraphModule):
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedStandaloneGraphModule(fake_mod, self.graph)

def mark_observed_standalone_module(module):
    return ObservedStandaloneGraphModule(module, module.graph)

def is_observed_standalone_module(module):
    return isinstance(module, ObservedStandaloneGraphModule)
