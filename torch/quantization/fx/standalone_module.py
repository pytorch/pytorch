from torch.fx import GraphModule

class ObservedStandaloneGraphModule(GraphModule):
    _PRESERVED_ATTR_NAMES = [
        '_activation_post_process_map',
        '_patterns',
        '_qconfig_map',
        '_standalone_module_observed_input_idxs',
        '_output_is_observed']

    def __init__(self, root, graph):
        preserved_attrs = dict()
        for attr in self._PRESERVED_ATTR_NAMES:
            preserved_attrs[attr] = getattr(root, attr)
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    def __deepcopy__(self, memo):
        preserved_attrs = dict()
        for attr in self._PRESERVED_ATTR_NAMES:
            preserved_attrs[attr] = getattr(self, attr)
        copied = super().__deepcopy__(memo)
        for attr in preserved_attrs:
            setattr(copied, attr, preserved_attrs[attr])
        return ObservedStandaloneGraphModule(copied, copied.graph)

def mark_observed_standalone_module(module):
    return ObservedStandaloneGraphModule(module, module.graph)

def is_observed_standalone_module(module):
    return isinstance(module, ObservedStandaloneGraphModule)
