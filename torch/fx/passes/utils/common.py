from torch.nn import Module

from torch.fx.graph_module import GraphModule
from torch.fx.graph import Graph
from torch.fx._compatibility import compatibility


@compatibility(is_backward_compatible=False)
class HolderModule(Module):
    """
    HolderModule is used to copy all the attributes from original module to submodules
    that uses the attributes
    """

    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            self.add_module(k, v)


def lift_subgraph_as_module(gm: GraphModule, subgraph: Graph, class_name: str = 'GraphModule') -> GraphModule:
    """
    Create a GraphModule for subgraph, which copies the necessory attributes from the original parent graph_module.

    Args:
        gm (GraphModule): parent graph module

        subgraph (Graph): a valid subgraph that contains copied nodes from the parent graph

        class_name (str): name for the submodule

    """

    # Loop through all module calls (call_module) and param fetches (get_attr)
    # in this component, creating HolderModules as necessary to match the path.
    # e.g. if in the original module there's a get_attr node fetches "conv.weight".
    # We create a HolderModule as root -> add a HolderModule named "conv" ->
    # make "weight" a attribute of "conv" HolderModule and point to conv.weight in
    # the original module.
    submodule = HolderModule({})
    for n in subgraph.nodes:
        if n.op not in ("call_module", "get_attr"):
            continue

        target = n.target
        assert isinstance(target, str)
        target_name_parts = target.split(".")
        curr = submodule
        orig_gm = gm

        for name in target_name_parts[:-1]:
            if not hasattr(curr, name):
                curr.add_module(name, HolderModule({}))

            curr = getattr(curr, name)
            orig_gm = getattr(orig_gm, name)

        leaf_node_name = target_name_parts[-1]
        leaf_node = getattr(orig_gm, leaf_node_name)

        # Relies on custom __setattr__ magic.
        setattr(curr, leaf_node_name, leaf_node)

    return GraphModule(submodule, subgraph, class_name)
