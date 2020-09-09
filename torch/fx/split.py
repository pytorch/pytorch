import torch
from .graph_module import GraphModule
from .node import Node
from .graph import Graph
from .symbolic_trace import DefaultDelegate
from typing import List, Set, Dict

def extract_module(mod : GraphModule, target_qualname : str) -> GraphModule:
    """
    Given a GraphModule and a qualified name for a Module resident in the
    GraphModule's module hierarchy, return a new GraphModule that is equivalent
    to the original, but with the nodes originating from `target_qualname`
    out-lined (opposite of inlined).
    """
    target_qualname_atoms : List[str] = target_qualname.split('.')
    target_module : torch.nn.Module = mod
    for atom in target_qualname_atoms:
        target_module = getattr(target_module, atom)


    # ===== Stage 1: identify Nodes to split off and external data deps =====
    block_nodes = []

    # Keep track of data dependencies used and referenced in the
    # extracted block and the rest of the code, respectively.
    #
    # We'll use this later to figure out what needs to be inputs/outputs
    # to/from the extracted block
    block_defs : Set[Node] = set()
    block_uses : Set[Node] = set()
    rest_defs : Set[Node] = set()
    rest_uses : Set[Node] = set()

    for node in mod.graph.nodes:
        def in_block(node):
            if not node.module_qualname:
                return False
            module_qualname_atoms = node.module_qualname.split('.')
            if len(module_qualname_atoms) < len(target_qualname_atoms):
                return False
            for l, r in zip(module_qualname_atoms, target_qualname_atoms):
                if l != r:
                    return False
            return True

        def_set = block_defs if in_block(node) else rest_defs
        use_set = block_uses if in_block(node) else rest_uses
        if in_block(node):
            block_nodes.append(node)

        for arg in node.args:
            if isinstance(arg, Node):
                use_set.add(arg)
        for k, v in node.kwargs.items():
            if isinstance(v, Node):
                use_set.add(v)
        def_set.add(node)

    block_inputs = block_uses - block_defs
    block_outputs = list(rest_uses - rest_defs)
    if len(block_outputs) != 1:
        err = f"Found more than one output value while extracting {target_qualname}. " \
              f"This could happen if some transformation has modified Module qualified "\
              f"names in a way that no longer preserves the hierarchical structure."
        raise RuntimeError(err)


    # ===== Stage 2: Create submodule for extracted module =====
    # TODO: ideally we would just symbolically trace `target_module` but
    # I have to think through how to associate the positional/kwarg
    # inputs with traced values

    submod_graph = Graph()
    submod_delegate = DefaultDelegate(target_module, submod_graph)

    # Map Node in the original Graph to the node in the new graph
    remap_table = {}
    for block_input in block_inputs:
        remap_table[block_input] = submod_delegate.placeholder(block_input.name)

    for node in block_nodes:
        def arg_transform(n : Node):
            return remap_table[n]

        # We need to fixup all qualfied names to strip off the base
        # qualname, since now all qualnames in the new Module should
        # be wrt that new module
        def qualname_transform(qualname : str) -> str:
            return qualname.replace(f'{target_qualname}.', '')

        submod_graph_node = submod_graph.node_copy(node, arg_transform, qualname_transform)

        # Keep track of created output values
        remap_table[node] = submod_graph_node

    # Register output
    submod_graph.output(remap_table[block_outputs[0]])
    submod_module = GraphModule(target_module, submod_graph)

    # ===== Stage 3: Create new base GraphModule with submodule nodes replaced
    #       with a callsite to that new submodule =====

    base_graph : Graph = Graph()
    base_delegate : DefaultDelegate = DefaultDelegate(mod, base_graph)
    base_remap_table : Dict[Node, Node] = {}

    callsite_inserted = False
    for orig_node in mod.graph.nodes:
        def arg_transform(n : Node):
            return base_remap_table[n]
        if orig_node in block_defs:
            if not callsite_inserted:
                args = tuple(arg_transform(n) for n in block_inputs)
                base_remap_table[block_outputs[0]] = base_delegate.create_node(
                    'call_module', target_qualname, args, {}, block_outputs[0].name,
                    orig_node.module_qualname)
                callsite_inserted = True
            # Fallthrough. For each node that's a def in the subgraph,
            # just elide copying it.
        else:
            base_remap_table[orig_node] = base_graph.node_copy(orig_node, arg_transform)

    # Register output
    assert isinstance(mod.graph.result, Node)
    base_graph.output(base_remap_table[mod.graph.result])

    rv = GraphModule(mod, base_graph)

    new_root = rv
    new_root_target = new_root
    for atom in target_qualname.split('.')[:-1]:
        new_root_target = getattr(new_root_target, atom)
    setattr(new_root_target, target_qualname.split('.')[-1], submod_module)

    return rv


def fully_outline_module(mod : GraphModule) -> GraphModule:
    """
    Fully out-line all nodes within a GraphModule. After this happens, all
    nodes will appear in nested GraphModules reflecting the original Module
    hierarchy
    """
    # First, extract all unique Module qualnames
    all_unique_qualnames = set(n.module_qualname for n in mod.graph.nodes if n.module_qualname)

    # Split qualnames by atoms
    qualnames_by_atoms = [qualname.split('.') for qualname in all_unique_qualnames]
    # Sort in descending order. We want to uninline the most deeply nested
    # Modules first.
    qualnames_by_atoms = sorted(qualnames_by_atoms, reverse=True)

    # Finally, uninline all submodules
    uninlined = mod
    for qualname_atoms in qualnames_by_atoms:
        uninlined = extract_module(uninlined, '.'.join(qualname_atoms))

    return uninlined
