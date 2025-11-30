import operator

import torch


def annotate_getitem_nodes(graph: torch.fx.Graph) -> None:
    """
    Annotate the type of getitem nodes, inferred from the type of sequence node.
    If sequence node is not annotated with a type, do nothing.
    Currently support getitem nodes from tuple, list, and NamedTuple sequence node.

    This is helpful since annotations on local names within function are lost during FX transforms.
    Adding back known type annotation for getitem nodes to improve jit scriptability.

    Args:
        graph (Graph): The graph to be annotated
    """
    for node in graph.nodes:
        if node.target is operator.getitem:
            sequence_node, index_node = node.args
            if not sequence_node.type:
                continue
            # container types
            if hasattr(sequence_node.type, "_name"):
                parameterized_types = sequence_node.type.__args__
                if sequence_node.type._name == "Tuple":
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        assert len(parameterized_types) > index_node
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type._name == "List":
                    assert len(parameterized_types) == 1
                    node.type = parameterized_types[0]
            # Generic Alias Type
            elif hasattr(sequence_node.type, "__origin__"):
                parameterized_types = sequence_node.type.__args__
                if sequence_node.type.__origin__ is tuple:
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        assert len(parameterized_types) > index_node
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type.__origin__ is list:
                    assert len(parameterized_types) == 1
                    node.type = parameterized_types[0]
            # NamedTuple type
            elif hasattr(sequence_node.type, "__annotations__"):
                if sequence_node.type == torch.Tensor:
                    continue
                sequence_node_field_types = sequence_node.type.__annotations__
                field_name = sequence_node.type._fields[index_node]
                node.type = sequence_node_field_types[field_name]
