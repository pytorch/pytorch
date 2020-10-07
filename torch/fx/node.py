# Nodes represent a definition of a value in our graph of operators.
from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict
import torch

if TYPE_CHECKING:
    from .graph import Graph

BaseArgumentTypes = Union[str, int, float, bool, torch.dtype, torch.Tensor]
base_types = BaseArgumentTypes.__args__  # type: ignore

Target = Union[Callable[..., Any], str]

Argument = Optional[Union[
    Tuple[Any, ...],  # actually Argument, but mypy can't represent recursive types
    List[Any],  # actually Argument
    Dict[str, Any],  # actually Argument
    slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
    'Node',
    BaseArgumentTypes
]]

class Node:
    def __init__(self, graph: 'Graph', name: str, op: str, target: Target,
                 args: Tuple[Argument, ...], kwargs: Dict[str, Argument],
                 type : Optional[Any] = None) -> None:
        self.graph = graph
        self.name = name  # unique name of value being created
        assert op in ['placeholder', 'call_method', 'call_module', 'call_function', 'get_attr', 'output']
        self.op = op  # the kind of operation = placeholder|call_method|call_module|call_function|get_attr
        if op in ['call_method', 'call_module']:
            assert isinstance(target, str)
        self.target = target  # for method/module/function, the name of the method/module/function/attr
        # being invoked, e.g add, layer1, or torch.add
        self.args = args
        self.kwargs = kwargs
        self.uses = 0
        # Type expression representing the output value of this node.
        # This should contain the same class of Type objects that would appear
        # as type annotations for function inputs/outputs.
        #
        # For placeholder nodes, this value will be used to type-annotate the
        # generated function parameters.
        # For the return ndoe, this value will be used to type-annotate the
        # generated function return type. (Note this is a special case. `return`
        # does not produce a value, it's more of a notation. Thus, this value
        # describes the type of args[0] in the `return` node.
        self.type : Optional[Any] = type

    def find_uses(self) -> List['Node']:
        """
        Find all nodes that use the value produced by `self`. The complexity of
        this function is linear in the number of nodes * number of arguments to
        each node.

        Note that len(find_uses()) is not necessarily equal to attribute `uses`.
        This node could be used multiple times in the same `Node`. In that case,
        the user node would appear once in the return value here, but `uses` would
        account for the total number of times this Node is used by the user node.
        e.g. a node for `x + x` would have two uses for the `x` node, but the
        `x + x` node would appear once in the return from `find_uses`
        """
        use_nodes : List[Node] = []
        for node in self.graph._nodes:
            def record_use(arg_node : Node) -> None:
                if arg_node == self and (len(use_nodes) == 0 or use_nodes[-1] != node):
                    use_nodes.append(node)
            map_arg(node.args, record_use)
            map_arg(node.kwargs, record_use)
        return use_nodes

    def __repr__(self) -> str:
        return self.name

    def replace_all_uses_with(self, replace_with : 'Node') -> List['Node']:
        """
        Replace all uses of `self` in the Graph with the Node `replace_with`.
        Returns the list of nodes on which this change was made.
        """
        use_nodes : List[Node] = self.find_uses()
        for use_node in use_nodes:
            def maybe_replace_node(n : Node) -> Node:
                if n == self:
                    self.uses -= 1
                    return replace_with
                else:
                    return n
            new_args = map_arg(use_node.args, maybe_replace_node)
            assert isinstance(new_args, tuple)
            use_node.args = new_args
            new_kwargs = map_arg(use_node.kwargs, maybe_replace_node)
            assert isinstance(new_kwargs, dict)
            use_node.kwargs = new_kwargs

        return use_nodes


def map_arg(a: Argument, fn: Callable[[Node], Argument]) -> Argument:
    """ apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys. """
    if isinstance(a, (tuple, list)):
        return type(a)(map_arg(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return {k: map_arg(v, fn) for k, v in a.items()}
    elif isinstance(a, slice):
        return slice(map_arg(a.start, fn), map_arg(a.stop, fn), map_arg(a.step, fn))
    elif isinstance(a, Node):
        return fn(a)
    else:
        return a
