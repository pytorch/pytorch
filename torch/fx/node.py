# Nodes represent a definition of a value in our graph of operators.
from typing import TYPE_CHECKING, Union, Callable, Any, Set, Tuple, List, Optional, Dict
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
                 args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> None:
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
        self.uses : Set['Node'] = set()

    def __repr__(self) -> str:
        return self.name

    def replace_all_uses_with(self, replace_with : 'Node') -> List['Node']:
        """
        Replace all uses of `self` in the Graph with the Node `replace_with`.
        Returns the list of nodes on which this change was made.
        """
        to_process = list(self.uses)
        for use_node in to_process:
            def maybe_replace_node(n : Node) -> Node:
                if n == self:
                    self.uses.remove(use_node)
                    return replace_with
                else:
                    return n
            use_node.args = map_arg(use_node.args, maybe_replace_node)
            use_node.kwargs = map_arg(use_node.kwargs, maybe_replace_node)

        return to_process


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
