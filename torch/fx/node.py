# Nodes represent a definition of a value in our graph of operators.
from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict
from .immutable_collections import immutable_dict, immutable_list
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
    """
    ``Node`` is the data structure that represents individual operations within
    a ``Graph``. For the most part, Nodes represent callsites to various entities,
    such as operators, methods, and Modules (some exceptions include nodes that
    specify function inputs and outputs). Each ``Node`` has a function specified
    by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

    - ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
      ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
      denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
      the function parameters (e.g. ``x``) in the graph printout.
    - ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
      fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
      ``args`` and ``kwargs`` are don't-care
    - ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
      to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
      following the Python calling convention
    - ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
      as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
      ``args`` and ``kwargs`` represent the arguments to invoke the module on, *including the self argument*.
    - ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
      to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
      *including the self argument*
    - ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
      in the Graph printout.
    """
    def __init__(self, graph: 'Graph', name: str, op: str, target: 'Target',
                 args: Tuple['Argument', ...], kwargs: Dict[str, 'Argument'],
                 type : Optional[Any] = None) -> None:
        self.graph = graph
        self.name = name  # unique name of value being created
        assert op in ['placeholder', 'call_method', 'call_module', 'call_function', 'get_attr', 'output', 'root']
        self.op = op  # the kind of operation = placeholder|call_method|call_module|call_function|get_attr
        if op in ['call_method', 'call_module']:
            assert isinstance(target, str)
        self.target = target  # for method/module/function, the name of the method/module/function/attr
        # being invoked, e.g add, layer1, or torch.add

        # All `Node`-valued inputs. Key is the Node, value is don't-care.
        # The public API for this is `all_input_nodes`, this private attribute
        # should not be accessed directly.
        self._input_nodes : Dict[Node, None] = {}
        self.__update_args_kwargs(map_arg(args, lambda x: x), map_arg(kwargs, lambda x: x))  # type: ignore

        # All of the nodes that use the value produced by this Node
        # Note one user may correspond to several uses, e.g. the node fo ``x + x``
        # would appear once here, but represents two uses.
        #
        # Is a dict to act as an "ordered set". Keys are significant, value dont-care
        self.users : Dict['Node', None] = {}
        # Type expression representing the output value of this node.
        # This should contain the same class of Type objects that would appear
        # as type annotations for function inputs/outputs.
        #
        # For placeholder nodes, this value will be used to type-annotate the
        # generated function parameters.
        # For the return ndoe, this value will be used to type-annotate the
        # generated function return type. (Note this is a special case. ``return``
        # does not produce a value, it's more of a notation. Thus, this value
        # describes the type of args[0] in the ``return`` node.
        self.type : Optional[Any] = type
        self._prev = self
        self._next = self
        self._erased = False

    @property
    def next(self) -> 'Node':
        """
        Returns the next ``Node`` in the linked list of Nodes.

        Returns:

            The next ``Node`` in the linked list of Nodes.
        """
        return self._next

    @property
    def prev(self) -> 'Node':
        """
        Returns the previous ``Node`` in the linked list of Nodes.

        Returns:

            The previous ``Node`` in the linked list of Nodes.
        """
        return self._prev

    def prepend(self, x: 'Node') -> None:
        """
        Insert x before this node in the list of nodes in the graph. Example::

            Before: p -> self
                    bx -> x -> ax
            After:  p -> x -> self
                    bx -> ax

        Args:
            x (Node): The node to put before this node. Must be a member of the same graph.
        """
        assert self.graph == x.graph, "Attempting to move a Node into a different Graph"
        x._remove_from_list()
        p = self._prev
        p._next, x._prev = x, p
        x._next, self._prev = self, x

    def append(self, x: 'Node') -> None:
        """
        Insert x after this node in the list of nodes in the graph.
        Equvalent to ``self.next.prepend(x)``

        Args:
            x (Node): The node to put after this node. Must be a member of the same graph.
        """
        self._next.prepend(x)

    def _remove_from_list(self):
        p, n = self._prev, self._next
        p._next, n._prev = n, p

    @property
    def args(self) -> Tuple[Argument, ...]:
        """
        The tuple of arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._args

    @args.setter
    def args(self, a : Tuple[Argument, ...]):
        """
        Set the tuple of arguments to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `__update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.args = new_args`
        self.__update_args_kwargs(map_arg(a, lambda x: x), self._kwargs)  # type: ignore

    @property
    def kwargs(self) -> Dict[str, Argument]:
        """
        The dict of keyword arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._kwargs

    @kwargs.setter
    def kwargs(self, k : Dict[str, Argument]):
        """
        Set the dict of kwargs to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `__update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.kwargs = new_kwargs`
        self.__update_args_kwargs(self._args, map_arg(k, lambda x: x))  # type: ignore

    @property
    def all_input_nodes(self) -> List['Node']:
        """
        Return all Nodes that are inputs to this Node. This is equivalent to
        iterating over ``args`` and ``kwargs`` and only collecting the values that
        are Nodes.

        Returns:

            List of ``Nodes`` that appear in the ``args`` and ``kwargs`` of this
            ``Node``, in that order.
        """
        return list(self._input_nodes.keys())

    def __update_args_kwargs(self, new_args : Tuple['Argument', ...], new_kwargs : Dict[str, 'Argument']):
        """
        This API is internal. Do *not* call it directly.
        """
        self._args = new_args
        self._kwargs = new_kwargs

        for old_use in self._input_nodes.keys():
            old_use.users.pop(self)

        self._input_nodes = {}
        map_arg(self._args, lambda n: self._input_nodes.setdefault(n))
        map_arg(self._kwargs, lambda n: self._input_nodes.setdefault(n))

        for new_use in self._input_nodes.keys():
            new_use.users.setdefault(self)

    def __repr__(self) -> str:
        return self.name

    def replace_all_uses_with(self, replace_with : 'Node') -> List['Node']:
        """
        Replace all uses of ``self`` in the Graph with the Node ``replace_with``.

        Args:

            replace_with (Node): The node to replace all uses of ``self`` with.

        Returns:

            The list of Nodes on which this change was made.
        """
        to_process = list(self.users)
        for use_node in to_process:
            def maybe_replace_node(n : Node) -> Node:
                if n == self:
                    return replace_with
                else:
                    return n

            new_args = map_arg(use_node.args, maybe_replace_node)
            new_kwargs = map_arg(use_node.kwargs, maybe_replace_node)
            assert isinstance(new_args, tuple)
            assert isinstance(new_kwargs, dict)
            use_node.__update_args_kwargs(new_args, new_kwargs)

        assert len(self.users) == 0
        return to_process

def map_arg(a: Argument, fn: Callable[[Node], Argument]) -> Argument:
    """ Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys. """
    return map_aggregate(a, lambda x: fn(x) if isinstance(x, Node) else x)

def map_aggregate(a: Argument, fn: Callable[[Argument], Argument]) -> Argument:
    """ Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys. """
    if isinstance(a, tuple):
        return tuple(map_aggregate(elem, fn) for elem in a)
    elif isinstance(a, list):
        return immutable_list(map_aggregate(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return immutable_dict((k, map_aggregate(v, fn)) for k, v in a.items())
    elif isinstance(a, slice):
        return slice(map_aggregate(a.start, fn), map_aggregate(a.stop, fn), map_aggregate(a.step, fn))
    else:
        return fn(a)
