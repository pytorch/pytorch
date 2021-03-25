# Nodes represent a definition of a value in our graph of operators.
from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
from .immutable_collections import immutable_dict, immutable_list
import torch
import builtins
import inspect
import types
from typing import cast
from torch._jit_internal import boolean_dispatched
from torch.fx.operator_schemas import get_signature_for_torch_op, type_matches

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

_side_effectful_functions: Set[Callable] = {torch._assert}

# this is fixed on master, WAR for 1.5
def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')

# Borrowed from CPython typing module
# https://github.com/python/cpython/blob/f90dc36c15d7fee0efaf6d39e97be0bdf2683e93/Lib/typing.py#L156
def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    # HACK: In Python 3.6, type aliases from ``typing`` are instances of ``type``, but in
    # later Python versions, type aliases are not instances of ``type``!! We want
    # all type aliases to fall through to ``repr``, so if we have a type that is
    # in the module typing, don't go down this path.
    if isinstance(obj, type) and obj.__module__ != 'typing':
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is ...:
        return('...')
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)

def _get_qualified_name(func: Callable[..., Any]) -> str:
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
    return f'{module}.{name}'

def _format_arg(arg) -> str:
    if isinstance(arg, list):
        items = ', '.join(_format_arg(a) for a in arg)
        return f'[{items}]'
    elif isinstance(arg, tuple):
        items = ', '.join(_format_arg(a) for a in arg)
        maybe_comma = ',' if len(arg) == 1 else ''
        return f'({items}{maybe_comma})'
    elif isinstance(arg, dict):
        items_str = ', '.join(f'{k}: {_format_arg(v)}' for k, v in arg.items())
        return f'{{{items_str}}}'

    if isinstance(arg, Node):
        return '%' + str(arg)
    else:
        return str(arg)

class NormalizedArguments(object):
    def __init__(self, args_dict):
        self.args_dict = args_dict
        self.keys = self.args_dict.keys()

    def __iter__(self):
        return iter(self.args_dict)

    def __getitem__(self, key):
        return self.args_dict[key]

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
        # For the return node, this value will be used to type-annotate the
        # generated function return type. (Note this is a special case. ``return``
        # does not produce a value, it's more of a notation. Thus, this value
        # describes the type of args[0] in the ``return`` node.
        self.type : Optional[Any] = type
        self._prev = self
        self._next = self
        self._erased = False

        # If set, use this fn to print this node
        self._repr_fn : Optional[Callable[[Node], str]] = None
        self._stack_trace : Optional[str] = None

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

    @property
    def stack_trace(self) -> Optional[str]:
        """
        Return the Python stack trace that was recorded during tracing, if any.
        This property is usually populated by `Tracer.create_proxy`. To record
        stack traces during tracing for debug purposes, set
        `record_stack_traces = True` on the `Tracer` instance.
        """
        return self._stack_trace

    @stack_trace.setter
    def stack_trace(self, trace : Optional[str]):
        self._stack_trace = trace

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
        if self._repr_fn:
            return self._repr_fn(self)
        return self.name

    def _pretty_print_target(self, target):
        """
        Make target printouts more user-friendly.
        1) builtins will be printed as `builtins.xyz`
        2) operators will be printed as `operator.xyz`
        3) other callables will be printed with qualfied name, e.g. torch.add
        """
        if isinstance(target, str):
            return target
        if hasattr(target, '__module__'):
            if not hasattr(target, '__name__'):
                # Just to be defensive, if we don't have `__name__`, get the
                # qualname. Not sure if this happens for any members of `operator`
                # or `builtins`. This fallback path is not as good, since e.g.
                # things in `operator` have `_operator` as their __module__.
                return _get_qualified_name(target)
            if target.__module__ == 'builtins':
                return f'builtins.{target.__name__}'
            elif target.__module__ == '_operator':
                return f'operator.{target.__name__}'
        return _get_qualified_name(target)

    def format_node(self,
                    placeholder_names: List[str] = None,
                    maybe_return_typename: List[str] = None) -> Optional[str]:
        """
        Return a descriptive string representation of ``self``.

        This method can be used with no arguments as a debugging
        utility.

        This function is also used internally in the ``__str__`` method
        of ``Graph``. Together, the strings in ``placeholder_names``
        and ``maybe_return_typename`` make up the signature of the
        autogenerated ``forward`` function in this Graph's surrounding
        GraphModule. ``placeholder_names`` and ``maybe_return_typename``
        should not be used otherwise.

        Args:
            placeholder_names: A list that will store formatted strings
                representing the placeholders in the generated
                ``forward`` function. Internal use only.
            maybe_return_typename: A single-element list that will store
                a formatted string representing the output of the
                generated ``forward`` function. Internal use only.

        Returns:
            str: If 1) we're using ``format_node`` as an internal helper
                in the ``__str__`` method of ``Graph``, and 2) ``self``
                is a placeholder Node, return ``None``. Otherwise,
                return a  descriptive string representation of the
                current Node.
        """
        if self.op == 'placeholder':
            assert isinstance(self.target, str)
            arg_str = self.target
            arg_str += arg_str + f': {_type_repr(self.type)}' if self.type else ''
            if placeholder_names:
                placeholder_names.append(arg_str)
                return None
            maybe_typename = f'{_type_repr(self.type)} ' if self.type else ''
            default_val = '(default=' + str(self.args[0]) + ')' if self.args else ''
            return f'%{self.name} : {maybe_typename}[#users={len(self.users)}] = {self.op}[target={self.target}]{default_val}'
        elif self.op == 'get_attr':
            maybe_typename = f'{_type_repr(self.type)} ' if self.type is not None else ''
            return f'%{self.name} : {maybe_typename}[#users={len(self.users)}] = ' \
                   f'{self.op}[target={self._pretty_print_target(self.target)}]'
        elif self.op == 'output':
            if self.type and maybe_return_typename:
                maybe_return_typename[0] = f' -> {_type_repr(self.type)}'
            return f'return {self.args[0]}'
        else:
            maybe_typename = f'{_type_repr(self.type)} ' if self.type is not None else ''
            return f'%{self.name} : {maybe_typename}[#users={len(self.users)}] = ' \
                   f'{self.op}[target={self._pretty_print_target(self.target)}](' \
                   f'args = {_format_arg(self.args)}, kwargs = {_format_arg(self.kwargs)})'

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

    def is_impure(self):
        """
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Returns:

            bool: If the op is impure or not.
        """
        if self.op in {"placeholder", "output"}:
            return True

        # Check if an impure function.
        if self.op == "call_function":
            return self.target in _side_effectful_functions

        # Check if an impure module.
        if self.op == "call_module":
            assert (
                self.graph.owning_module is not None
            ), "self.graph.owning_module not set for purity check"
            target_mod = self.graph.owning_module.get_submodule(self.target)
            assert (
                target_mod is not None
            ), f"Did not find expected submodule target {self.target}"
            return getattr(target_mod, "_is_impure", False)

        return False

    def normalized_arguments(
            self, root : torch.nn.Module, arg_types : Optional[Tuple[Any]] = None,
            kwarg_types : Optional[Dict[str, Any]] = None) -> Optional[NormalizedArguments]:
        if self.op == 'call_function':
            new_kwargs = None

            if self.target.__module__ == 'torch.nn.functional':
                target_for_analysis = self.target
                if self.target in boolean_dispatched:
                    # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
                    # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
                    # branches of the dispatch have exactly the same signature. If they do, use the `true`
                    # branch signature for analysis. Otherwise, leave this un-normalized
                    assert not isinstance(self.target, str)
                    dispatched = boolean_dispatched[self.target]
                    if_true, if_false = dispatched['if_true'], dispatched['if_false']
                    if inspect.signature(if_true).parameters != inspect.signature(if_false).parameters:
                        return None
                    target_for_analysis = if_true

                assert callable(target_for_analysis)
                sig = inspect.signature(inspect.unwrap(target_for_analysis))
                new_kwargs = self._args_kwargs_to_normalized_kwargs(sig, self.args, self.kwargs)
            else:
                assert callable(self.target)
                torch_op_schemas = get_signature_for_torch_op(self.target)
                matched_schemas = []
                if torch_op_schemas:
                    # Iterate through all of the schema until we find one that matches
                    # If one matches, populate `new_kwargs` with the combined args/kwargs
                    # values. If none matches, `new_kwargs` will be None
                    for candidate_signature in torch_op_schemas:
                        try:
                            candidate_signature.bind(*self.args, **self.kwargs)
                            matched_schemas.append(candidate_signature)
                        except TypeError as e:
                            continue

                    if len(matched_schemas) == 0:
                        # Did not match any schema. Cannot normalize
                        pass
                    elif len(matched_schemas) == 1:
                        # Matched exactly one schema, unambiguous
                        new_kwargs = self._args_kwargs_to_normalized_kwargs(matched_schemas[0], self.args, self.kwargs)
                    else:
                        if arg_types is not None or kwarg_types is not None:
                            for candidate_signature in torch_op_schemas:
                                sig_matches = True
                                try:
                                    arg_types = arg_types if arg_types else cast(Tuple[Any], ())
                                    kwarg_types = kwarg_types if kwarg_types else {}
                                    bound_types = candidate_signature.bind(*arg_types, **kwarg_types)
                                    for arg_name, arg_type in bound_types.arguments.items():
                                        param = candidate_signature.parameters[arg_name]
                                        sig_matches = sig_matches and type_matches(param.annotation, arg_type)
                                except TypeError as e:
                                    sig_matches = False
                                if sig_matches:
                                    new_kwargs = self._args_kwargs_to_normalized_kwargs(candidate_signature, self.args, self.kwargs)
                                    break
                        else:
                            # Matched more than one schema. In this situation, the caller must provide the types of
                            # the arguments of the overload they expect.
                            schema_printouts = '\n'.join(str(schema) for schema in matched_schemas)
                            raise RuntimeError(f'Tried to normalize arguments to {torch.typename(self.target)} but '
                                               f'the schema match was ambiguous! Please provide argument types to '
                                               f'the normalize_arguments() call. Available schemas:\n{schema_printouts}')
            if new_kwargs:
                return NormalizedArguments(new_kwargs)

        elif self.op == 'call_module':
            assert isinstance(self.target, str)
            try:
                submod = root.get_submodule(self.target)
            except AttributeError:
                raise RuntimeError(f"Tried to normalize node with target {self.target} but root did not "
                                   f"have that target!")
            if hasattr(submod.__class__, '__name__'):
                classname = submod.__class__.__name__
                if getattr(torch.nn, classname, None) == submod.__class__:
                    sig = inspect.signature(inspect.unwrap(submod.forward))
                    new_kwargs = self._args_kwargs_to_normalized_kwargs(sig, self.args, self.kwargs)
                    if new_kwargs:
                        return NormalizedArguments(new_kwargs)
            return None

        return None

    def _args_kwargs_to_normalized_kwargs(self, sig : inspect.Signature, args : Tuple[Argument, ...],
                                          kwargs : Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Given a call target, args, and kwargs, return the arguments normalized into
        a single kwargs dict, or None if the type signature is not supported by
        this normalization.

        Args:

            target (inspect.Signature): Signature object for the target
            args (Tuple): Arguments that appear at the callsite for `target`
            kwargs (Dict): Keyword arugments that appear at the callsite for `target`

        Returns:

            Optional[Dict]: Normalized kwargs for `target`, or `None` if this target is not
                supported
        """

        # Don't currently support positional-only
        # or varargs (*args, **kwargs) signatures
        supported_parameter_types = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
            return None

        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        new_kwargs : Dict[str, Any] = {}
        for param in sig.parameters:
            new_kwargs[param] = bound_args.arguments[param]

        return new_kwargs



def map_arg(a: Argument, fn: Callable[[Node], Argument]) -> Argument:
    """ Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys. """
    assert callable(fn), "torch.fx.map_arg(a, fn): fn must be a callable"
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
