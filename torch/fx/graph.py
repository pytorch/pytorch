from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name

from typing import Callable, Any, List, Dict, Optional, Tuple, Set, FrozenSet
from dataclasses import dataclass
import torch
import keyword
import re
import builtins
import math

# Mapping of builtins to their `typing` equivalent.
_origin_type_map = {
    list: List,
    dict: Dict,
    set: Set,
    frozenset: FrozenSet
}

def _is_magic(x: str) -> bool:
    return x.startswith('__') and x.endswith('__')

def _snake_case(s: str) -> str:
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    chars = []
    prev_lower = False
    for c in s:
        if prev_lower and c.isupper():
            chars.append('_')
        chars.append(c.lower())
        prev_lower = c.islower()
    return ''.join(chars)


def _is_custom_torch(qualname: str) -> bool:
    return qualname.startswith('torch.ops') or qualname.startswith('torch.classes')


class _Globals:
    """Tracks all external objects referenced from a Graph.

    We assign each external object a unique global name, which is how it's
    referenced in the Graph's generated Python code.

    A dictionary of global name -> obj is passed to 'exec' to resolve
    external references to their actual objects.
    """
    def __init__(self):
        # The global name -> external object mapping.
        self.globals: Dict[str, Any] = {}

        # qualified name -> keys in self.globals
        self._names: Dict[str, str] = {}
        self._dedupe_index = 0

        # Always import torch
        self._import_strs: Set[str] = {'import torch'}

        # Preload names into globals. These are objects whose `repr` returns
        # something that actually needs to be imported to execute correctly.
        #
        # If you add to this, you probably need to add an import string in
        # '_format_import_str` as well.
        self.add_global(math.nan, 'nan')
        self.add_global(math.inf, 'inf')
        self.add_global(type(None), 'NoneType')

    def add_global(self, obj: Any, qualname: str) -> str:
        """Add an external obj to be tracked.

        Returns the global name that should be used to reference 'obj' in
        Python source code.

        Example: 'foo.bar.baz' becomes 'foo_bar_baz'
        """
        if _is_custom_torch(qualname):
            # HACK: workaround for how torch custom ops are registered. We can't
            # import them like normal modules so they must retain their fully qualified name.
            self._names[qualname] = qualname
            self.globals['torch'] = torch
            return qualname

        # Check if we already added this global.
        if qualname in self._names:
            global_name = self._names[qualname]
            assert self.globals[global_name] is obj
            return global_name

        global_name = qualname.replace('.', '_')

        # Resolve any collisions, e.g. between 'foo._bar' and 'foo_.bar'
        base_name = global_name
        while global_name in self.globals:
            global_name = f'{base_name}_{self._dedupe_index}'
            self._dedupe_index += 1

        # Populate our name tables
        self._names[qualname] = global_name
        self.globals[global_name] = obj

        # Save the import string
        import_str = self._format_import_str(qualname, global_name)
        if import_str is not None:
            self._import_strs.add(import_str)

        return global_name

    def format_import_block(self):
        """Return Python source code that can be executed to import external
        objects and assign them to the correct global names.

        This is used for serializing the source code, since we can't
        serialize the globals dict itself.
        """
        return '\n'.join(self._import_strs)

    @staticmethod
    def _format_import_str(qualname: str, global_name: str) -> Optional[str]:
        """Return a snippet of source code that imports `qualname` and assigns
        the resulting object to `global_name`."""
        module_name, _sep, type_name = qualname.rpartition('.')
        if len(module_name) == 0:
            # This was a single-atom qualified name, like 'getattr', or 'len'
            # These generally are builtins so don't need to be imported.
            #
            # But there are some exceptions, for objects whose `repr`
            # returns something that actually needs to be imported to
            # execute correctly.
            if qualname == 'NoneType':
                return f'{global_name} = type(None)'
            if qualname == 'inf':
                return f'from math import inf as {global_name}'
            if qualname == 'nan':
                return f'from math import nan as {global_name}'
            return None

        if _is_custom_torch(qualname):
            # HACK: workaround for how torch custom ops are registered. We can't
            # import them like normal modules so they must retain their fully qualified name.
            return 'import torch'
        else:
            return f'from {module_name} import {type_name} as {global_name}'


@dataclass
class PythonCode:
    """Represents all the information necessary to exec or save a graph as Python code."""
    # Python source code for the forward function definition.
    fn_src: str
    # Values in global scope during exection of `src_def`.
    globals: Dict[str, Any]
    # Python source code for import statements that will recreate the `globals` dict.
    import_block: str


def _format_args(args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> str:
    args_s = ', '.join(repr(a) for a in args)
    kwargs_s = ', '.join(f'{k} = {repr(v)}' for k, v in kwargs.items())
    if args_s and kwargs_s:
        return f'{args_s}, {kwargs_s}'
    return args_s or kwargs_s

def _format_target(base: str, target: str) -> str:
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f'{r}.{e}'
    return r

class _InsertPoint:
    def __init__(self, graph, new_insert):
        self.graph = graph
        self.orig_insert, graph._insert = graph._insert, new_insert

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        self.graph._insert = self.orig_insert

class _node_list:
    def __init__(self, graph: 'Graph', direction: str = '_next'):
        assert direction in ['_next', '_prev']
        self.graph = graph
        self.direction = direction

    def __len__(self):
        return self.graph._len

    def __iter__(self):
        root, direction = self.graph._root, self.direction
        cur = getattr(root, direction)
        while cur is not root:
            if not cur._erased:
                yield cur
            cur = getattr(cur, direction)

    def __reversed__(self):
        return _node_list(self.graph, '_next' if self.direction == '_prev' else '_prev')

class Graph:
    """
    ``Graph`` is the main data structure used in the FX Intermediate Representation.
    It consists of a series of ``Node`` s, each representing callsites (or other
    syntactic constructs). The list of ``Node`` s, taken together, constitute a
    valid Python function.

    For example, the following code

    .. code-block:: python

        import torch
        import torch.fx

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return torch.topk(torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3)

        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

    Will produce the following Graph::

        print(gm.graph)

    .. code-block:: text

        graph(x):
            %linear_weight : [#users=1] = self.linear.weight
            %add_1 : [#users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
            %linear_1 : [#users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
            %relu_1 : [#users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
            %sum_1 : [#users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
            %topk_1 : [#users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
            return topk_1

    For the semantics of operations represented in the ``Graph``, please see :class:`Node`.
    """
    def __init__(self):
        """
        Construct an empty Graph.
        """
        self._root : Node = Node(self, '', 'root', '', (), {})
        self._used_names : Dict[str, int] = {}  # base name -> number
        self._insert = self._root.prepend
        self._len = 0
        self._globals = _Globals()

    @property
    def nodes(self) -> _node_list:
        """
        Get the list of Nodes that constitute this Graph.

        Note that this ``Node`` list representation is a doubly-linked list. Mutations
        during iteration (e.g. delete a Node, add a Node) are safe.

        Returns:

            A doubly-linked list of Nodes. Note that ``reversed`` can be called on
            this list to switch iteration order.
        """
        return _node_list(self)

    def graph_copy(self, g : 'Graph', val_map : Dict[Node, Node]) -> 'Optional[Argument]':
        """
        Copy all nodes from a given graph into ``self``.

        Args:

            g (Graph): The source graph from which to copy Nodes.

            val_map (Dict[Node, Node]): a dictionary that will be populated with a mapping
                from nodes in ``g`` to nodes in ``self``. Note that ``val_map`` can be passed
                in with values in it already to override copying of certain values.

        Returns:

            The value in ``self`` that is now equivalent to the output value in ``g``,
            if ``g`` had an ``output`` node. ``None`` otherwise.
        """
        for node in g.nodes:
            if node in val_map:
                continue
            if node.op == 'output':
                rv = map_arg(node.args[0], lambda n: val_map[n])
                return rv
            val_map[node] = self.node_copy(node, lambda n : val_map[n])
        return None

    def __deepcopy__(self, memo=None) -> 'Graph':
        """
        Explicitly implement __deepcopy__ to prevent excessive recursion depth
        from the default implementation. This uses graph_copy to copy the nodes
        in an iterative way, rather than recursive. It also populates the
        memoization table to prevent unnecessary copies (e.g. references to
        nodes or other parts of the Graph from a custom GraphModule implementation
        """
        memo = memo if memo else {}
        g = Graph()
        output_val = g.graph_copy(self, val_map=memo)
        g.output(output_val)
        return g

    def create_node(self, op: str, target: 'Target',
                    args: Optional[Tuple['Argument', ...]] = None,
                    kwargs: Optional[Dict[str, 'Argument']] = None,
                    name: Optional[str] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Create a ``Node`` and add it to the ``Graph`` at the current insert-point.
        Note that the current insert-point can be set via :meth:`Graph.inserting_before`
        and :meth:`Graph.inserting_after`.

        Args:
            op (str): the opcode for this Node. One of 'call_function', 'call_method', 'get_attr',
                'call_module', 'placeholder', or 'output'. The semantics of these opcodes are
                described in the ``Graph`` docstring.

            args (Optional[Tuple[Argument, ...]]): is a tuple of arguments to this node.

            kwargs (Optional[Dict[str, Argument]]): the kwargs of this Node

            name (Optional[str]): an optional string name for the ``Node``.
                This will influence the name of the value assigned to in the
                Python generated code.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted node.
        """
        assert op in ('call_function', 'call_method', 'get_attr', 'call_module', 'placeholder', 'output')
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        assert isinstance(args, tuple), "args must be a tuple"
        assert isinstance(kwargs, dict), "kwargs must be a dict"
        unique_name = self._create_unique_name(name if name is not None else self._target_to_str(target))
        n = Node(self, unique_name, op, target, args, kwargs, type_expr)
        self._insert(n)
        self._len += 1
        return n

    def erase_node(self, to_erase : Node) -> None:
        """
        Erases a ``Node`` from the ``Graph``. Throws an exception if
        there are still users of that node in the ``Graph``.

        Args:

            to_erase (Node): The ``Node`` to erase from the ``Graph``.
        """
        if len(to_erase.users) > 0:
            raise RuntimeError(f'Tried to erase Node {to_erase} but it still had {len(to_erase.users)} '
                               f'users in the graph: {to_erase.users}!')

        to_erase._remove_from_list()
        to_erase._erased = True  # iterators may retain handles to erased nodes
        self._len -= 1

        # Null out this Node's argument nodes so that the Nodes referred to
        # can update their ``users`` accordingly
        new_args = map_arg(to_erase.args, lambda n: None)
        assert isinstance(new_args, tuple)
        to_erase.args = new_args
        new_kwargs = map_arg(to_erase.kwargs, lambda n: None)
        assert isinstance(new_kwargs, dict)
        to_erase.kwargs = new_kwargs

    def inserting_before(self, n: Optional[Node] = None):
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits::

            with g.inserting_before(n):
                ... # inserting before node n
            ... # insert point restored to what it was previously
            g.inserting_before(n) #  set the insert point permanently

        Args:
            n (Optional[Node]): The node before which to insert. If None this will insert before
              the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
        if n is None:
            return self.inserting_after(self._root)
        assert n.graph == self, "Node to insert before is not in graph."
        return _InsertPoint(self, n.prepend)

    def inserting_after(self, n: Optional[Node] = None):
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits::

            with g.inserting_after(n):
                ... # inserting after node n
            ... # insert point restored to what it was previously
            g.inserting_after(n) #  set the insert point permanently

        Args:
            n (Optional[Node]): The node before which to insert. If None this will insert after
              the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
        if n is None:
            return self.inserting_before(self._root)
        assert n.graph == self, "Node to insert after is not in graph."
        return _InsertPoint(self, n.append)

    # sugar for create_node when you know the op
    def placeholder(self, name: str, type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``placeholder`` node into the Graph. A ``placeholder`` represents
        a function input.

        Args:

            name (str): A name for the input value. This corresponds to the name
                of the positional argument to the function this ``Graph`` represents.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have. This is needed in some
                cases for proper code generation (e.g. when the function is used
                subsequently in TorchScript compilation).

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        return self.create_node('placeholder', name, type_expr=type_expr)

    def get_attr(self, qualified_name: str, type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``get_attr`` node into the Graph. A ``get_attr`` ``Node`` represents the
        fetch of an attribute from the ``Module`` hierarchy.

        Args:

            qualified_name (str): the fully-qualified name of the attribute to be retrieved.
                For example, if the traced Module has a submodule named ``foo``, which has a
                submodule named ``bar``, which has an attribute named ``baz``, the qualified
                name ``foo.bar.baz`` should be passed as ``qualified_name``.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.


        Returns:

            The newly-created and inserted ``get_attr`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        return self.create_node('get_attr', qualified_name, type_expr=type_expr)

    def call_module(self,
                    module_name: str,
                    args: Optional[Tuple['Argument', ...]] = None,
                    kwargs: Optional[Dict[str, 'Argument']] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``call_module`` ``Node`` into the ``Graph``. A ``call_module`` node
        represents a call to the forward() function of a ``Module`` in the ``Module``
        hierarchy.

        Args:

            module_name (str): The qualified name of the ``Module`` in the ``Module``
                hierarchy to be called. For example, if the traced ``Module`` has a
                submodule named ``foo``, which has a submodule named ``bar``, the
                qualified name ``foo.bar`` should be passed as ``module_name`` to
                call that module.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this should *not* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted ``call_module`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node('call_module', module_name, args, kwargs, type_expr=type_expr)

    def call_method(self,
                    method_name: str,
                    args: Optional[Tuple['Argument', ...]] = None,
                    kwargs: Optional[Dict[str, 'Argument']] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``call_method`` ``Node`` into the ``Graph``. A ``call_method`` node
        represents a call to a given method on the 0th element of ``args``.

        Args:

            method_name (str): The name of the method to apply to the self argument.
                For example, if args[0] is a ``Node`` representing a ``Tensor``,
                then to call ``relu()`` on that ``Tensor``, pass ``relu`` to ``method_name``.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this *should* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly created and inserted ``call_method`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node('call_method', method_name, args, kwargs, type_expr=type_expr)

    def call_function(self,
                      the_function: Callable[..., Any],
                      args: Optional[Tuple['Argument', ...]] = None,
                      kwargs: Optional[Dict[str, 'Argument']] = None,
                      type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``call_function`` ``Node`` into the ``Graph``. A ``call_function`` node
        represents a call to a Python callable, specified by ``the_function``. ``the_function``
        can be

        Args:

            the_function (Callable[..., Any]): The function to be called. Can be any PyTorch
                operator, Python function, or member of the ``builtins`` or ``operator``
                namespaces.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called function.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called function

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns

            The newly created and inserted ``call_function`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node('call_function', the_function, args, kwargs, type_expr=type_expr)

    def node_copy(self, node: Node, arg_transform: Callable[[Node], 'Argument'] = lambda x: x) -> Node:
        """
        Copy a node from one graph into another. ``arg_transform`` needs to transform arguments from
        the graph of node to the graph of self. Example::

            # Copying all the nodes in `g` into `new_graph`
            g : torch.fx.Graph = ...
            new_graph = torch.fx.graph()
            value_remap = {}
            for node in g.nodes:
                value_remap[node] = new_graph.node_copy(node, lambda n : value_remap[n])

        Args:

            node (Node): The node to copy into ``self``.

            arg_transform (Callable[[Node], Argument]): A function that transforms
                ``Node`` arguments in node's ``args`` and ``kwargs`` into the
                equivalent argument in ``self``. In the simplest case, this should
                retrieve a value out of a table mapping Nodes in the original
                graph to ``self``.
        """
        args = map_arg(node.args, arg_transform)
        kwargs = map_arg(node.kwargs, arg_transform)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        return self.create_node(node.op, node.target, args, kwargs, node.name, node.type)

    def output(self, result: 'Argument', type_expr: Optional[Any] = None):
        """
        Insert an ``output`` ``Node`` into the ``Graph``. An ``output`` node represents
        a ``return`` statement in Python code. ``result`` is the value that should
        be returned.

        Args:

            result (Argument): The value to be returned.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        .. note::

            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        return self.create_node(op='output', target='output', args=(result,), type_expr=type_expr)

    def _target_to_str(self, target : Target) -> str:
        if callable(target):
            op = target.__name__
        else:
            assert isinstance(target, str)
            op = target
            if _is_magic(op):
                op = op[2:-2]
        op = _snake_case(op)
        return op

    def _shadows_global_name(self, name: str) -> bool:
        return (
            name in builtins.__dict__
            or name in keyword.kwlist
            or name in self._globals.globals
        )

    def _create_unique_name(self, candidate : str) -> str:
        # delete all characters that are illegal in a Python identifier
        candidate = re.sub('[^0-9a-zA-Z_]+', '_', candidate)
        if candidate[0].isdigit():
            candidate = f'_{candidate}'

        def illegal_shadowing_name(name : str) -> bool:
            return hasattr(torch, name) or \
                hasattr(torch.nn.functional, name) or \
                hasattr(torch.nn, name) or \
                self._shadows_global_name(name)

        while candidate in self._used_names or illegal_shadowing_name(candidate):
            match = re.match(r"(.*)_(\d+)$", candidate)
            if match is None:
                candidate = candidate + '_1'
            else:
                base, num = match.group(1, 2)
                candidate = f'{base}_{int(num) + 1}'

        self._used_names.setdefault(candidate)
        return candidate

    def python_code(self, root_module: str) -> PythonCode:
        """
        Turn this ``Graph`` into valid Python code.

        Args:

            root_module (str): The name of the root module on which to look-up
                qualified name targets. This is usually 'self'.

        Returns:

            The string source code generated from this ``Graph``.
        """
        free_vars: List[str] = []
        body: List[str] = []
        self._globals = _Globals()

        # Wrap string in list to pass by reference
        maybe_return_annotation : List[str] = ['']

        def type_repr(o : Any):
            typename = _type_repr(o)

            # Common case: this is a regular module name like 'foo.bar.baz'
            if all(x.isidentifier() for x in typename.split('.')):
                return self._globals.add_global(o, typename)

            # This is a generic type, e.g. typing.List[torch.Tensor]
            origin_type = _origin_type_map.get(o.__origin__, o.__origin__)
            origin_typename = self._globals.add_global(origin_type, _type_repr(origin_type))

            # Assign global names for each of the inner type variables.
            args = [type_repr(arg) for arg in o.__args__]
            return f'{origin_typename}[{",".join(args)}]'


        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use : Dict[Node, Node] = {}
        user_to_last_uses : Dict[Node, List[Node]] = {}

        def register_last_uses(n : Node, user : Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(self.nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        def delete_unused_values(user : Node):
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
            if user.op == 'placeholder':
                return
            if user.op == 'output':
                body.append('\n')
                return
            nodes_to_delete = user_to_last_uses.get(user, [])
            if len(nodes_to_delete):
                to_delete_str = ' = '.join([n.name for n in nodes_to_delete] + ['None'])
                body.append(f';  {to_delete_str}\n')
            else:
                body.append('\n')

        def emit_node(node : Node):
            if node.op == 'placeholder':
                assert isinstance(node.target, str)
                maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
                maybe_default_arg = '' if not node.args else f' = {repr(node.args[0])}'
                free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
                raw_name = node.target.replace('*', '')
                if raw_name != node.name:
                    body.append(f'{node.name} = {raw_name}\n')
                return
            elif node.op == 'call_method':
                assert isinstance(node.target, str)
                body.append(
                    f'{node.name} = {_format_target(repr(node.args[0]), node.target)}'
                    f'({_format_args(node.args[1:], node.kwargs)})')
                return
            elif node.op == 'call_function':
                assert callable(node.target)
                # pretty print operators
                if node.target.__module__ == '_operator' and node.target.__name__ in magic_methods:
                    assert isinstance(node.args, tuple)
                    body.append(f'{node.name} = {magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}')
                    return
                qualified_name = _get_qualified_name(node.target)
                global_name = self._globals.add_global(node.target, qualified_name)
                if global_name == 'getattr' and \
                   isinstance(node.args, tuple) and \
                   isinstance(node.args[1], str) and \
                   node.args[1].isidentifier():
                    # pretty print attribute access
                    body.append(f'{node.name} = {_format_target(repr(node.args[0]), node.args[1])}')
                    return
                body.append(f'{node.name} = {global_name}({_format_args(node.args, node.kwargs)})')
                return
            elif node.op == 'call_module':
                assert isinstance(node.target, str)
                body.append(f'{node.name} = {_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
                return
            elif node.op == 'get_attr':
                assert isinstance(node.target, str)
                body.append(f'{node.name} = {_format_target(root_module, node.target)}')
                return
            elif node.op == 'output':
                if node.type is not None:
                    maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                body.append(f'return {repr(node.args[0])}')
                return
            raise NotImplementedError(f'node: {node.op} {node.target}')

        for node in self.nodes:
            # NOTE: emit_node does not emit a string with newline. It depends
            # on delete_unused_values to append one
            emit_node(node)
            delete_unused_values(node)

        # repr() for inf and nan floating point values aren't parseable by
        # python as literals. Explicitly import the names from the ``math`` module.

        if len(body) == 0:
            # If the Graph has no non-placeholder nodes, no lines for the body
            # have been emitted. To continue to have valid Python code, emit a
            # single pass statement
            body.append('pass\n')

        code = ''.join(body)
        code = '\n'.join('    ' + line for line in code.split('\n'))
        fn_code = f"""
def forward(self, {', '.join(free_vars)}){maybe_return_annotation[0]}:
{code}"""

        return PythonCode(fn_code,
                          self._globals.globals.copy(),
                          self._globals.format_import_block())

    def __str__(self) -> str:
        """
        Print a human-readable (not machine-readable) string representation
        of this Graph
        """
        placeholder_names : List[str] = []
        # This is a one-element array just so ``format_node`` can modify the closed
        # over value
        maybe_return_typename : List[str] = ['']

        node_strs = [node.format_node(placeholder_names) for node in self.nodes]
        param_str = ', '.join(placeholder_names)
        s = f'graph({param_str}){maybe_return_typename[0]}:'
        for node_str in node_strs:
            if node_str:
                s += '\n    ' + node_str
        return s

    def print_tabular(self):
        """
        Prints the intermediate representation of the graph in tabular
        format.
        """
        try:
            from tabulate import tabulate
        except ImportError:
            print("`print_tabular` relies on the library `tabulate`, "
                  "which could not be found on this machine. Run `pip "
                  "install tabulate` to install the library.")
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs]
                      for n in self.nodes]
        print(tabulate(node_specs,
              headers=['opcode', 'name', 'target', 'args', 'kwargs']))

    def lint(self, root : Optional[torch.nn.Module] = None):
        """
        Runs various checks on this Graph to make sure it is well-formed. In
        particular:
        - Checks Nodes have correct ownership (owned by this graph)
        - Checks Nodes appear in topological order
        - If ``root`` is provided, checks that targets exist in ``root``

        Args:

            root (Optional[torch.nn.Module]): The root module with which to check
                for targets. This is equivalent to the ``root`` argument that is
                passed when constructing a ``GraphModule``.
        """

        # Check topo order
        def check_arg(arg : Node, n : Optional[Node] = None) -> None:
            context_str = f' of Node \'{n}\' ' if n else ' '
            if arg.graph is not self:
                raise RuntimeError(f'Argument \'{arg}\'{context_str}does not belong to this Graph, '
                                   f'but was used as an argument! If you are copying nodes from another graph, make '
                                   f'sure to use ``arg_transform`` on node_copy() to remap values\n{self}')
            if arg not in seen_values:
                raise RuntimeError(f'Argument \'{arg}\'{context_str}was used before it has been '
                                   f'defined! Please check that Nodes in the graph are topologically ordered\n{self}')

        seen_names : Set[str] = set()
        seen_values : Set[Node] = set()
        for node in self.nodes:
            if node.op not in ['placeholder', 'call_method', 'call_module', 'call_function', 'get_attr', 'output']:
                raise RuntimeError(f'Node {node} had unknown opcode {node.op}!')
            if node.graph is not self:
                raise RuntimeError(f'Node \'{node}\' does not belong to this Graph!')
            map_arg(node.args, lambda arg: check_arg(arg, node))
            map_arg(node.kwargs, lambda arg: check_arg(arg, node))
            seen_values.add(node)

            if node.name in seen_names:
                raise RuntimeError(f'Node redefined name {node.name}!')
            seen_names.add(node.name)

        # Check targets are legit
        if root:
            for node in self.nodes:
                if node.op in ['get_attr', 'call_module']:
                    assert isinstance(node.target, str)
                    target_atoms = node.target.split('.')
                    m_itr = root
                    for i, atom in enumerate(target_atoms):
                        m_itr = getattr(m_itr, atom, None)
                        if m_itr is None:
                            seen_qualname = '.'.join(target_atoms[:i])
                            raise RuntimeError(f'Node {node} target {node.target} references nonexistent attribute '
                                               f'{atom} of {seen_qualname}')


reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'pow': '{} ** {}',
    'lshift': '{} << {}',
    'rshift': '{} >> {}',
    'and': '{} & {}',
    'or': '{} | {}',
    'xor': '{} ^ {}',
    'getitem': '{}[{}]'
}

magic_methods = dict({
    'eq': '{} == {}',
    'ne': '{} != {}',
    'lt': '{} < {}',
    'gt': '{} > {}',
    'le': '{} <= {}',
    'ge': '{} >= {}',
    'pos': '+{}',
    'neg': '-{}',
    'invert': '~{}'}, **reflectable_magic_methods)
