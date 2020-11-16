from .node import Node, Argument, Target, map_arg

from typing import Callable, Any, List, Dict, Optional, Tuple, Set
import builtins
import torch
import types
import keyword
import re

def _shadows_builtin_name(name: str) -> bool:
    return name in builtins.__dict__ or name in keyword.kwlist or name in {'inf', 'nan'}

def _is_magic(x: str) -> bool:
    return x.startswith('__') and x.endswith('__')

def _snake_case(s: str) -> str:
    return ''.join(['_' + i.lower() if i.isupper() else i for i in s]).lstrip('_')

def get_qualified_name(func: Callable[..., Any]) -> str:
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
    return f'{module}.{name}'

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

# Borrowed from CPython typing module
# https://github.com/python/cpython/blob/f90dc36c15d7fee0efaf6d39e97be0bdf2683e93/Lib/typing.py#L156
def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    # HACK: In Python 3.6, type aliases from `typing` are instances of `type`, but in
    # later Python versions, type aliases are not instances of `type`!! We want
    # all type aliases to fall through to `repr`, so if we have a type that is
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
    `Graph` is the main data structure used in the FX Intermediate Representation.
    It consists of a series of `Node`s, each representing callsites (or other
    syntactic constructs). The list of `Node`s, taken together, constitute a
    valid Python function.

    For example, the following code

    ```
    import torch
    from torch.fx import symbolic_trace

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x):
            return torch.topk(torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3)

    m = MyModule()
    gm = symbolic_trace(m)
    ```

    Will produce the following Graph:

    ```
    print(gm.graph)
    ```

    ```
    graph(x):
        %linear_weight : [uses=1] = self.linear.weight
        %add_1 : [uses=1] = call_function[target=<built-in function add>](args = (%x, %linear_weight), kwargs = {})
        %linear_1 : [uses=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
        %relu_1 : [uses=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
        %sum_1 : [uses=1] = call_function[target=<built-in method sum of type object at 0x7fad0a3c16a0>](args = (%relu_1,), kwargs = {dim: -1}) # noqa: B950
        %topk_1 : [uses=1] = call_function[target=<built-in method topk of type object at 0x7fad0a3c16a0>](args = (%sum_1, 3), kwargs = {}) # noqa: B950
        return topk_1
    ```

    The Node semantics are as follows:

    - `placeholder` represents a function input. The `name` attribute specifies the name this value will take on.
    `target` is similarly the name of the argument. `args` and `kwargs` are don't-care. Placeholders correspond to
    the function parameters (e.g. `x`) in the graph printout.
    - `get_attr` retrieves a parameter from the module hierarchy. `name` is similarly the name the result of the
    fetch is assigned to. `target` is the fully-qualified name of the parameter's position in the module hierarchy.
    `args` and `kwargs` are don't-care
    - `call_function` applies a free function to some values. `name` is similarly the name of the value to assign
    to. `target` is the function to be applied. `args` and `kwargs` represent the arguments to the function,
    following the Python calling convention
    - `call_module` applies a module in the module hierarchy's `forward()` method to given arguments. `name` is
    as previous. `target` is the fully-qualified name of the module in the module hierarchy to call.
    `args` and `kwargs` represent the arguments to invoke the module on, _including the self argument_.
    - `call_method` calls a method on a value. `name` is as similar. `target` is the string name of the method
    to apply to the `self` argument. `args` and `kwargs` represent the arguments to invoke the module on,
    _including the self argument_.
    - `output` contains the output of the traced function in its `args[0]` attribute. This corresponds to the "return" statement
    in the Graph printout.
    """
    def __init__(self):
        """
        Construct an empty Graph.
        """
        self._root : Node = Node(self, '', 'root', '', (), {})
        self._used_names : Dict[str, int] = {}  # base name -> number
        self._insert = self._root.prepend
        self._len = 0

    @property
    def nodes(self) -> _node_list:
        """
        Get the list of `Node`s that constitute this Graph.

        Note that this `Node` list representation is a doubly-linked list. Mutations
        during iteration (e.g. delete a Node, add a Node) are safe.
        """
        return _node_list(self)

    def graph_copy(self, g : 'Graph', val_map : Dict[Node, Node]) -> Optional[Argument]:
        """
        Append all nodes from graph `g` to this graph. `val_map` should be a dictionary
        that maps nodes in `g` to nodes in `self. `val_map` will be populated with more
        items by this function. Returns the equivalent output value of `g` with
        Nodes switched to refer to nodes in `self`.
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

    def create_node(self, op: str, target: Target,
                    args: Optional[Tuple[Argument, ...]] = None,
                    kwargs: Optional[Dict[str, Argument]] = None,
                    name: Optional[str] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Create a `Node` and add it to the `Graph` at the current insert-point.
        Note that the current insert-point can be set via `Graph.inserting_before`
        and `Graph.inserting_after`.

        - op is the opcode for this Node. One of 'call_function', 'call_method', 'get_attr',
          'call_module', 'placeholder', or 'output'. The semantics of these opcodes are
          described in the `Graph` docstring.
        - args is a tuple of arguments to this node.
        - kwargs is a dict from string to argument, representing the kwargs of this Node
        - name is an optional string name for the `Node`. This will influence the name
          of the value assigned to in the Python generated code.
        - type_expr is an optional type annotation representing the Python type
          the output of this node will have.
        """
        assert op in ('call_function', 'call_method', 'get_attr', 'call_module', 'placeholder', 'output')
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        sanitized_name = self._register_name_used(name) if name is not None else self._name(target)
        n = Node(self, sanitized_name, op, target, args, kwargs, type_expr)
        self._insert(n)
        self._len += 1
        return n

    def erase_node(self, to_erase : Node):
        """
        Erases the node `to_erase` from the `Graph`. Throws an exception if
        there are still users of that node in the `Graph`.
        """
        if len(to_erase.users) > 0:
            raise RuntimeError(f'Tried to erase Node {to_erase} but it still had {len(to_erase.users)} '
                               f'users in the graph: {to_erase.users}!')

        to_erase._remove_from_list()
        to_erase._erased = True  # iterators may retain handles to erased nodes
        self._len -= 1

        # Null out this Node's argument nodes so that the Nodes referred to
        # can update their `users` accordingly
        new_args = map_arg(to_erase.args, lambda n: None)
        assert isinstance(new_args, tuple)
        to_erase.args = new_args
        new_kwargs = map_arg(to_erase.kwargs, lambda n: None)
        assert isinstance(new_kwargs, dict)
        to_erase.kwargs = new_kwargs

    def inserting_before(self, n: Optional[Node] = None):
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits:

            with g.inserting_before(n):
                ... # inserting before node n
            ... # insert point restored to what it was previously
            g.inserting_before(n) #  set the insert point permanently

        Args:
            n (Optional[Node]): The node before which to insert. If None this will insert before
              the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on `__exit__`.
        """
        if n is None:
            return self.inserting_after(self._root)
        assert n.graph == self, "Node to insert before is not in graph."
        return _InsertPoint(self, n.prepend)

    def inserting_after(self, n: Optional[Node] = None):
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits:

            with g.inserting_after(n):
                ... # inserting after node n
            ... # insert point restored to what it was previously
            g.inserting_after(n) #  set the insert point permanently

        Args:
            n (Optional[Node]): The node before which to insert. If None this will insert after
              the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on `__exit__`.
        """
        if n is None:
            return self.inserting_before(self._root)
        assert n.graph == self, "Node to insert after is not in graph."
        return _InsertPoint(self, n.append)

    # sugar for create_node when you know the op
    def placeholder(self, name: str, type_expr: Optional[Any] = None) -> Node:
        """
        Insert a `placeholder` node into the Graph. A `placeholder` represents
        a function input. This function takes a string `name` for the input
        value as well as an optional `type_expr`, which is a type expression
        describing the type of value this input will take. The type expression
        is needed in some cases for proper code generation.

        The same insertion point rules apply for this method as `Graph.create_node`.
        """
        return self.create_node('placeholder', name, type_expr=type_expr)

    def get_attr(self, qualified_name: str, type_expr: Optional[Any] = None) -> Node:
        """
        Insert a `get_attr` node into the Graph. A `get_attr` `Node` represents the
        fetch of an attribute from the `Module` hierarchy. `qualified_name` is the
        fully-qualified name of the attribute to be retrieved. For example, if
        the traced Module has a submodule named `foo`, which has a submodule named
        `bar`, which has an attribute named `baz`, the qualified name `foo.bar.baz`
        should be passed as `qualified_name`.

        The same insertion point and type expression rules apply for this method
        as `Graph.create_node`.
        """
        return self.create_node('get_attr', qualified_name, type_expr=type_expr)

    def call_module(self,
                    module_name: str,
                    args: Optional[Tuple[Argument, ...]] = None,
                    kwargs: Optional[Dict[str, Argument]] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Insert a `call_module` `Node` into the `Graph`. A `call_module` node
        represents a call to the forward() function of a `Module` in the `Module`
        hierarchy. For example, if the traced `Module` has a submodule named `foo`,
        which has a submodule named `bar`, the qualified name `foo.bar` should
        be passed as `module_name` to call that module.

        `args` and `kwargs` represent the args and kwargs passed to the called
        `Module`, respectively.

        The same insertion point and type expression rules apply for this method
        as `Graph.create_node`.
        """
        return self.create_node('call_module', module_name, args, kwargs, type_expr=type_expr)

    def call_method(self,
                    method_name: str,
                    args: Optional[Tuple[Argument, ...]] = None,
                    kwargs: Optional[Dict[str, Argument]] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Insert a `call_method` `Node` into the `Graph`. A `call_method` node
        represents a call to a given method on the 0th element of `args.
        For example, if args[0] is a `Node` representing a `Tensor`, then to call
        `relu()` on that `Tensor`, pass `relu` to `method_name`.

        `args` and `kwargs` represent the args and kwargs passed to the called
        method, respectively.

        The same insertion point and type expression rules apply for this method
        as `Graph.create_node`.
        """
        return self.create_node('call_method', method_name, args, kwargs, type_expr=type_expr)

    def call_function(self,
                      the_function: Callable[..., Any],
                      args: Optional[Tuple[Argument, ...]] = None,
                      kwargs: Optional[Dict[str, Argument]] = None,
                      type_expr: Optional[Any] = None) -> Node:
        """
        Insert a `call_function` `Node` into the `Graph`. A `call_function` node
        represents a call to a Python callable, specified by `the_function`. `the_function`
        can be any PyTorch operator, Python function, or member of the `builtins`
        or `operator` namespaces.

        `args` and `kwargs` represent the args and kwargs passed to the called
        method, respectively.

        The same insertion point and type expression rules apply for this method
        as `Graph.create_node`.
        """
        return self.create_node('call_function', the_function, args, kwargs, type_expr=type_expr)

    def node_copy(self, node: Node, arg_transform: Callable[[Node], Argument] = lambda x: x) -> Node:
        """ Copy a node from one graph into another. arg_transform needs to transform arguments from the graph of node
            to the graph of self. Example:

            g : torch.fx.Graph = ...
            new_graph = torch.fx.graph()
            value_remap = {}
            for node in g.nodes:
                value_remap[node] = new_graph.node_copy(node, lambda n : value_remap[n])
        """
        args = map_arg(node.args, arg_transform)
        kwargs = map_arg(node.kwargs, arg_transform)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        if node.op == "placeholder":
            # Placeholder names are user-visible, so they should be copied as-is without normalizing them.
            name = node.name
        else:
            sanitized_name = node.name
            if '_' in node.name:
                base, maybe_idx = node.name.rsplit('_', 1)
                if base != '':
                    try:
                        int(maybe_idx)
                        sanitized_name = base
                    except ValueError:
                        pass
            name = self._name(sanitized_name)
        return self.create_node(node.op, node.target, args, kwargs, name, node.type)

    def output(self, result: Argument, type_expr: Optional[Any] = None):
        """
        Insert an `output` `Node` into the `Graph`. An `output` node represents
        a `return` statement in the Python code. `result` is the value that should
        be returned.

        The same insertion point and type expression rules apply for this method
        as `Graph.create_node`.
        """
        return self.create_node(op='output', target='output', args=(result,), type_expr=type_expr)

    def _name(self, target: Target) -> str:
        if callable(target):
            op = target.__name__
        else:
            assert isinstance(target, str)
            op = target
            if _is_magic(op):
                op = op[2:-2]
        op = op.replace('.', '_')
        # delete all characters that are illegal in a Python identifier
        op = re.sub('[^0-9a-zA-Z_]+', '_', op)
        op = _snake_case(op)
        if op[0].isdigit():
            op = f'_{op}'

        return self._register_name_used(op)

    def _register_name_used(self, op : str) -> str:
        """
        Even if a user provides us with a name, we must register that that
        name is used to prevent duplication of names from further nodes as
        well as ensure that the name provided does not shadow a builtin.
        """
        if op not in self._used_names:
            self._used_names[op] = 0
            # Avoid shadowing PyTorch and Python builtins.
            if not hasattr(torch, op) and \
               not hasattr(torch.nn.functional, op) and \
               not hasattr(torch.nn, op) and \
               not _shadows_builtin_name(op):
                return op
        i = self._used_names[op] = self._used_names[op] + 1
        return f'{op}_{i}'

    def python_code(self, root_module: str) -> str:
        """
        Turn this `Graph` into valid Python code.
        """
        free_vars: List[str] = []
        modules_used : Set[str] = set()
        body: List[str] = []
        maybe_return_annotation : str = ''

        def register_modules_used(qualified_name : str):
            if '.' in qualified_name:
                module_name = qualified_name.split('.', maxsplit=1)[0]
                modules_used.add(module_name)

        def type_repr(o : Any):
            typename = _type_repr(o)
            if all(x.isidentifier() for x in typename.split('.')):
                register_modules_used(typename)
            else:
                # this is a constructor type, e.g. typing.List[torch.Tensor]
                modules_used.add(o.__module__)
                for sub_type in o.__args__:
                    # make sure we have torch.Tensor
                    type_repr(sub_type)
            return typename

        for node in self.nodes:
            if node.op == 'placeholder':
                assert isinstance(node.target, str)
                maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
                maybe_default_arg = '' if not node.args else f' = {repr(node.args[0])}'
                free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
                raw_name = node.target.replace('*', '')
                if raw_name != node.name:
                    body.append(f'{node.name} = {raw_name}\n')
                continue
            elif node.op == 'call_method':
                assert isinstance(node.target, str)
                body.append(
                    f'{node.name} = {_format_target(repr(node.args[0]), node.target)}'
                    f'({_format_args(node.args[1:], node.kwargs)})\n')
                continue
            elif node.op == 'call_function':
                assert callable(node.target)
                # pretty print operators
                if node.target.__module__ == '_operator' and node.target.__name__ in magic_methods:
                    assert isinstance(node.args, tuple)
                    body.append(f'{node.name} = {magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}\n')
                    continue
                qualified_name = get_qualified_name(node.target)
                register_modules_used(qualified_name)
                if qualified_name == 'getattr' and \
                   isinstance(node.args, tuple) and \
                   isinstance(node.args[1], str) and \
                   node.args[1].isidentifier():
                    # pretty print attribute access
                    body.append(f'{node.name} = {_format_target(repr(node.args[0]), node.args[1])}\n')
                    continue
                body.append(f'{node.name} = {qualified_name}({_format_args(node.args, node.kwargs)})\n')
                continue
            elif node.op == 'call_module':
                assert isinstance(node.target, str)
                body.append(f'{node.name} = {_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})\n')
                continue
            elif node.op == 'get_attr':
                assert isinstance(node.target, str)
                body.append(f'{node.name} = {_format_target(root_module, node.target)}\n')
                continue
            elif node.op == 'output':
                if node.type is not None:
                    maybe_return_annotation = f" -> {type_repr(node.type)}"
                body.append(f'return {repr(node.args[0])}')
                continue
            raise NotImplementedError(f'node: {node.op} {node.target}')

        # repr() for inf and nan floating point values aren't parseable by
        # python as literals. Explicitly import the names from the `math` module.
        import_strs = [f'import {name}' for name in sorted(modules_used)]
        import_block = '\n'.join(import_strs)

        code = ''.join(body)
        code = '\n'.join('    ' + line for line in code.split('\n')) + '\n'
        fn_code = f"""\
{import_block}
def forward(self, {', '.join(free_vars)}){maybe_return_annotation}:
{code}
"""

        return fn_code

    def __str__(self) -> str:
        """
        Print a human-readable (not machine-readable) string representation
        of this Graph
        """
        placeholder_names : List[str] = []
        # This is a one-element array just so `format_node` can modify the closed
        # over value
        maybe_return_typename : List[str] = ['']

        def format_arg(arg) -> str:
            if isinstance(arg, list):
                items = ', '.join(format_arg(a) for a in arg)
                return f'[{items}]'
            elif isinstance(arg, tuple):
                items = ', '.join(format_arg(a) for a in arg)
                maybe_comma = ',' if len(arg) == 1 else ''
                return f'({items}{maybe_comma})'
            elif isinstance(arg, dict):
                items_str = ', '.join(f'{k}: {format_arg(v)}' for k, v in arg.items())
                return f'{{{items_str}}}'

            if isinstance(arg, Node):
                return '%' + str(arg)
            else:
                return str(arg)

        def format_node(n : Node) -> Optional[str]:
            if n.op == 'placeholder':
                assert isinstance(n.target, str)
                arg_str = n.target
                arg_str += arg_str + f': {_type_repr(n.type)}' if n.type is not None else ''
                placeholder_names.append(arg_str)
                return None
            elif n.op == 'get_attr':
                maybe_typename = f'{_type_repr(n.type)} ' if n.type is not None else ''
                return f'%{n.name} : {maybe_typename}[#users={len(n.users)}] = self.{n.target}'
            elif n.op == 'output':
                if n.type is not None:
                    maybe_return_typename[0] = f' -> {_type_repr(n.type)}'
                return f'return {n.args[0]}'
            else:
                maybe_typename = f'{_type_repr(n.type)} ' if n.type is not None else ''
                return f'%{n.name} : {maybe_typename}[#users={len(n.users)}] = {n.op}[target={n.target}](' \
                       f'args = {format_arg(n.args)}, kwargs = {format_arg(n.kwargs)})'


        node_strs = [format_node(node) for node in self.nodes]
        param_str = ', '.join(placeholder_names)
        s = f'graph({param_str}){maybe_return_typename[0]}:'
        for node_str in node_strs:
            if node_str:
                s += '\n    ' + node_str
        return s

    def lint(self, root : Optional[torch.nn.Module] = None):
        """
        Runs various checks on this Graph to make sure it is well-formed. In
        particular:
            - Checks Nodes have correct ownership (owned by this graph)
            - Checks Nodes appear in topological order
            - If `root` is provided, checks that `target`s exist in `root`
        """

        # Check topo order
        def check_arg(arg : Node, n : Optional[Node] = None) -> None:
            context_str = f' of Node \'{n}\' ' if n else ' '
            if arg.graph is not self:
                raise RuntimeError(f'Argument \'{arg}\'{context_str}does not belong to this Graph, '
                                   f'but was used as an argument! If you are copying nodes from another graph, make '
                                   f'sure to use `arg_transform` on node_copy() to remap values\n{self}')
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
