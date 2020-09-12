from .node import Node, Argument, Target

from typing import Callable, Any, List, Dict, Optional, Tuple
import builtins
import torch
import keyword

def _shadows_builtin_name(name: str) -> bool:
    return name in builtins.__dict__ or name in keyword.kwlist

def _is_magic(x: str) -> bool:
    return x.startswith('__') and x.endswith('__')

def snake_case(s: str) -> str:
    return ''.join(['_' + i.lower() if i.isupper() else i for i in s]).lstrip('_')

def _qualified_name(func: Callable[..., Any]) -> str:
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

class Graph:
    def __init__(self):
        self.nodes : List[Node] = []
        self._used_names : Dict[str, int] = {}  # base name -> number

    def _mark_uses(self, a: Argument):
        def add_use(n: Node):
            n.uses += 1
            return n
        map_arg(a, add_use)

    def create_node(self, op: str, target: Target,
                    args: Optional[Tuple[Argument, ...]] = None,
                    kwargs: Optional[Dict[str, Argument]] = None,
                    name: Optional[str] = None):
        assert op in ('call_function', 'call_method', 'get_param', 'call_module', 'placeholder')
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        self._mark_uses(args)
        self._mark_uses(kwargs)
        n = Node(self, name if name is not None else self._name(target), op, target, args, kwargs)
        self.nodes.append(n)
        return n

    def node_copy(self, node: Node, arg_transform: Callable[[Node], Argument] = lambda x: x) -> Node:
        """ copy a node from one graph into another. arg_transform needs to transform arguments from the graph of node
            to the graph of self"""
        args = map_arg(node.args, arg_transform)
        kwargs = map_arg(node.kwargs, arg_transform)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        if node.op == "placeholder":
            # Placeholder names are user-visible, so they should be copied as-is without normalizing them.
            name = node.name
        else:
            name = self._name(node.name)
        return self.create_node(node.op, node.target, args, kwargs, name)

    def output(self, result: Argument):
        self.result = result
        self._mark_uses(result)

    def _name(self, target: Target) -> str:
        if callable(target):
            op = target.__name__
        else:
            assert isinstance(target, str)
            op = target
            if _is_magic(op):
                op = op[2:-2]
        op = op.replace('.', '_')
        op = snake_case(op)

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

    def python_code(self, root_module: str) -> Tuple[str, str, List[str]]:
        free_vars: List[str] = []
        body: List[str] = []
        for node in self.nodes:
            if node.op == 'placeholder':
                assert isinstance(node.target, str)
                free_vars.append(node.target)
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
                qualified_name = _qualified_name(node.target)
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
            elif node.op == 'get_param':
                assert isinstance(node.target, str)
                body.append(f'{node.name} = {_format_target(root_module, node.target)}\n')
                continue
            raise NotImplementedError(f'node: {node.op} {node.target}')

        src = ''.join(body)
        return src, str(self.result), free_vars

    def __str__(self) -> str:
        placeholder_names : List[str] = []

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
                placeholder_names.append(n.target)
                return None
            elif n.op == 'get_param':
                return f'%{n.name} : [uses={n.uses}] = self.{n.target}'
            else:
                return f'%{n.name} : [uses={n.uses}] = {n.op}[target={n.target}](' \
                       f'args = {format_arg(n.args)}, kwargs = {format_arg(n.kwargs)})'


        node_strs = [format_node(node) for node in self.nodes]
        param_str = ', '.join(placeholder_names)
        s = f'graph({param_str}):'
        for node_str in node_strs:
            if node_str:
                s += '\n    ' + node_str
        if self.result:
            s += f'\n    return {format_arg(self.result)}'
        return s

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
