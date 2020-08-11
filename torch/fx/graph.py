from .node import Node, Attribute, magic_methods

import builtins
import torch

def _is_magic(x):
    return x.startswith('__') and x.endswith('__')

def snake_case(s):
    return ''.join(['_' + i.lower() if i.isupper() else i for i in s]).lstrip('_')

def _reference_is_in(x, l):
    for elem in l:
        if x is elem:
            return True
    return False

def _qualified_name(func):
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
    return f'{module}.{name}'

# this is fixed on master, WAR for 1.5
def _find_module_of_method(orig_method):
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')

def _format_args(args, kwargs):
    args_s = ', '.join(repr(a) for a in args)
    kwargs_s = ', '.join(f'{k} = {repr(v)}' for k, v in kwargs.items())
    if args_s and kwargs_s:
        return f'{args_s}, {kwargs_s}'
    return args_s or kwargs_s

def _format_target(base, target):
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f'{r}.{e}'
    return r

def map_arg(a, fn):
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
    def __init__(self, arg_handler=None):
        self.nodes = []
        self._used_names = {}  # base name -> number
        self.arg_handler = arg_handler

    def _create_node(self, op, target=None, args=None, kwargs=None, name=None):
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        args = self._create_args(args)
        kwargs = self._create_args(kwargs)
        n = Node(self, name if name is not None else self._name(target or op), op, target, args, kwargs)
        self.nodes.append(n)
        return n

    def _create_args(self, a):
        # aggregates
        if isinstance(a, (tuple, list)):
            return type(a)(self._create_args(elem) for elem in a)
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                if not isinstance(k, str):
                    raise NotImplementedError(f"dictionaries with non-string keys: {a}")
                r[k] = self._create_args(v)
            return r
        elif isinstance(a, slice):
            return slice(self._create_args(a.start), self._create_args(a.stop), self._create_args(a.step))

        # individual elements
        r = NotImplemented
        if self.arg_handler is not None:
            r = self.arg_handler(self, a)

        if r is NotImplemented:
            if isinstance(a, Attribute):
                if not _reference_is_in(a, self.nodes):
                    self.nodes.append(a)
                r = a
            elif isinstance(a, (str, int, float, bool, Node, torch.dtype, torch.Tensor)) or a is None:
                r = a

        if r is NotImplemented:
            raise NotImplementedError(f"argument of type: {type(a)}")

        if isinstance(r, Node):
            assert r.graph is self
            r.uses += 1
        return r

    def node_copy(self, node, arg_transform=lambda x: x):
        """ copy a node from one graph into another. arg_transform needs to transform arguments from the graph of node
            to the graph of self"""
        return self._create_node(
            node.op, node.target, map_arg(node.args, arg_transform), map_arg(node.kwargs, arg_transform),
            self._name(node.name))

    def output(self, result):
        self.result = result


    def _name(self, op):
        if hasattr(op, '__name__'):
            op = op.__name__

        if _is_magic(op):
            op = op[2:-2]
        op = op.replace('.', '_')
        op = snake_case(op)

        if op not in self._used_names:
            self._used_names[op] = 0
            if not hasattr(torch, op) and not hasattr(torch.nn.functional, op) and not hasattr(torch.nn, op):
                return op
        i = self._used_names[op] = self._used_names[op] + 1
        return f'{op}_{i}'

    def get_param(self, target):
        return self._create_node('get_param', target)

    def placeholder(self, name):
        return self._create_node('placeholder', target=name, name=name.replace('*', ''))

    def call_module(self, target, args, kwargs):
        return self._create_node('call_module', target, args, kwargs)

    def call_function(self, target, args, kwargs):
        return self._create_node('call_function', target, args, kwargs)

    def python_code(self, root_module):
        free_vars = []
        body = []
        for node in self.nodes:
            if node.op == 'placeholder':
                free_vars.append(node.target)
                continue
            elif node.op == 'call_method':
                body.append(
                    f'{node.name} = {_format_target(repr(node.args[0]), node.target)}({_format_args(node.args[1:], node.kwargs)})\n')
                continue
            elif node.op == 'call_function':
                # pretty print operators
                if node.target.__module__ == '_operator' and node.target.__name__ in magic_methods:
                    body.append(f'{node.name} = {magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}\n')
                    continue
                qualified_name = _qualified_name(node.target)
                if qualified_name == 'getattr' and isinstance(node.args[1], str) and node.args[1].isidentifier():
                    # pretty print attribute access
                    body.append(f'{node.name} = {_format_target(repr(node.args[0]), node.args[1])}\n')
                    continue
                body.append(f'{node.name} = {qualified_name}({_format_args(node.args, node.kwargs)})\n')
                continue
            elif node.op == 'call_module':
                body.append(f'{node.name} = {_format_target(root_module,node.target)}({_format_args(node.args, node.kwargs)})\n')
                continue
            elif node.op == 'get_param':
                body.append(f'{node.name} = {_format_target(root_module, node.target)}\n')
                continue
            raise NotImplementedError(f'node: {node.op} {node.target}')

        src = ''.join(body)
        return src, str(self.result), free_vars
