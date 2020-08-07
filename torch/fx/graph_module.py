import torch
import torch.overrides
import linecache
import inspect
import dis  # or dat
import operator
from types import FunctionType, CodeType


# normal exec loses the source code, however we can patch
# the linecache module to still recover it.
# using exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
_next_id = 0
def exec_with_source(src, globals):
    global _next_id
    key = f'<eval_with_key_{_next_id}>'
    _next_id += 1
    _eval_cache[key] = [line + '\n' for line in src.splitlines()]
    exec(compile(src, key, 'exec'), globals)

# patch linecache so that any code we exec using exec_with_source
# works with inspect
_eval_cache = {}
_orig_getlines = linecache.getlines
def patched_getline(*args, **kwargs):
    if args[0] in _eval_cache:
        return _eval_cache[args[0]]
    return _orig_getlines(*args, **kwargs)
linecache.getlines = patched_getline




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

HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS
def _patch_function(fn, nargs):
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    if hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs, 0,
            0, co.co_nlocals, co.co_stacksize,
            co_flags, co.co_code, co.co_consts, co.co_names,
            co.co_varnames, co.co_filename, co.co_name,
            co.co_firstlineno, co.co_lnotab, co.co_freevars,
            co.co_cellvars
        )
    else:
        co_args = (
            nargs, 0, co.co_nlocals,
            co.co_stacksize, co_flags, co.co_code, co.co_consts,
            co.co_names, co.co_varnames, co.co_filename,
            co.co_name, co.co_firstlineno, co.co_lnotab,
            co.co_freevars, co.co_cellvars)
    new_code = CodeType(*co_args)
    return FunctionType(new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)

    # we need to insert placeholder nodes for *args, and **kwargs,
    # so we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normay variables



class TraceError(ValueError):
    pass

# Nodes represent a definition of a value in our graph of operators.
# For simplicitly, they also have methods defined on them so that they serve as proxy objects for _building_
# more nodes.
class Node:
    def __init__(self, graph, name, op, target, args, kwargs):
        self.graph = graph
        self.name = name  # unique name of value being created
        self.op = op  # the kind of operation = placeholder|call_method|call_module|call_function|getattr
        self.target = target  # for method/module/function, the name of the method/module/function/attr
        # being invoked, e.g add, layer1, or torch.add
        self.args = args
        self.kwargs = kwargs
        self.uses = 0

    def __repr__(self):
        return self.name

    def __getattr__(self, k):
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __call__(self, *args, **kwargs):
        return self._create_node('call_method', '__call__', [self] + args, kwargs)

    def __iter__(self):
        frame = inspect.currentframe()
        calling_frame = frame.f_back
        inst = list(dis.get_instructions(calling_frame.f_code))[calling_frame.f_lasti // 2]
        if inst.opname == 'UNPACK_SEQUENCE':
            return (self[i] for i in range(inst.argval))
        self._no_control_flow()

    def _no_control_flow(self):
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')

    def __bool__(self):
        self._no_control_flow()

    def __torch_function__(self, orig_method, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return self.graph._create_node('call_method', orig_method.__name__, args, kwargs)
        else:
            return self.graph._create_node('call_function', orig_method, args, kwargs, name=self.graph._name(orig_method.__name__))

def _qualified_name(func):
    # things like getattr just appear in builtins
    if getattr(__builtins__, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
    return f'{module}.{name}'

class Attribute(Node):
    def __init__(self, node, attr):
        super().__init__(node.graph, node.graph._name(attr), 'call_function', getattr, [node, attr], {})

    def __call__(self, *args, **kwargs):
        return self.node.graph._create_node('call_method', self.args[1], [self.args[0]] + list(args), kwargs)

reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'pow': '{} ** {}',
    'lshift': '{} < {}',
    'rshift': '{} > {}',
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

for method in magic_methods:
    def scope(method):
        def impl(*args, **kwargs):
            g = args[0].graph
            target = getattr(operator, method)
            return g._create_node('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method}__'
        setattr(Node, as_magic, impl)
    scope(method)

for orig_method_name in reflectable_magic_methods:
    def scope(orig_method_name):
        method_name = f'__r{orig_method_name}__'

        def impl(self, rhs):
            target = getattr(operator, orig_method_name)
            return self.graph._create_node('call_function', target, [rhs, self])
        impl.__name__ = method_name
        impl.__qualname__ = method_name
        setattr(Node, method_name, impl)
    scope(orig_method_name)

def snake_case(s):
    return ''.join(['_' + i.lower() if i.isupper() else i for i in s]).lstrip('_')

def _is_magic(x):
    return x.startswith('__') and x.endswith('__')

def _reference_is_in(x, l):
    for elem in l:
        if x is elem:
            return True
    return False

def _find_module(root, m):
    for n, p in root.named_modules():
        if m is p:
            return n
    raise NameError('module is not installed as a submodule')

def is_leaf_module(m):
    return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

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
                if qualified_name == 'builtins.getattr' and isinstance(node.args[1], str) and node.args[1].isidentifier():
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

def _run_symbolic_forward(root, is_leaf_module):
    def _use_parameter(graph, a):
        if isinstance(a, torch.nn.Parameter):
            for n, p in root.named_parameters():
                if a is p:
                    return graph.get_param(n)
            raise NameError('parameter is not a member of this module')
        return NotImplemented
    graph = Graph(arg_handler=_use_parameter)
    fn = type(root).forward

    co = fn.__code__
    total_args = co.co_argcount + co.co_kwonlyargcount
    names_iter = iter(co.co_varnames)
    next(names_iter)  # skip self
    args = [root]
    args.extend(graph.placeholder(next(names_iter)) for name in range(1, total_args))

    if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
        if co.co_flags & inspect.CO_VARARGS:
            args.append(graph.placeholder('*' + next(names_iter)))
        if co.co_flags & inspect.CO_VARKEYWORDS:
            args.append(graph.placeholder('**' + next(names_iter)))
        fn = _patch_function(fn, len(args))

    args = tuple(args)
    orig_call = torch.nn.Module.__call__

    def module_call_wrapper(mod, *args, **kwargs):
        if not is_leaf_module(mod):
            return orig_call(mod, *args, **kwargs)
        else:
            target = _find_module(root, mod)
            return graph.call_module(target, args, kwargs)
    try:
        torch.nn.Module.__call__ = module_call_wrapper
        graph.output(fn(*args))
    finally:
        torch.nn.Module.__call__ = orig_call
    return graph


class GraphModule(torch.nn.Module):
    def __new__(cls, *args, **kwargs):
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method
        class GraphModuleImpl(cls):
            pass
        return super().__new__(GraphModuleImpl)

    def __init__(self, root, graph=None, is_leaf_module=is_leaf_module):
        super().__init__()
        self.root = root
        self.graph = graph if graph is not None else _run_symbolic_forward(root, is_leaf_module)
        self._generate_forward()

    def _generate_forward(self):
        body, result, free_variables = self.graph.python_code(root_module='self')
        body = '\n'.join('    ' + line for line in body.split('\n')) + '\n'
        self.src = f"""\
def forward(self, {', '.join(free_variables)}):
    self = self.root
{body}
    return {result}
"""
        # print(self.src)
        # install forward into the classes dictionary, this is what normally happens in the
        # 'class' statement
        # __new__ ensured that each instance has its own class
        gbls = {
            'torch': torch
        }
        exec_with_source(self.src, gbls)
        cls = type(self)
        for k, v in gbls.items():
            setattr(cls, k, v)

# workarounds for issues in __torch_function__

# WAR for __torch_function__ not handling tensor lists,
# fix is in https://github.com/pytorch/pytorch/pull/34725
# orig_cat = torch.cat
# def patched_cat(*args, **kwargs):
#     tensors = args[0]
#     for t in tensors:
#         if isinstance(t, Node):
#             return t.__torch_function__(patched_cat, (), args, kwargs)
#     return orig_cat(*args, **kwargs)
# patched_cat.__module__ = 'torch'
# patched_cat.__name__ = 'cat'
# torch.cat = patched_cat
