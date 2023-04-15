from .core import unify, reify  # type: ignore[attr-defined]
from .variable import isvar
from .utils import _toposort, freeze
from .unification_tools import groupby, first  # type: ignore[import]


class Dispatcher:
    def __init__(self, name):
        self.name = name
        self.funcs = {}
        self.ordering = []

    def add(self, signature, func):
        self.funcs[freeze(signature)] = func
        self.ordering = ordering(self.funcs)

    def __call__(self, *args, **kwargs):
        func, s = self.resolve(args)
        return func(*args, **kwargs)

    def resolve(self, args):
        n = len(args)
        for signature in self.ordering:
            if len(signature) != n:
                continue
            s = unify(freeze(args), signature)
            if s is not False:
                result = self.funcs[signature]
                return result, s
        raise NotImplementedError("No match found. \nKnown matches: "
                                  + str(self.ordering) + "\nInput: " + str(args))

    def register(self, *signature):
        def _(func):
            self.add(signature, func)
            return self
        return _


class VarDispatcher(Dispatcher):
    """ A dispatcher that calls functions with variable names
    >>> # xdoctest: +SKIP
    >>> d = VarDispatcher('d')
    >>> x = var('x')
    >>> @d.register('inc', x)
    ... def f(x):
    ...     return x + 1
    >>> @d.register('double', x)
    ... def f(x):
    ...     return x * 2
    >>> d('inc', 10)
    11
    >>> d('double', 10)
    20
    """
    def __call__(self, *args, **kwargs):
        func, s = self.resolve(args)
        d = {k.token: v for k, v in s.items()}
        return func(**d)


global_namespace = {}  # type: ignore[var-annotated]


def match(*signature, **kwargs):
    namespace = kwargs.get('namespace', global_namespace)
    dispatcher = kwargs.get('Dispatcher', Dispatcher)

    def _(func):
        name = func.__name__

        if name not in namespace:
            namespace[name] = dispatcher(name)
        d = namespace[name]

        d.add(signature, func)

        return d
    return _


def supercedes(a, b):
    """ ``a`` is a more specific match than ``b`` """
    if isvar(b) and not isvar(a):
        return True
    s = unify(a, b)
    if s is False:
        return False
    s = {k: v for k, v in s.items() if not isvar(k) or not isvar(v)}
    if reify(a, s) == a:
        return True
    if reify(b, s) == b:
        return False


# Taken from multipledispatch
def edge(a, b, tie_breaker=hash):
    """ A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
    if supercedes(a, b):
        if supercedes(b, a):
            return tie_breaker(a) > tie_breaker(b)
        else:
            return True
    return False


# Taken from multipledispatch
def ordering(signatures):
    """ A sane ordering of signatures to check, first to last
    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(first, edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    edges = {k: [b for a, b in v] for k, v in edges.items()}  # type: ignore[attr-defined, assignment]
    return _toposort(edges)
