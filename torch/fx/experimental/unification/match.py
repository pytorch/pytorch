from __future__ import annotations

from typing import TYPE_CHECKING

from .core import reify, unify  # type: ignore[attr-defined]


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .variable import Var

from .unification_tools import first, groupby  # type: ignore[import]
from .utils import _toposort, freeze
from .variable import isvar


class Dispatcher:
    def __init__(self, name: str) -> None:
        self.name = name
        self.funcs: dict[object, Callable[..., object]] = {}
        self.ordering: list[object] = []

    def add(self, signature: tuple[object, ...], func: Callable[..., object]) -> None:
        self.funcs[freeze(signature)] = func
        self.ordering = ordering(self.funcs)

    def __call__(self, *args: object, **kwargs: object) -> object:
        func, _ = self.resolve(args)
        return func(*args, **kwargs)

    def resolve(
        self, args: tuple[object, ...]
    ) -> tuple[Callable[..., object], dict[Var, object]]:
        n = len(args)
        for signature in self.ordering:
            if len(signature) != n:  # pyrefly: ignore[bad-argument-type]
                continue
            s = unify(freeze(args), signature)
            if s is not False:
                result = self.funcs[signature]
                return result, s  # pyrefly: ignore[bad-return]
        raise NotImplementedError(
            "No match found. \nKnown matches: "
            + str(self.ordering)
            + "\nInput: "
            + str(args)
        )

    def register(self, *signature: object) -> Callable[..., object]:
        def _(func: Callable[..., object]) -> Dispatcher:
            self.add(signature, func)
            return self

        return _


class VarDispatcher(Dispatcher):
    """A dispatcher that calls functions with variable names
    >>> # xdoctest: +SKIP
    >>> d = VarDispatcher("d")
    >>> x = var("x")
    >>> @d.register("inc", x)
    ... def f(x):
    ...     return x + 1
    >>> @d.register("double", x)
    ... def f(x):
    ...     return x * 2
    >>> d("inc", 10)
    11
    >>> d("double", 10)
    20
    """

    def __call__(self, *args: object, **kwargs: object) -> object:
        func, s = self.resolve(args)
        d = {k.token: v for k, v in s.items()}  # pyrefly: ignore[missing-attribute]
        return func(**d)


global_namespace: dict[str, Dispatcher] = {}


def match(*signature: object, **kwargs: object) -> Callable[..., object]:
    namespace: dict[str, Dispatcher] = kwargs.get(  # type: ignore[assignment]
        "namespace", global_namespace
    )
    dispatcher_cls: type[Dispatcher] = kwargs.get(  # type: ignore[assignment]
        "Dispatcher", Dispatcher
    )

    def _(func: Callable[..., object]) -> Dispatcher:
        name = func.__name__

        if name not in namespace:
            namespace[name] = dispatcher_cls(name)
        d = namespace[name]

        d.add(signature, func)

        return d

    return _


def supercedes(a: object, b: object) -> bool:
    """``a`` is a more specific match than ``b``"""
    if isvar(b) and not isvar(a):
        return True
    s = unify(a, b)
    if s is False:
        return False
    s = {
        k: v
        for k, v in s.items()  # pyrefly: ignore[missing-attribute]
        if not isvar(k) or not isvar(v)
    }
    if reify(a, s) == a:
        return True
    if reify(b, s) == b:
        return False
    return False


# Taken from multipledispatch
def edge(a: object, b: object, tie_breaker: Callable[[object], int] = hash) -> bool:
    """A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
    if supercedes(a, b):
        if supercedes(b, a):
            return tie_breaker(a) > tie_breaker(b)
        else:
            return True
    return False


# Taken from multipledispatch
def ordering(signatures: Iterable[object]) -> list[object]:
    """A sane ordering of signatures to check, first to last
    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))  # pyrefly: ignore[bad-argument-type]
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(first, edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    edges = {k: [b for a, b in v] for k, v in edges.items()}  # type: ignore[attr-defined, assignment]
    return _toposort(edges)  # pyrefly: ignore[bad-argument-type]
