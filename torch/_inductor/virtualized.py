from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from threading import local
from typing import List, Sequence, Union

import sympy

from torch.fx.graph import inplace_methods, magic_methods

from .utils import sympy_str, sympy_symbol

threadlocal = local()


class Virtualized:
    """
    A global variable that redirects via thread local variable

    This allows us to swap in different op implementations in codegen.
    """

    def __init__(self, vname, default):
        self._key = f"__torchinductor_{vname}"
        self._default = default

    def _set_handler(self, value):
        prior = self._get_handler()
        setattr(threadlocal, self._key, value)

        @contextmanager
        def ctx():
            try:
                yield
            finally:
                self._set_handler(prior)

        return ctx()

    def _get_handler(self):
        try:
            return getattr(threadlocal, self._key)
        except AttributeError:
            return self._default()

    def __getattr__(self, name):
        return getattr(self._get_handler(), name)


class NullHandler:
    pass


def _arg_str(a):
    if isinstance(a, sympy.Expr):
        return sympy_str(a)
    return str(a)


class MockHandler:
    def __getattr__(self, name):
        def inner(*args, **kwargs):
            fargs = [_arg_str(a) for a in args]
            fargs.extend(f"{k}={v}" for k, v in kwargs.items())
            return self.truncate_expr(f"{name}({', '.join(fargs)})")

        return inner

    @staticmethod
    def truncate_expr(expr):
        return expr

    @classmethod
    def masked(cls, mask, body, other):
        return cls.truncate_expr(f"masked({mask}, {body()}, {other})")

    @staticmethod
    def indirect_indexing(index_var):
        return sympy_symbol(str(index_var))

    @classmethod
    def _init_cls(cls):
        def make_handler(format_string):
            @staticmethod
            def inner(*args):
                return format_string.format(*args)

            return inner

        for name, format_string in chain(
            magic_methods.items(), inplace_methods.items()
        ):
            setattr(cls, name, make_handler(format_string))


class KernelBuilder:
    @dataclass(eq=False)
    class Expr:
        var_name: str
        format_string: str
        inputs: Sequence[Union["KernelBuilder.Expr", str]]

        def format_line(self):
            arg_strings = [
                i.var_name if isinstance(i, KernelBuilder.Expr) else i
                for i in self.inputs
            ]
            expression = self.format_string.format(*arg_strings)
            return f"{self.var_name} = {expression}\n"

    var_counter: int
    program: List["KernelBuilder.Expr"]

    def __init__(self):
        self.var_counter = 0
        self.program = []

    def _Expr(self, format_string, inputs):
        # Assign unique variable name
        var_name = f"tmp{self.var_counter}"
        self.var_counter += 1
        inputs = tuple(
            i if isinstance(i, (KernelBuilder.Expr, str)) else _arg_str(i)
            for i in inputs
        )
        expr = KernelBuilder.Expr(
            var_name=var_name, format_string=format_string, inputs=inputs
        )
        self.program.append(expr)
        return expr

    def __getattr__(self, fn_name):
        def inner(*args):
            placeholders = ", ".join(["{}"] * len(args))
            format_string = f"{fn_name}({placeholders})"
            return self._Expr(format_string=format_string, inputs=args)

        return inner

    def masked(self, mask, body, other):
        body = body()
        return self._Expr(
            format_string="masked({}, {}, {})",
            inputs=[mask, body, other],
        )

    @staticmethod
    def indirect_indexing(index_var):
        return sympy_symbol(str(index_var))

    @classmethod
    def _init_cls(cls):
        def make_handler(format_string):
            def inner(self, *args):
                return self._Expr(format_string=format_string, inputs=args)

            return inner

        for name, format_string in chain(
            magic_methods.items(), inplace_methods.items()
        ):
            setattr(cls, name, make_handler(format_string))

    def compile(self, result):
        if not isinstance(result, KernelBuilder.Expr):
            return _arg_str(result)

        # Traverse expression tree from result to find live code
        stack = [result]
        live_exprs = set((result,))
        while stack:
            expr = stack.pop()

            unseen_inputs = set(
                i
                for i in expr.inputs
                if isinstance(i, KernelBuilder.Expr) and i not in live_exprs
            )
            live_exprs.update(unseen_inputs)
            stack += unseen_inputs

        # Remove the dead code
        live_program = [e for e in self.program if e in live_exprs]

        # Output the final string
        return (
            "".join(e.format_line() for e in live_program)
            + f"return {result.var_name}\n"
        )


class WrapperHandler:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)


MockHandler._init_cls()
KernelBuilder._init_cls()

ops = Virtualized("ops", MockHandler)
_graph = Virtualized("graph", NullHandler)
_kernel = Virtualized("kernel", NullHandler)
_debug = Virtualized("debug", NullHandler)


class _V:
    MockHandler = MockHandler
    KernelBuilder = KernelBuilder
    WrapperHandler = WrapperHandler

    set_ops_handler = ops._set_handler
    get_ops_handler = ops._get_handler
    set_graph_handler = _graph._set_handler
    set_kernel_handler = _kernel._set_handler
    set_debug_handler = _debug._set_handler

    @property
    def ops(self) -> MockHandler:
        """The operator handler specific to the current codegen task"""
        return ops._get_handler()

    @property
    def graph(self):
        """The graph currently being generated"""
        return _graph._get_handler()

    @property
    def kernel(self):
        """The kernel currently being generated"""
        return _kernel._get_handler()

    @property
    def debug(self):
        return _debug._get_handler()


V = _V()
