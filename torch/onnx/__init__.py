import functools
import types

import torch
from torch.autograd import Function, function


def export(*args, **kwargs):
    import torch.onnx.utils
    return torch.onnx.utils.export(*args, **kwargs)


def _export(*args, **kwargs):
    import torch.onnx.utils
    return torch.onnx.utils._export(*args, **kwargs)


def _symbolic_override_wrapper_maker(symbolic_fn, do_skip, fn):

    def wrapper(*args, **kwargs):
        output = fn(*args, **kwargs)
        # fast path
        import torch.jit
        if do_skip(args):
            return output

        flat_args = tuple(function._iter_variables(args))
        if not any(map(torch._C._jit_is_tracing, flat_args)):
            return output
        flat_output_tensors = tuple(
            v.data for v in function._iter_variables(output))
        # TODO: kwargs aren't traced

        class ExportProxy(Function):
            @staticmethod
            def symbolic(g, *flat_args):
                symbolic_args = function._unflatten(flat_args, args)
                symbolic_output = symbolic_fn(g, *symbolic_args, **kwargs)
                return tuple(function._iter_jit_values(symbolic_output))

            @staticmethod
            def forward(ctx, *unused_args):
                return flat_output_tensors

            @staticmethod
            def backward(ctx, *unused_args, **unused_kwargs):
                raise RuntimeError(
                    "symbolic_override is meant for inference export only")

        flat_proxy_output = ExportProxy.apply(*flat_args)
        return function._unflatten(flat_proxy_output, output)

    # fn might be autograd.Function too, in this case wrapping doesn't work
    if isinstance(fn, types.FunctionType):
        wrapper = functools.wraps(fn)(wrapper)

    return wrapper


def symbolic_override(symbolic_fn):
    r"""
    Decorator to override ONNX export of the a function with specified subgraph.

    Effectively allows to attach symbolic() implementation to an arbitrary
    python function or autograd.Function. Requirements for the decorated
    function:
     - being non-member function or autograd.Function
     - positional inputs are Variables/Tensors or (nested) lists or tuples of
       them (similar requirement to NestedIOFunction)
     - outputs are similarly Variables/Tensors or (nested) lists or tuples of
       them
     - non-tensor typed values should be keyword arguments both in definition
       and when called

    Example usage:

    ```
    def symb(g, x, y):
        return g.op('Sum', x, y[0], y[1])

    @symbolic_override(symb)
    def foo(x, y):
        return x + y[0] + y[1]
    ```
    """

    return functools.partial(_symbolic_override_wrapper_maker, symbolic_fn, lambda args: False)


def symbolic_override_with_do_skip(symbolic_fn, do_skip):
    return functools.partial(_symbolic_override_wrapper_maker, symbolic_fn, do_skip)


def symbolic_override_first_arg_based(symbolic_fn):
    r"""
    Decorator to override ONNX export of the a function with specified subgraph.

    Equivalent to :func:`symbolic_override` but checks only the first argument
    of the function to figure out whether the tracing is on. Thus the first arg
    needs to be a Variable.
    """

    def do_skip(args):
        return not torch._C._jit_is_tracing(args[0])

    return symbolic_override_with_do_skip(symbolic_fn, do_skip)


def symbolic_override_packed_sequence_based(symbolic_fn):
    def do_skip(args):
        return not torch._C._jit_is_tracing(args[0][0])

    return symbolic_override_with_do_skip(symbolic_fn, do_skip)
