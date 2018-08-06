import functools
import types

import torch._C as _C

TensorProtoDataType = _C._onnx.TensorProtoDataType
OperatorExportTypes = _C._onnx.OperatorExportTypes

ONNX_ARCHIVE_MODEL_PROTO_NAME = "__MODEL_PROTO"


class ExportTypes:
    PROTOBUF_FILE = 1
    ZIP_ARCHIVE = 2
    COMPRESSED_ZIP_ARCHIVE = 3
    DIRECTORY = 4


def _export(*args, **kwargs):
    from torch.onnx import utils
    return utils._export(*args, **kwargs)


def export(*args, **kwargs):
    from torch.onnx import utils
    return utils.export(*args, **kwargs)


def export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils.export_to_pretty_string(*args, **kwargs)


def _export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils._export_to_pretty_string(*args, **kwargs)


def _optimize_trace(trace, operator_export_type):
    from torch.onnx import utils
    trace.set_graph(utils._optimize_graph(trace.graph(), operator_export_type))


def set_training(*args, **kwargs):
    from torch.onnx import utils
    return utils.set_training(*args, **kwargs)


def _run_symbolic_function(*args, **kwargs):
    from torch.onnx import utils
    return utils._run_symbolic_function(*args, **kwargs)


def _run_symbolic_method(*args, **kwargs):
    from torch.onnx import utils
    return utils._run_symbolic_method(*args, **kwargs)


def symbolic_override(symbolic_fn):
    r"""
    Decorator to override ONNX export of the a function with specified subgraph.

    Effectively allows to attach symbolic() implementation to an arbitrary
    python function or autograd.Function. Requirements for the decorated
    function:
     - being non-member function or autograd.Function
     - positional inputs are Tensors or (nested) lists or tuples of
       them (similar requirement to NestedIOFunction)
     - outputs are similarly Tensors or (nested) lists or tuples of them
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
    def decorator(fn):
        import torch
        from torch.autograd import function

        def wrapper(*args, **kwargs):
            tstate = torch._C._get_tracing_state()
            if not tstate:
                return fn(*args, **kwargs)

            flat_args = tuple(function._iter_tensors_permissive(args))
            arg_values = [torch._C._get_value_trace(x) if isinstance(x, torch.Tensor) else x for x in flat_args]

            # This must come after the calls to get_value_trace, lest we
            # lose information due to in-place operations.
            output_vars = fn(*args, **kwargs)

            symbolic_args = function._unflatten(arg_values, args)
            output_vals = symbolic_fn(tstate.graph(), *symbolic_args, **kwargs)

            for var, val in zip(
                    function._iter_tensors(output_vars),
                    function._iter_jit_values(output_vals)):
                val.inferTypeFrom(var.data)
                torch._C._set_value_trace(var, val)

            return output_vars

        # fn might be autograd.Function too, in this case wrapping doesn't work
        if isinstance(fn, types.FunctionType):
            wrapper = functools.wraps(fn)(wrapper)

        return wrapper
    return decorator
