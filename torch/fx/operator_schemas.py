import torch
import inspect
import numbers
import typing
import enum
from typing import Any, Callable, Dict, List, Optional

_manual_overrides : Dict[Callable, Callable[[], inspect.Signature]] = {}

def nonzero_schemas():
    signatures = []

    def nonzero(self):
        pass
    signatures.append(inspect.signature(nonzero))

    def nonzero(self, *, as_tuple : bool):
        pass
    signatures.append(inspect.signature(nonzero))

    return signatures

_manual_overrides[torch.nonzero] = nonzero_schemas()

class _FakeGlobalNamespace:
    def __getattr__(self, name):
        if name == 'torch':
            return torch
        raise RuntimeError('Expected a torch namespace lookup lookup')

def _torchscript_type_to_python_type(ts_type : torch._C.Type) -> Any:
    g = {'Tensor' : torch.Tensor, 'Device' : torch.device, 'Layout' : torch.layout,
         'number' : numbers.Number, 'Future' : torch.jit.Future,
         'AnyEnumType' : enum.Enum, 'QScheme' : torch.qscheme,
         '__torch__': _FakeGlobalNamespace(), 't': typing.TypeVar('t')}
    for k in dir(typing):
        g[k] = getattr(typing, k)

    return eval(ts_type.annotation_str, g)

def _torchscript_schema_to_signature(ts_schema : torch._C.FunctionSchema) -> inspect.Signature:
    parameters : List[inspect.Parameter] = []
    for arg in ts_schema.arguments:
        arg_type = _torchscript_type_to_python_type(arg.type)
        default = arg.default_value if arg.has_default_value() else inspect.Parameter.empty
        # TODO: Figure out if this is safe. It seems like when generating the type signatures for
        # PythonArgParser, we emit signatures with `input` instead of `self` as the first tensor
        # argument name. Downstream, if someone converts that positional argument to a keyword
        # argument, the name mismatch will break things, so here we're going to normalize the
        # name to "input"
        name = arg.name if arg.name != 'self' else 'input'
        kind = inspect.Parameter.KEYWORD_ONLY if arg.kwarg_only else inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters.append(inspect.Parameter(name=name, kind=kind, default=default, annotation=arg_type))
    return_types = [_torchscript_type_to_python_type(ret.type) for ret in ts_schema.returns]
    if len(return_types) == 0:
        return_type = None
    elif len(return_types) == 1:
        return_type = return_types[0]
    else:
        return_type = tuple(return_types)

    return inspect.Signature(parameters, return_annotation=return_type)

def get_signature_for_torch_op(op : Callable) -> Optional[List[inspect.Signature]]:
    override = _manual_overrides.get(op)
    if override:
        return override

    aten_fn = torch.jit._builtins._find_builtin(op)

    if aten_fn is None:
        return None

    schemas = torch._C._jit_get_schemas_for_operator(aten_fn)
    signatures = [_torchscript_schema_to_signature(schema) for schema in schemas]

    return signatures
