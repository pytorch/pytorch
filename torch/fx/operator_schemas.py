import torch
import inspect
import numbers
import typing
import enum
from typing import Any, Callable, Dict, List, Optional

_manual_overrides : Dict[Callable, List[inspect.Signature]] = {}

def _nonzero_schemas():
    signatures = []

    def nonzero(self):
        pass
    signatures.append(inspect.signature(nonzero))

    def nonzero(self, *, as_tuple : bool):  # type: ignore
        pass
    signatures.append(inspect.signature(nonzero))

    return signatures

_manual_overrides[torch.nonzero] = _nonzero_schemas()

class _FakeGlobalNamespace:
    def __getattr__(self, name):
        if name == 'torch':
            return torch
        raise RuntimeError('Expected a torch namespace lookup')

_type_eval_globals = {'Tensor' : torch.Tensor, 'Device' : torch.device, 'Layout' : torch.layout,
                      'number' : numbers.Number, 'Future' : torch.jit.Future,
                      'AnyEnumType' : enum.Enum, 'QScheme' : torch.qscheme,
                      '__torch__': _FakeGlobalNamespace(), 'NoneType': type(None),
                      't': typing.TypeVar('t')}  # type: ignore
for k in dir(typing):
    _type_eval_globals[k] = getattr(typing, k)

def _torchscript_type_to_python_type(ts_type : 'torch._C.JitType') -> Any:
    """
    Convert a TorchScript type to a Python type (including subtypes) via
    eval'ing the annotation_str. _type_eval_globals sets up expressions
    like "List" and "Future" to map to actual types (typing.List and jit.Future)
    """
    return eval(ts_type.annotation_str, _type_eval_globals)

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
    """
    Given an operator on the `torch` namespace, return a list of `inspect.Signature`
    objects corresponding to the overloads of that op.. May return `None` if a signature
    could not be retrieved.

    Args:
        op (Callable): An operator on the `torch` namespace to look up a signature for

    Returns:
        Optional[List[inspect.Signature]]: A list of signatures for the overloads of this
            operator, or None if the operator signatures could not be retrieved.
    """
    override = _manual_overrides.get(op)
    if override:
        return override

    aten_fn = torch.jit._builtins._find_builtin(op)

    if aten_fn is None:
        return None

    schemas = torch._C._jit_get_schemas_for_operator(aten_fn)
    signatures = [_torchscript_schema_to_signature(schema) for schema in schemas]

    return signatures


def type_matches(signature_type : Any, argument_type : Any):
    sig_origin_type = getattr(signature_type, '__origin__', signature_type)
    argument_origin_type = getattr(argument_type, '__origin__', argument_type)

    # Union types in signature. Given type needs to match one of the
    # contained types in the Union
    if sig_origin_type is typing.Union:
        contained = signature_type.__args__
        return any(type_matches(c, argument_type) for c in contained)

    if type(signature_type) is typing._GenericAlias and sig_origin_type is list:
        contained = signature_type.__args__
        assert len(contained) == 1

        if contained[0] == int:
            # int can be promoted to List[int]
            if argument_type is int:
                return True

            # Tuple[int] is accepted for List[int] parameters
            if type(argument_type) is typing._GenericAlias and argument_origin_type is tuple:
                argument_contained_types = argument_type.__args__
                return all(a is int for a in argument_contained_types)

    # Dtype is an int in schemas
    if signature_type is int and argument_type is torch.dtype:
        return True

    if signature_type is numbers.Number and argument_type in {int, float}:
        return True

    return signature_type is argument_type
