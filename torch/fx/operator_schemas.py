# mypy: allow-untyped-defs
import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload

if TYPE_CHECKING:
    from .node import Argument

__all__ = ["ArgsKwargsPair", "check_for_mutable_operation", "get_signature_for_torch_op", "create_type_hint",
           "type_matches", "normalize_function", "normalize_module"]

@compatibility(is_backward_compatible=False)
class ArgsKwargsPair(NamedTuple):
    """
    Simple named tuple for wrapping args/kwargs pairs.
    """
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

_manual_overrides : Dict[Callable, List[inspect.Signature]] = {}

def _nonzero_schemas():
    signatures = []

    def nonzero(self):
        pass
    signatures.append(inspect.signature(nonzero))

    def nonzero(self, *, as_tuple : bool):  # type: ignore[no-redef]
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
                      'Storage': torch.UntypedStorage,
                      't': typing.TypeVar('t')}
for k in dir(typing):
    _type_eval_globals[k] = getattr(typing, k)

def _torchscript_type_to_python_type(ts_type : 'torch._C.JitType') -> Any:
    """
    Convert a TorchScript type to a Python type (including subtypes) via
    eval'ing the annotation_str. _type_eval_globals sets up expressions
    like "List" and "Future" to map to actual types (typing.List and jit.Future)
    """
    return eval(ts_type.annotation_str, _type_eval_globals)

def _torchscript_schema_to_signature_impl(ts_schema : torch._C.FunctionSchema) -> inspect.Signature:
    from inspect import Parameter
    parameters : List[Parameter] = []
    for arg in ts_schema.arguments:
        arg_type = _torchscript_type_to_python_type(arg.type)
        default = arg.default_value if arg.has_default_value() else Parameter.empty
        # TODO: Figure out if this is safe. It seems like when generating the type signatures for
        # PythonArgParser, we emit signatures with `input` instead of `self` as the first tensor
        # argument name. Downstream, if someone converts that positional argument to a keyword
        # argument, the name mismatch will break things, so here we're going to normalize the
        # name to "input"
        name = arg.name if arg.name != 'self' else 'input'
        kind = Parameter.KEYWORD_ONLY if arg.kwarg_only else Parameter.POSITIONAL_OR_KEYWORD
        # "from" is a keyword therefore it must be a POSITIONAL_ONLY argument
        if name == "from":
            assert kind == Parameter.POSITIONAL_OR_KEYWORD
            # ParameterKind type is internal implementation detail to inspec package
            # which makes it hard to do type annotation
            kind = Parameter.POSITIONAL_ONLY  # type: ignore[assignment]
            # This renders all previous arguments to positional only
            for idx, p in enumerate(parameters):
                assert p.kind == Parameter.POSITIONAL_OR_KEYWORD
                parameters[idx] = Parameter(name=p.name, kind=Parameter.POSITIONAL_ONLY, default=p.default, annotation=p.annotation)
        parameters.append(Parameter(name=name, kind=kind, default=default, annotation=arg_type))
    return_types = [_torchscript_type_to_python_type(ret.type) for ret in ts_schema.returns]
    if len(return_types) == 0:
        return_type = None
    elif len(return_types) == 1:
        return_type = return_types[0]
    else:
        return_type = tuple(return_types)

    return inspect.Signature(parameters, return_annotation=return_type)

_SCHEMA_TO_SIGNATURE_CACHE : Dict[Tuple[str, str], inspect.Signature] = {}

def _torchscript_schema_to_signature(ts_schema : torch._C.FunctionSchema) -> inspect.Signature:
    # Cached as it's called in the hot path of FakeTensor dispatch
    cache_key = ts_schema.name, ts_schema.overload_name
    cache_val = _SCHEMA_TO_SIGNATURE_CACHE.get(cache_key)
    if cache_val is not None:
        return cache_val

    res = _torchscript_schema_to_signature_impl(ts_schema)
    _SCHEMA_TO_SIGNATURE_CACHE[cache_key] = res
    return res

@compatibility(is_backward_compatible=False)
def check_for_mutable_operation(target : Callable, args : Tuple['Argument', ...], kwargs : Dict[str, 'Argument']):
    signatures, schemas = get_signature_for_torch_op(target, return_schemas=True)

    if signatures and schemas:
        matched_schemas = []

        # Iterate through all of the schema until we find one that matches
        # If one matches, populate `new_args_and_kwargs` with the new args/kwargs
        # values. If none matches, `new_args_and_kwargs` will be None
        for candidate_signature, schema in zip(signatures, schemas):
            try:
                candidate_signature.bind(*args, **kwargs)
                matched_schemas.append((candidate_signature, schema))
            except TypeError as e:
                continue

        def throw_if_mutable(schema):
            if schema.is_mutable:
                raise RuntimeError(f'Tried to trace mutable operation {schema}. FX only supports functional '
                                   f'code, so operations that mutate operands in-place (e.g. via `out` arguments) '
                                   f'are not supported')

        if len(matched_schemas) == 0:
            # Did not match any schema. Cannot check for mutation
            pass
        elif len(matched_schemas) == 1:
            # Matched exactly one schema, unambiguous
            _, schema_to_check = matched_schemas[0]
            throw_if_mutable(schema_to_check)
            pass
        else:
            # Ambiguous schema match. Since mutability checking is best effort,
            # do nothing.
            pass

@compatibility(is_backward_compatible=False)
def get_signature_for_torch_op(op : Callable, return_schemas : bool = False):
    """
    Given an operator on the `torch` namespace, return a list of `inspect.Signature`
    objects corresponding to the overloads of that op.. May return `None` if a signature
    could not be retrieved.

    Args:
        op (Callable): An operator on the `torch` namespace to look up a signature for

    Returns:
        Optional[List[inspect.Signature]]: A list of signatures for the overloads of this
            operator, or None if the operator signatures could not be retrieved. If
            return_schemas=True, returns a tuple containing the optional Python signatures
            and the optional TorchScript Function signature
    """
    if isinstance(op, OpOverload):
        schemas = [op._schema]
    elif isinstance(op, OpOverloadPacket):
        schemas = [getattr(op, overload)._schema for overload in op.overloads()]
    else:
        override = _manual_overrides.get(op)
        if override:
            return (override, None) if return_schemas else None

        aten_fn = torch.jit._builtins._find_builtin(op)

        if aten_fn is None:
            return (None, None) if return_schemas else None
        schemas = torch._C._jit_get_schemas_for_operator(aten_fn)

    signatures = [_torchscript_schema_to_signature(schema) for schema in schemas]
    return (signatures, schemas) if return_schemas else signatures

@compatibility(is_backward_compatible=False)
def create_type_hint(x):
    try:
        if isinstance(x, (list, tuple)):
            # todo(chilli): Figure out the right way for mypy to handle this
            if isinstance(x, list):
                def ret_type(x):
                    return List[x]  # type: ignore[valid-type]
            else:
                def ret_type(x):
                    return Tuple[x, ...]
            if len(x) == 0:
                return ret_type(Any)
            base_type = x[0]
            for t in x:
                if issubclass(t, base_type):
                    continue
                elif issubclass(base_type, t):
                    base_type = t
                else:
                    return ret_type(Any)
            return ret_type(base_type)
    except Exception as e:
        # We tried to create a type hint for list but failed.
        warnings.warn(f"We were not able to successfully create type hint from the type {x}")
        pass
    return x

@compatibility(is_backward_compatible=False)
def type_matches(signature_type : Any, argument_type : Any):
    sig_origin_type = getattr(signature_type, '__origin__', signature_type)

    if signature_type is argument_type:
        return True

    # Union types in signature. Given type needs to match one of the
    # contained types in the Union
    if sig_origin_type is typing.Union and signature_type != argument_type:
        sig_contained = signature_type.__args__
        return any(type_matches(c, argument_type) for c in sig_contained)

    if signature_type is List[int] and argument_type is int:
        # int can be promoted to List[int]
        return True

    if getattr(signature_type, '__origin__', None) in {list, List}:
        sig_el_type = signature_type.__args__[0]
        if not inspect.isclass(sig_el_type):
            warnings.warn(
                f"Does not support nested parametric types, got {signature_type}. Please file a bug.")
            return False
        if getattr(argument_type, '__origin__', None) in {list, List}:
            return issubclass(argument_type.__args__[0], sig_el_type)

        def is_homogeneous_tuple(t):
            if getattr(t, "__origin__", None) not in {tuple, Tuple}:
                return False
            contained = t.__args__
            if t.__args__ == ((),):  # Tuple[()].__args__ == ((),) for some reason
                return True
            return all((c is Ellipsis) or issubclass(c, sig_el_type) for c in contained)

        # Tuple[T] is accepted for List[T] parameters
        return is_homogeneous_tuple(argument_type)

    # Dtype is an int in schemas
    if signature_type is int and argument_type is torch.dtype:
        return True

    if signature_type is numbers.Number and argument_type in {int, float}:
        return True
    if inspect.isclass(argument_type) and inspect.isclass(signature_type):
        return issubclass(argument_type, signature_type)

    return False

@compatibility(is_backward_compatible=False)
def normalize_function(
        target: Callable, args: Tuple[Any], kwargs : Optional[Dict[str, Any]] = None, arg_types : Optional[Tuple[Any]] = None,
        kwarg_types : Optional[Dict[str, Any]] = None,
        normalize_to_only_use_kwargs : bool = False) -> Optional[ArgsKwargsPair]:
    """
    Returns normalized arguments to PyTorch functions. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs). Does not support modules.

    May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

    Args:
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
        kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
    if kwargs is None:
        kwargs = {}
    new_args_and_kwargs = None
    if not isinstance(target, types.BuiltinFunctionType) and not (
        isinstance(target, (OpOverloadPacket, OpOverload))
    ):
        target_for_analysis = target
        if target in boolean_dispatched:
            # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
            # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
            # branches of the dispatch have exactly the same signature. If they do, use the `true`
            # branch signature for analysis. Otherwise, leave this un-normalized
            assert not isinstance(target, str)
            dispatched = boolean_dispatched[target]
            if_true, if_false = dispatched['if_true'], dispatched['if_false']
            if inspect.signature(if_true).parameters != inspect.signature(if_false).parameters:
                return None
            target_for_analysis = if_true

        assert callable(target_for_analysis)
        sig = inspect.signature(inspect.unwrap(target_for_analysis))
        new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(sig, args, kwargs, normalize_to_only_use_kwargs)
    else:
        assert callable(target)
        torch_op_schemas = get_signature_for_torch_op(target)
        matched_schemas = []
        if torch_op_schemas:
            # Iterate through all of the schema until we find one that matches
            # If one matches, populate `new_args_and_kwargs` with the new args/kwargs
            # values. If none matches, `new_args_and_kwargs` will be None
            for candidate_signature in torch_op_schemas:
                try:
                    candidate_signature.bind(*args, **kwargs)
                    matched_schemas.append(candidate_signature)
                except TypeError as e:
                    continue

            if len(matched_schemas) == 0:
                # Did not match any schema. Cannot normalize
                pass
            elif len(matched_schemas) == 1:
                # Matched exactly one schema, unambiguous
                new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(matched_schemas[0], args, kwargs,
                                                                             normalize_to_only_use_kwargs)
            else:
                if arg_types is not None or kwarg_types is not None:
                    arg_types = arg_types if arg_types else cast(Tuple[Any], ())
                    kwarg_types = kwarg_types if kwarg_types else {}
                    for candidate_signature in torch_op_schemas:
                        sig_matches = True
                        try:
                            bound_types = candidate_signature.bind(*arg_types, **kwarg_types)
                            for arg_name, arg_type in bound_types.arguments.items():
                                param = candidate_signature.parameters[arg_name]
                                sig_matches = sig_matches and type_matches(param.annotation, arg_type)
                        except TypeError as e:
                            sig_matches = False
                        if sig_matches:
                            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(candidate_signature, args, kwargs,
                                                                                         normalize_to_only_use_kwargs)
                            break
                else:
                    # Matched more than one schema. In this situation, the caller must provide the types of
                    # the arguments of the overload they expect.
                    schema_printouts = '\n'.join(str(schema) for schema in matched_schemas)
                    raise RuntimeError(f'Tried to normalize arguments to {torch.typename(target)} but '
                                       f'the schema match was ambiguous! Please provide argument types to '
                                       f'the normalize_arguments() call. Available schemas:\n{schema_printouts}')

    return new_args_and_kwargs

@compatibility(is_backward_compatible=False)
def normalize_module(
        root: torch.nn.Module, target: str, args: Tuple[Any], kwargs : Optional[Dict[str, Any]] = None,
        normalize_to_only_use_kwargs : bool = False) -> Optional[ArgsKwargsPair]:
    """
    Returns normalized arguments to PyTorch modules. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs).

    Args:
        root (nn.Module): root module upon which we query modules
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
    try:
        submod = root.get_submodule(target)
    except AttributeError as e:
        raise RuntimeError(f"Tried to normalize node with target {target} but root did not "
                           f"have that target!") from e
    if hasattr(submod.__class__, '__name__'):
        classname = submod.__class__.__name__
        if getattr(torch.nn, classname, None) == submod.__class__:
            sig = inspect.signature(inspect.unwrap(submod.forward))
            if kwargs is None:
                kwargs = {}
            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(sig, args, kwargs,
                                                                         normalize_to_only_use_kwargs)
            return new_args_and_kwargs
    return None

def _args_kwargs_to_normalized_args_kwargs(sig : inspect.Signature, args : Tuple[Any, ...],
                                           kwargs : Dict[str, Any],
                                           normalize_to_only_use_kwargs : bool) -> Optional[ArgsKwargsPair]:
    """
    Given a call target, args, and kwargs, return the arguments normalized into
    an ArgsKwargsPair, or None if the type signature is not supported by
    this normalization.

    Args:

        sig (inspect.Signature): Signature object for the target
        args (Tuple): Arguments that appear at the callsite for `target`
        kwargs (Dict): Keyword arguments that appear at the callsite for `target`
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Optional[ArgsKwargsPair]: Normalized args and kwargs for `target`, or `None` if
            this target is not supported.
    """

    # Don't currently support positional-only
    # or varargs (*args, **kwargs) signatures
    supported_parameter_types = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
        # Add an exception for one signature, which is common for random/uniform, i.e.:
        # Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None
        # `from` is Python keyword and as such functions with that signature should have
        # positional-only args, but at the same time they could be dispatched as kwargs
        if list(sig.parameters.keys()) != ['input', 'from', 'to', 'generator']:
            return None

    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    new_kwargs : Dict[str, Any] = {}
    new_args : List[Any] = []
    for i, param in enumerate(sig.parameters):
        if not normalize_to_only_use_kwargs and i < len(args):
            new_args.append(bound_args.arguments[param])
        else:
            new_kwargs[param] = bound_args.arguments[param]

    return ArgsKwargsPair(tuple(new_args), new_kwargs)
