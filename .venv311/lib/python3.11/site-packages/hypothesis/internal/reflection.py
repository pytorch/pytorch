# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""This file can approximately be considered the collection of hypothesis going
to really unreasonable lengths to produce pretty output."""

import ast
import hashlib
import inspect
import re
import textwrap
import types
import warnings
from collections.abc import Callable, Sequence
from functools import partial, wraps
from inspect import Parameter, Signature
from io import StringIO
from keyword import iskeyword
from random import _inst as global_random_instance
from tokenize import COMMENT, generate_tokens, untokenize
from types import EllipsisType, ModuleType
from typing import TYPE_CHECKING, Any, TypeVar, Union
from unittest.mock import _patch as PatchType

from hypothesis.errors import HypothesisWarning
from hypothesis.internal import lambda_sources
from hypothesis.internal.compat import is_typed_named_tuple
from hypothesis.utils.conventions import not_set
from hypothesis.vendor.pretty import pretty

if TYPE_CHECKING:
    from hypothesis.strategies._internal.strategies import SearchStrategy

T = TypeVar("T")


def is_mock(obj: object) -> bool:
    """Determine if the given argument is a mock type."""

    # We want to be able to detect these when dealing with various test
    # args. As they are sneaky and can look like almost anything else,
    # we'll check this by looking for an attribute with a name that it's really
    # unlikely to implement accidentally, and that anyone who implements it
    # deliberately should know what they're doing. This is more robust than
    # looking for types.
    return hasattr(obj, "hypothesis_internal_is_this_a_mock_check")


def _clean_source(src: str) -> bytes:
    """Return the source code as bytes, without decorators or comments.

    Because this is part of our database key, we reduce the cache invalidation
    rate by ignoring decorators, comments, trailing whitespace, and empty lines.
    We can't just use the (dumped) AST directly because it changes between Python
    versions (e.g. ast.Constant)
    """
    # Get the (one-indexed) line number of the function definition, and drop preceding
    # lines - i.e. any decorators, so that adding `@example()`s keeps the same key.
    try:
        funcdef = ast.parse(src).body[0]
        src = "".join(src.splitlines(keepends=True)[funcdef.lineno - 1 :])
    except Exception:
        pass
    # Remove blank lines and use the tokenize module to strip out comments,
    # so that those can be changed without changing the database key.
    try:
        src = untokenize(
            t for t in generate_tokens(StringIO(src).readline) if t.type != COMMENT
        )
    except Exception:
        pass
    # Finally, remove any trailing whitespace and empty lines as a last cleanup.
    return "\n".join(x.rstrip() for x in src.splitlines() if x.rstrip()).encode()


def function_digest(function: Any) -> bytes:
    """Returns a string that is stable across multiple invocations across
    multiple processes and is prone to changing significantly in response to
    minor changes to the function.

    No guarantee of uniqueness though it usually will be. Digest collisions
    lead to unfortunate but not fatal problems during database replay.
    """
    hasher = hashlib.sha384()
    try:
        src = inspect.getsource(function)
    except (OSError, TypeError):
        # If we can't actually get the source code, try for the name as a fallback.
        # NOTE: We might want to change this to always adding function.__qualname__,
        # to differentiate f.x. two classes having the same function implementation
        # with class-dependent behaviour.
        try:
            hasher.update(function.__name__.encode())
        except AttributeError:
            pass
    else:
        hasher.update(_clean_source(src))
    try:
        # This is additional to the source code because it can include the effects
        # of decorators, or of post-hoc assignment to the .__signature__ attribute.
        hasher.update(repr(get_signature(function)).encode())
    except Exception:
        pass
    try:
        # We set this in order to distinguish e.g. @pytest.mark.parametrize cases.
        hasher.update(function._hypothesis_internal_add_digest)
    except AttributeError:
        pass
    return hasher.digest()


def check_signature(sig: Signature) -> None:
    # Backport from Python 3.11; see https://github.com/python/cpython/pull/92065
    for p in sig.parameters.values():
        if iskeyword(p.name) and p.kind is not p.POSITIONAL_ONLY:
            raise ValueError(
                f"Signature {sig!r} contains a parameter named {p.name!r}, "
                f"but this is a SyntaxError because `{p.name}` is a keyword. "
                "You, or a library you use, must have manually created an "
                "invalid signature - this will be an error in Python 3.11+"
            )


def get_signature(
    target: Any, *, follow_wrapped: bool = True, eval_str: bool = False
) -> Signature:
    # Special case for use of `@unittest.mock.patch` decorator, mimicking the
    # behaviour of getfullargspec instead of reporting unusable arguments.
    patches = getattr(target, "patchings", None)
    if isinstance(patches, list) and all(isinstance(p, PatchType) for p in patches):
        return Signature(
            [
                Parameter("args", Parameter.VAR_POSITIONAL),
                Parameter("keywargs", Parameter.VAR_KEYWORD),
            ]
        )

    if isinstance(getattr(target, "__signature__", None), Signature):
        # This special case covers unusual codegen like Pydantic models
        sig = target.__signature__
        check_signature(sig)
        # And *this* much more complicated block ignores the `self` argument
        # if that's been (incorrectly) included in the custom signature.
        if sig.parameters and (inspect.isclass(target) or inspect.ismethod(target)):
            selfy = next(iter(sig.parameters.values()))
            if (
                selfy.name == "self"
                and selfy.default is Parameter.empty
                and selfy.kind.name.startswith("POSITIONAL_")
            ):
                return sig.replace(
                    parameters=[v for k, v in sig.parameters.items() if k != "self"]
                )
        return sig
    sig = inspect.signature(target, follow_wrapped=follow_wrapped, eval_str=eval_str)
    check_signature(sig)
    return sig


def arg_is_required(param: Parameter) -> bool:
    return param.default is Parameter.empty and param.kind in (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    )


def required_args(
    target: Callable[..., Any],
    args: tuple["SearchStrategy[Any]", ...] = (),
    kwargs: dict[str, Union["SearchStrategy[Any]", EllipsisType]] | None = None,
) -> set[str]:
    """Return a set of names of required args to target that were not supplied
    in args or kwargs.

    This is used in builds() to determine which arguments to attempt to
    fill from type hints.  target may be any callable (including classes
    and bound methods).  args and kwargs should be as they are passed to
    builds() - that is, a tuple of values and a dict of names: values.
    """
    kwargs = {} if kwargs is None else kwargs
    # We start with a workaround for NamedTuples, which don't have nice inits
    if inspect.isclass(target) and is_typed_named_tuple(target):
        provided = set(kwargs) | set(target._fields[: len(args)])
        return set(target._fields) - provided
    # Then we try to do the right thing with inspect.signature
    try:
        sig = get_signature(target)
    except (ValueError, TypeError):
        return set()
    return {
        name
        for name, param in list(sig.parameters.items())[len(args) :]
        if arg_is_required(param) and name not in kwargs
    }


def convert_keyword_arguments(
    function: Any, args: Sequence[object], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Returns a pair of a tuple and a dictionary which would be equivalent
    passed as positional and keyword args to the function. Unless function has
    kwonlyargs or **kwargs the dictionary will always be empty.
    """
    sig = inspect.signature(function, follow_wrapped=False)
    bound = sig.bind(*args, **kwargs)
    return bound.args, bound.kwargs


def convert_positional_arguments(
    function: Any, args: Sequence[object], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Return a tuple (new_args, new_kwargs) where all possible arguments have
    been moved to kwargs.

    new_args will only be non-empty if function has pos-only args or *args.
    """
    sig = inspect.signature(function, follow_wrapped=False)
    bound = sig.bind(*args, **kwargs)
    new_args = []
    new_kwargs = dict(bound.arguments)
    for p in sig.parameters.values():
        if p.name in new_kwargs:
            if p.kind is p.POSITIONAL_ONLY:
                new_args.append(new_kwargs.pop(p.name))
            elif p.kind is p.VAR_POSITIONAL:
                new_args.extend(new_kwargs.pop(p.name))
            elif p.kind is p.VAR_KEYWORD:
                assert set(new_kwargs[p.name]).isdisjoint(set(new_kwargs) - {p.name})
                new_kwargs.update(new_kwargs.pop(p.name))
    return tuple(new_args), new_kwargs


def ast_arguments_matches_signature(args: ast.arguments, sig: Signature) -> bool:
    expected: list[tuple[str, int]] = []
    for node in args.posonlyargs:
        expected.append((node.arg, Parameter.POSITIONAL_ONLY))
    for node in args.args:
        expected.append((node.arg, Parameter.POSITIONAL_OR_KEYWORD))
    if args.vararg is not None:
        expected.append((args.vararg.arg, Parameter.VAR_POSITIONAL))
    for node in args.kwonlyargs:
        expected.append((node.arg, Parameter.KEYWORD_ONLY))
    if args.kwarg is not None:
        expected.append((args.kwarg.arg, Parameter.VAR_KEYWORD))
    return expected == [(p.name, p.kind) for p in sig.parameters.values()]


def is_first_param_referenced_in_function(f: Any) -> bool:
    """Is the given name referenced within f?"""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    except Exception:
        return True  # Assume it's OK unless we know otherwise
    name = next(iter(get_signature(f).parameters))
    return any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )


def get_pretty_function_description(f: object) -> str:
    if isinstance(f, partial):
        return pretty(f)
    if not hasattr(f, "__name__"):
        return repr(f)
    name = f.__name__  # type: ignore
    if name == "<lambda>":
        return lambda_sources.lambda_description(f)
    elif isinstance(f, (types.MethodType, types.BuiltinMethodType)):
        self = f.__self__
        # Some objects, like `builtins.abs` are of BuiltinMethodType but have
        # their module as __self__.  This might include c-extensions generally?
        if not (self is None or inspect.isclass(self) or inspect.ismodule(self)):
            if self is global_random_instance:
                return f"random.{name}"
            return f"{self!r}.{name}"
    elif isinstance(name, str) and getattr(dict, name, object()) is f:
        # special case for keys/values views in from_type() / ghostwriter output
        return f"dict.{name}"
    return name


def nicerepr(v: Any) -> str:
    if inspect.isfunction(v):
        return get_pretty_function_description(v)
    elif isinstance(v, type):
        return v.__name__
    else:
        # With TypeVar T, show List[T] instead of TypeError on List[~T]
        return re.sub(r"(\[)~([A-Z][a-z]*\])", r"\g<1>\g<2>", pretty(v))


def repr_call(
    f: Any, args: Sequence[object], kwargs: dict[str, object], *, reorder: bool = True
) -> str:
    # Note: for multi-line pretty-printing, see RepresentationPrinter.repr_call()
    if reorder:
        args, kwargs = convert_positional_arguments(f, args, kwargs)

    bits = [nicerepr(x) for x in args]

    for p in get_signature(f).parameters.values():
        if p.name in kwargs and not p.kind.name.startswith("VAR_"):
            bits.append(f"{p.name}={nicerepr(kwargs.pop(p.name))}")
    if kwargs:
        for a in sorted(kwargs):
            bits.append(f"{a}={nicerepr(kwargs[a])}")

    rep = nicerepr(f)
    if rep.startswith("lambda") and ":" in rep:
        rep = f"({rep})"
    repr_len = len(rep) + sum(len(b) for b in bits)  # approx
    if repr_len > 30000:
        warnings.warn(
            "Generating overly large repr. This is an expensive operation, and with "
            f"a length of {repr_len//1000} kB is unlikely to be useful. Use -Wignore "
            "to ignore the warning, or -Werror to get a traceback.",
            HypothesisWarning,
            stacklevel=2,
        )
    return rep + "(" + ", ".join(bits) + ")"


def check_valid_identifier(identifier: str) -> None:
    if not identifier.isidentifier():
        raise ValueError(f"{identifier!r} is not a valid python identifier")


eval_cache: dict[str, ModuleType] = {}


def source_exec_as_module(source: str) -> ModuleType:
    try:
        return eval_cache[source]
    except KeyError:
        pass

    hexdigest = hashlib.sha384(source.encode()).hexdigest()
    result = ModuleType("hypothesis_temporary_module_" + hexdigest)
    assert isinstance(source, str)
    exec(source, result.__dict__)
    eval_cache[source] = result
    return result


COPY_SIGNATURE_SCRIPT = """
from hypothesis.utils.conventions import not_set

def accept({funcname}):
    def {name}{signature}:
        return {funcname}({invocation})
    return {name}
""".lstrip()


def get_varargs(
    sig: Signature, kind: int = Parameter.VAR_POSITIONAL
) -> Parameter | None:
    for p in sig.parameters.values():
        if p.kind is kind:
            return p
    return None


def define_function_signature(name, docstring, signature):
    """A decorator which sets the name, signature and docstring of the function
    passed into it."""
    if name == "<lambda>":
        name = "_lambda_"
    check_valid_identifier(name)
    for a in signature.parameters:
        check_valid_identifier(a)

    used_names = {*signature.parameters, name}

    newsig = signature.replace(
        parameters=[
            p if p.default is signature.empty else p.replace(default=not_set)
            for p in (
                p.replace(annotation=signature.empty)
                for p in signature.parameters.values()
            )
        ],
        return_annotation=signature.empty,
    )

    pos_args = [
        p
        for p in signature.parameters.values()
        if p.kind.name.startswith("POSITIONAL_")
    ]

    def accept(f):
        fsig = inspect.signature(f, follow_wrapped=False)
        must_pass_as_kwargs = []
        invocation_parts = []
        for p in pos_args:
            if p.name not in fsig.parameters and get_varargs(fsig) is None:
                must_pass_as_kwargs.append(p.name)
            else:
                invocation_parts.append(p.name)
        if get_varargs(signature) is not None:
            invocation_parts.append("*" + get_varargs(signature).name)
        for k in must_pass_as_kwargs:
            invocation_parts.append(f"{k}={k}")
        for p in signature.parameters.values():
            if p.kind is p.KEYWORD_ONLY:
                invocation_parts.append(f"{p.name}={p.name}")
        varkw = get_varargs(signature, kind=Parameter.VAR_KEYWORD)
        if varkw:
            invocation_parts.append("**" + varkw.name)

        candidate_names = ["f"] + [f"f_{i}" for i in range(1, len(used_names) + 2)]

        for funcname in candidate_names:  # pragma: no branch
            if funcname not in used_names:
                break

        source = COPY_SIGNATURE_SCRIPT.format(
            name=name,
            funcname=funcname,
            signature=str(newsig),
            invocation=", ".join(invocation_parts),
        )
        result = source_exec_as_module(source).accept(f)
        result.__doc__ = docstring
        result.__defaults__ = tuple(
            p.default
            for p in signature.parameters.values()
            if p.default is not signature.empty and "POSITIONAL" in p.kind.name
        )
        kwdefaults = {
            p.name: p.default
            for p in signature.parameters.values()
            if p.default is not signature.empty and p.kind is p.KEYWORD_ONLY
        }
        if kwdefaults:
            result.__kwdefaults__ = kwdefaults
        annotations = {
            p.name: p.annotation
            for p in signature.parameters.values()
            if p.annotation is not signature.empty
        }
        if signature.return_annotation is not signature.empty:
            annotations["return"] = signature.return_annotation
        if annotations:
            result.__annotations__ = annotations
        return result

    return accept


def impersonate(target):
    """Decorator to update the attributes of a function so that to external
    introspectors it will appear to be the target function.

    Note that this updates the function in place, it doesn't return a
    new one.
    """

    def accept(f):
        # Lie shamelessly about where this code comes from, to hide the hypothesis
        # internals from pytest, ipython, and other runtime introspection.
        f.__code__ = f.__code__.replace(
            co_filename=target.__code__.co_filename,
            co_firstlineno=target.__code__.co_firstlineno,
        )
        f.__name__ = target.__name__
        f.__module__ = target.__module__
        f.__doc__ = target.__doc__
        f.__globals__["__hypothesistracebackhide__"] = True
        # But leave an breadcrumb for _describe_lambda to follow, it's
        # just confused by the lies above
        f.__wrapped_target = target
        return f

    return accept


def proxies(target: T) -> Callable[[Callable], T]:
    replace_sig = define_function_signature(
        target.__name__.replace("<lambda>", "_lambda_"),  # type: ignore
        target.__doc__,
        get_signature(target, follow_wrapped=False),
    )

    def accept(proxy):
        return impersonate(target)(wraps(target)(replace_sig(proxy)))

    return accept


def is_identity_function(f: Callable) -> bool:
    try:
        code = f.__code__
    except AttributeError:
        try:
            f = f.__call__  # type: ignore
            code = f.__code__
        except AttributeError:
            return False

    # We only accept a single unbound argument. While it would be possible to
    # accept extra defaulted arguments, it would be pointless as they couldn't
    # be referenced at all in the code object (or the co_code check would fail).
    bound_args = int(inspect.ismethod(f))
    if code.co_argcount != bound_args + 1 or code.co_kwonlyargcount > 0:
        return False

    # We know that f accepts a single positional argument, now check that its
    # code object is simply "return first unbound argument".
    template = (lambda self, x: x) if bound_args else (lambda x: x)  # type: ignore
    try:
        return code.co_code == template.__code__.co_code
    except AttributeError:  # pragma: no cover  # pypy only
        # In PyPy, some builtin functions have a code object ('builtin-code')
        # lacking co_code, perhaps because they are native-compiled and don't have
        # a corresponding bytecode. Regardless, since Python doesn't have any
        # builtin identity function it seems safe to say that this one isn't
        return False
