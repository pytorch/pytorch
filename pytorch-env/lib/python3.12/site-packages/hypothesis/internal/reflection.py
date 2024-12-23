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
import linecache
import os
import re
import sys
import textwrap
import types
import warnings
from collections.abc import MutableMapping
from functools import partial, wraps
from io import StringIO
from keyword import iskeyword
from random import _inst as global_random_instance
from tokenize import COMMENT, detect_encoding, generate_tokens, untokenize
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable
from unittest.mock import _patch as PatchType
from weakref import WeakKeyDictionary

from hypothesis.errors import HypothesisWarning
from hypothesis.internal.compat import is_typed_named_tuple
from hypothesis.utils.conventions import not_set
from hypothesis.vendor.pretty import pretty

if TYPE_CHECKING:
    from hypothesis.strategies._internal.strategies import T

READTHEDOCS = os.environ.get("READTHEDOCS", None) == "True"
LAMBDA_SOURCE_CACHE: MutableMapping[Callable, str] = WeakKeyDictionary()


def is_mock(obj):
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


def function_digest(function):
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


def check_signature(sig: inspect.Signature) -> None:
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
) -> inspect.Signature:
    # Special case for use of `@unittest.mock.patch` decorator, mimicking the
    # behaviour of getfullargspec instead of reporting unusable arguments.
    patches = getattr(target, "patchings", None)
    if isinstance(patches, list) and all(isinstance(p, PatchType) for p in patches):
        P = inspect.Parameter
        return inspect.Signature(
            [P("args", P.VAR_POSITIONAL), P("keywargs", P.VAR_KEYWORD)]
        )

    if isinstance(getattr(target, "__signature__", None), inspect.Signature):
        # This special case covers unusual codegen like Pydantic models
        sig = target.__signature__
        check_signature(sig)
        # And *this* much more complicated block ignores the `self` argument
        # if that's been (incorrectly) included in the custom signature.
        if sig.parameters and (inspect.isclass(target) or inspect.ismethod(target)):
            selfy = next(iter(sig.parameters.values()))
            if (
                selfy.name == "self"
                and selfy.default is inspect.Parameter.empty
                and selfy.kind.name.startswith("POSITIONAL_")
            ):
                return sig.replace(
                    parameters=[v for k, v in sig.parameters.items() if k != "self"]
                )
        return sig
    # eval_str is only supported by Python 3.10 and newer
    if sys.version_info[:2] >= (3, 10):
        sig = inspect.signature(
            target, follow_wrapped=follow_wrapped, eval_str=eval_str
        )
    else:
        sig = inspect.signature(
            target, follow_wrapped=follow_wrapped
        )  # pragma: no cover
    check_signature(sig)
    return sig


def arg_is_required(param):
    return param.default is inspect.Parameter.empty and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def required_args(target, args=(), kwargs=()):
    """Return a set of names of required args to target that were not supplied
    in args or kwargs.

    This is used in builds() to determine which arguments to attempt to
    fill from type hints.  target may be any callable (including classes
    and bound methods).  args and kwargs should be as they are passed to
    builds() - that is, a tuple of values and a dict of names: values.
    """
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


def convert_keyword_arguments(function, args, kwargs):
    """Returns a pair of a tuple and a dictionary which would be equivalent
    passed as positional and keyword args to the function. Unless function has
    kwonlyargs or **kwargs the dictionary will always be empty.
    """
    sig = inspect.signature(function, follow_wrapped=False)
    bound = sig.bind(*args, **kwargs)
    return bound.args, bound.kwargs


def convert_positional_arguments(function, args, kwargs):
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


def ast_arguments_matches_signature(args, sig):
    assert isinstance(args, ast.arguments)
    assert isinstance(sig, inspect.Signature)
    expected = []
    for node in args.posonlyargs:
        expected.append((node.arg, inspect.Parameter.POSITIONAL_ONLY))
    for node in args.args:
        expected.append((node.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD))
    if args.vararg is not None:
        expected.append((args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for node in args.kwonlyargs:
        expected.append((node.arg, inspect.Parameter.KEYWORD_ONLY))
    if args.kwarg is not None:
        expected.append((args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return expected == [(p.name, p.kind) for p in sig.parameters.values()]


def is_first_param_referenced_in_function(f):
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


def extract_all_lambdas(tree, matching_signature):
    lambdas = []

    class Visitor(ast.NodeVisitor):
        def visit_Lambda(self, node):
            if ast_arguments_matches_signature(node.args, matching_signature):
                lambdas.append(node)

    Visitor().visit(tree)

    return lambdas


LINE_CONTINUATION = re.compile(r"\\\n")
WHITESPACE = re.compile(r"\s+")
PROBABLY_A_COMMENT = re.compile("""#[^'"]*$""")
SPACE_FOLLOWS_OPEN_BRACKET = re.compile(r"\( ")
SPACE_PRECEDES_CLOSE_BRACKET = re.compile(r" \)")


def _extract_lambda_source(f):
    """Extracts a single lambda expression from the string source. Returns a
    string indicating an unknown body if it gets confused in any way.

    This is not a good function and I am sorry for it. Forgive me my
    sins, oh lord
    """
    # You might be wondering how a lambda can have a return-type annotation?
    # The answer is that we add this at runtime, in new_given_signature(),
    # and we do support strange choices as applying @given() to a lambda.
    sig = inspect.signature(f)
    assert sig.return_annotation in (inspect.Parameter.empty, None), sig

    # Using pytest-xdist on Python 3.13, there's an entry in the linecache for
    # file "<string>", which then returns nonsense to getsource.  Discard it.
    linecache.cache.pop("<string>", None)

    if sig.parameters:
        if_confused = f"lambda {str(sig)[1:-1]}: <unknown>"
    else:
        if_confused = "lambda: <unknown>"
    try:
        source = inspect.getsource(f)
    except OSError:
        return if_confused

    source = LINE_CONTINUATION.sub(" ", source)
    source = WHITESPACE.sub(" ", source)
    source = source.strip()
    if "lambda" not in source and sys.platform == "emscripten":  # pragma: no cover
        return if_confused  # work around Pyodide bug in inspect.getsource()
    assert "lambda" in source, source

    tree = None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        for i in range(len(source) - 1, len("lambda"), -1):
            prefix = source[:i]
            if "lambda" not in prefix:
                break
            try:
                tree = ast.parse(prefix)
                source = prefix
                break
            except SyntaxError:
                continue
    if tree is None and source.startswith(("@", ".")):
        # This will always eventually find a valid expression because the
        # decorator or chained operator must be a valid Python function call,
        # so will eventually be syntactically valid and break out of the loop.
        # Thus, this loop can never terminate normally.
        for i in range(len(source) + 1):
            p = source[1:i]
            if "lambda" in p:
                try:
                    tree = ast.parse(p)
                    source = p
                    break
                except SyntaxError:
                    pass
        else:
            raise NotImplementedError("expected to be unreachable")

    if tree is None:
        return if_confused

    aligned_lambdas = extract_all_lambdas(tree, matching_signature=sig)
    if len(aligned_lambdas) != 1:
        return if_confused
    lambda_ast = aligned_lambdas[0]
    assert lambda_ast.lineno == 1

    # If the source code contains Unicode characters, the bytes of the original
    # file don't line up with the string indexes, and `col_offset` doesn't match
    # the string we're using.  We need to convert the source code into bytes
    # before slicing.
    #
    # Under the hood, the inspect module is using `tokenize.detect_encoding` to
    # detect the encoding of the original source file.  We'll use the same
    # approach to get the source code as bytes.
    #
    # See https://github.com/HypothesisWorks/hypothesis/issues/1700 for an
    # example of what happens if you don't correct for this.
    #
    # Note: if the code doesn't come from a file (but, for example, a doctest),
    # `getsourcefile` will return `None` and the `open()` call will fail with
    # an OSError.  Or if `f` is a built-in function, in which case we get a
    # TypeError.  In both cases, fall back to splitting the Unicode string.
    # It's not perfect, but it's the best we can do.
    try:
        with open(inspect.getsourcefile(f), "rb") as src_f:
            encoding, _ = detect_encoding(src_f.readline)

        source_bytes = source.encode(encoding)
        source_bytes = source_bytes[lambda_ast.col_offset :].strip()
        source = source_bytes.decode(encoding)
    except (OSError, TypeError):
        source = source[lambda_ast.col_offset :].strip()

    # This ValueError can be thrown in Python 3 if:
    #
    #  - There's a Unicode character in the line before the Lambda, and
    #  - For some reason we can't detect the source encoding of the file
    #
    # because slicing on `lambda_ast.col_offset` will account for bytes, but
    # the slice will be on Unicode characters.
    #
    # In practice this seems relatively rare, so we just give up rather than
    # trying to recover.
    try:
        source = source[source.index("lambda") :]
    except ValueError:
        return if_confused

    for i in range(len(source), len("lambda"), -1):  # pragma: no branch
        try:
            parsed = ast.parse(source[:i])
            assert len(parsed.body) == 1
            assert parsed.body
            if isinstance(parsed.body[0].value, ast.Lambda):
                source = source[:i]
                break
        except SyntaxError:
            pass
    lines = source.split("\n")
    lines = [PROBABLY_A_COMMENT.sub("", l) for l in lines]
    source = "\n".join(lines)

    source = WHITESPACE.sub(" ", source)
    source = SPACE_FOLLOWS_OPEN_BRACKET.sub("(", source)
    source = SPACE_PRECEDES_CLOSE_BRACKET.sub(")", source)
    return source.strip()


def extract_lambda_source(f):
    try:
        return LAMBDA_SOURCE_CACHE[f]
    except KeyError:
        pass

    source = _extract_lambda_source(f)
    LAMBDA_SOURCE_CACHE[f] = source
    return source


def get_pretty_function_description(f):
    if isinstance(f, partial):
        return pretty(f)
    if not hasattr(f, "__name__"):
        return repr(f)
    name = f.__name__
    if name == "<lambda>":
        return extract_lambda_source(f)
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


def nicerepr(v):
    if inspect.isfunction(v):
        return get_pretty_function_description(v)
    elif isinstance(v, type):
        return v.__name__
    else:
        # With TypeVar T, show List[T] instead of TypeError on List[~T]
        return re.sub(r"(\[)~([A-Z][a-z]*\])", r"\g<1>\g<2>", pretty(v))


def repr_call(f, args, kwargs, *, reorder=True):
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


def check_valid_identifier(identifier):
    if not identifier.isidentifier():
        raise ValueError(f"{identifier!r} is not a valid python identifier")


eval_cache: dict = {}


def source_exec_as_module(source):
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


def get_varargs(sig, kind=inspect.Parameter.VAR_POSITIONAL):
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
        varkw = get_varargs(signature, kind=inspect.Parameter.VAR_KEYWORD)
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
        return f

    return accept


def proxies(target: "T") -> Callable[[Callable], "T"]:
    replace_sig = define_function_signature(
        target.__name__.replace("<lambda>", "_lambda_"),  # type: ignore
        target.__doc__,
        get_signature(target, follow_wrapped=False),
    )

    def accept(proxy):
        return impersonate(target)(wraps(target)(replace_sig(proxy)))

    return accept


def is_identity_function(f):
    # TODO: pattern-match the AST to handle `def ...` identity functions too
    return bool(re.fullmatch(r"lambda (\w+): \1", get_pretty_function_description(f)))
