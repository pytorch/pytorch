# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Writing tests with Hypothesis frees you from the tedium of deciding on and
writing out specific inputs to test.  Now, the ``hypothesis.extra.ghostwriter``
module can write your test functions for you too!

The idea is to provide **an easy way to start** property-based testing,
**and a seamless transition** to more complex test code - because ghostwritten
tests are source code that you could have written for yourself.

So just pick a function you'd like tested, and feed it to one of the functions
below.  They follow imports, use but do not require type annotations, and
generally do their best to write you a useful test.  You can also use
:ref:`our command-line interface <hypothesis-cli>`::

    $ hypothesis write --help
    Usage: hypothesis write [OPTIONS] FUNC...

      `hypothesis write` writes property-based tests for you!

      Type annotations are helpful but not required for our advanced
      introspection and templating logic.  Try running the examples below to see
      how it works:

          hypothesis write gzip
          hypothesis write numpy.matmul
          hypothesis write pandas.from_dummies
          hypothesis write re.compile --except re.error
          hypothesis write --equivalent ast.literal_eval eval
          hypothesis write --roundtrip json.dumps json.loads
          hypothesis write --style=unittest --idempotent sorted
          hypothesis write --binary-op operator.add

    Options:
      --roundtrip                 start by testing write/read or encode/decode!
      --equivalent                very useful when optimising or refactoring code
      --errors-equivalent         --equivalent, but also allows consistent errors
      --idempotent                check that f(x) == f(f(x))
      --binary-op                 associativity, commutativity, identity element
      --style [pytest|unittest]   pytest-style function, or unittest-style method?
      -e, --except OBJ_NAME       dotted name of exception(s) to ignore
      --annotate / --no-annotate  force ghostwritten tests to be type-annotated
                                  (or not).  By default, match the code to test.
      -h, --help                  Show this message and exit.

.. tip::

    Using a light theme?  Hypothesis respects `NO_COLOR <https://no-color.org/>`__
    and ``DJANGO_COLORS=light``.

.. note::

    The ghostwriter requires :pypi:`black`, but the generated code only
    requires Hypothesis itself.

.. note::

    Legal questions?  While the ghostwriter fragments and logic is under the
    MPL-2.0 license like the rest of Hypothesis, the *output* from the ghostwriter
    is made available under the `Creative Commons Zero (CC0)
    <https://creativecommons.org/share-your-work/public-domain/cc0/>`__
    public domain dedication, so you can use it without any restrictions.
"""

import ast
import builtins
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping
from itertools import permutations, zip_longest
from keyword import iskeyword as _iskeyword
from string import ascii_lowercase
from textwrap import dedent, indent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    ForwardRef,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import black

from hypothesis import Verbosity, find, settings, strategies as st
from hypothesis.errors import InvalidArgument, SmallSearchSpaceWarning
from hypothesis.internal.compat import get_type_hints
from hypothesis.internal.reflection import get_signature, is_mock
from hypothesis.internal.validation import check_type
from hypothesis.provisional import domains
from hypothesis.strategies._internal.collections import ListStrategy
from hypothesis.strategies._internal.core import BuildsStrategy
from hypothesis.strategies._internal.deferred import DeferredStrategy
from hypothesis.strategies._internal.flatmapped import FlatMapStrategy
from hypothesis.strategies._internal.lazy import LazyStrategy, unwrap_strategies
from hypothesis.strategies._internal.strategies import (
    FilteredStrategy,
    MappedStrategy,
    OneOfStrategy,
    SampledFromStrategy,
)
from hypothesis.strategies._internal.types import _global_type_lookup, is_generic_type

if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)


IMPORT_SECTION = """
# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

{imports}
"""

TEMPLATE = """
@given({given_args})
def test_{test_kind}_{func_name}({arg_names}){return_annotation}:
{test_body}
"""

SUPPRESS_BLOCK = """
try:
{test_body}
except {exceptions}:
    reject()
""".strip()

Except = Union[type[Exception], tuple[type[Exception], ...]]
ImportSet = set[Union[str, tuple[str, str]]]
_quietly_settings = settings(
    database=None,
    deadline=None,
    derandomize=True,
    verbosity=Verbosity.quiet,
)


def _dedupe_exceptions(exc: tuple[type[Exception], ...]) -> tuple[type[Exception], ...]:
    # This is reminiscent of de-duplication logic I wrote for flake8-bugbear,
    # but with access to the actual objects we can just check for subclasses.
    # This lets us print e.g. `Exception` instead of `(Exception, OSError)`.
    uniques = list(exc)
    for a, b in permutations(exc, 2):
        if a in uniques and issubclass(a, b):
            uniques.remove(a)
    return tuple(sorted(uniques, key=lambda e: e.__name__))


def _check_except(except_: Except) -> tuple[type[Exception], ...]:
    if isinstance(except_, tuple):
        for i, e in enumerate(except_):
            if not isinstance(e, type) or not issubclass(e, Exception):
                raise InvalidArgument(
                    f"Expected an Exception but got except_[{i}]={e!r}"
                    f" (type={_get_qualname(type(e))})"
                )
        return except_
    if not isinstance(except_, type) or not issubclass(except_, Exception):
        raise InvalidArgument(
            "Expected an Exception or tuple of exceptions, but got except_="
            f"{except_!r} (type={_get_qualname(type(except_))})"
        )
    return (except_,)


def _exception_string(except_: tuple[type[Exception], ...]) -> tuple[ImportSet, str]:
    if not except_:
        return set(), ""
    exceptions = []
    imports: ImportSet = set()
    for ex in _dedupe_exceptions(except_):
        if ex.__qualname__ in dir(builtins):
            exceptions.append(ex.__qualname__)
        else:
            imports.add(ex.__module__)
            exceptions.append(_get_qualname(ex, include_module=True))
    return imports, (
        "(" + ", ".join(exceptions) + ")" if len(exceptions) > 1 else exceptions[0]
    )


def _check_style(style: str) -> None:
    if style not in ("pytest", "unittest"):
        raise InvalidArgument(f"Valid styles are 'pytest' or 'unittest', got {style!r}")


def _exceptions_from_docstring(doc: str) -> tuple[type[Exception], ...]:
    """Return a tuple of exceptions that the docstring says may be raised.

    Note that we ignore non-builtin exception types for simplicity, as this is
    used directly in _write_call() and passing import sets around would be really
    really annoying.
    """
    # TODO: it would be great to handle Google- and Numpy-style docstrings
    #       (e.g. by using the Napoleon Sphinx extension)
    assert isinstance(doc, str), doc
    raises = []
    for excname in re.compile(r"\:raises\s+(\w+)\:", re.MULTILINE).findall(doc):
        exc_type = getattr(builtins, excname, None)
        if isinstance(exc_type, type) and issubclass(exc_type, Exception):
            raises.append(exc_type)
    return tuple(_dedupe_exceptions(tuple(raises)))


def _type_from_doc_fragment(token: str) -> Optional[type]:
    # Special cases for "integer" and for numpy array-like and dtype
    if token == "integer":
        return int
    if "numpy" in sys.modules:
        if re.fullmatch(r"[Aa]rray[-_ ]?like", token):
            return sys.modules["numpy"].ndarray
        elif token == "dtype":
            return sys.modules["numpy"].dtype
    # Natural-language syntax, e.g. "sequence of integers"
    coll_match = re.fullmatch(r"(\w+) of (\w+)", token)
    if coll_match is not None:
        coll_token, elem_token = coll_match.groups()
        elems = _type_from_doc_fragment(elem_token)
        if elems is None and elem_token.endswith("s"):
            elems = _type_from_doc_fragment(elem_token[:-1])
        if elems is not None and coll_token in ("list", "sequence", "collection"):
            return list[elems]  # type: ignore
        # This might be e.g. "array-like of float"; arrays is better than nothing
        # even if we can't conveniently pass a generic type around.
        return _type_from_doc_fragment(coll_token)
    # Check either builtins, or the module for a dotted name
    if "." not in token:
        return getattr(builtins, token, None)
    mod, name = token.rsplit(".", maxsplit=1)
    return getattr(sys.modules.get(mod, None), name, None)


def _strip_typevars(type_):
    with contextlib.suppress(Exception):
        if {type(a) for a in get_args(type_)} == {TypeVar}:
            return get_origin(type_)
    return type_


def _strategy_for(param: inspect.Parameter, docstring: str) -> st.SearchStrategy:
    # Example types in docstrings:
    # - `:type a: sequence of integers`
    # - `b (list, tuple, or None): ...`
    # - `c : {"foo", "bar", or None}`
    for pattern in (
        rf"^\s*\:type\s+{param.name}\:\s+(.+)",  # RST-style
        rf"^\s*{param.name} \((.+)\):",  # Google-style
        rf"^\s*{param.name} \: (.+)",  # Numpy-style
    ):
        match = re.search(pattern, docstring, flags=re.MULTILINE)
        if match is None:
            continue
        doc_type = match.group(1)
        if doc_type.endswith(", optional"):
            # Convention to describe "argument may be omitted"
            doc_type = doc_type[: -len(", optional")]
        doc_type = doc_type.strip("}{")
        elements = []
        types = []
        for token in re.split(r",? +or +| *, *", doc_type):
            for prefix in ("default ", "python "):
                # e.g. `str or None, default "auto"` or `python int or numpy.int64`
                token = token.removeprefix(prefix)
            if not token:
                continue
            try:
                # Elements of `{"inner", "outer"}` etc.
                elements.append(ast.literal_eval(token))
                continue
            except (ValueError, SyntaxError):
                t = _type_from_doc_fragment(token)
                if isinstance(t, type) or is_generic_type(t):
                    assert t is not None
                    types.append(_strip_typevars(t))
        if (
            param.default is not inspect.Parameter.empty
            and param.default not in elements
            and not isinstance(
                param.default, tuple(t for t in types if isinstance(t, type))
            )
        ):
            with contextlib.suppress(SyntaxError):
                compile(repr(st.just(param.default)), "<string>", "eval")
                elements.insert(0, param.default)
        if elements or types:
            return (st.sampled_from(elements) if elements else st.nothing()) | (
                st.one_of(*map(st.from_type, types)) if types else st.nothing()
            )

    # If our default value is an Enum or a boolean, we assume that any value
    # of that type is acceptable.  Otherwise, we only generate the default.
    if isinstance(param.default, bool):
        return st.booleans()
    if isinstance(param.default, enum.Enum):
        return st.sampled_from(type(param.default))
    if param.default is not inspect.Parameter.empty:
        # Using `st.from_type(type(param.default))` would  introduce spurious
        # failures in cases like the `flags` argument to regex functions.
        # Better in to keep it simple, and let the user elaborate if desired.
        return st.just(param.default)
    return _guess_strategy_by_argname(name=param.name.lower())


# fmt: off
BOOL_NAMES = (
    "keepdims", "verbose", "debug", "force", "train", "training", "trainable", "bias",
    "shuffle", "show", "load", "pretrained", "save", "overwrite", "normalize",
    "reverse", "success", "enabled", "strict", "copy", "quiet", "required", "inplace",
    "recursive", "enable", "active", "create", "validate", "refresh", "use_bias",
)
POSITIVE_INTEGER_NAMES = (
    "width", "size", "length", "limit", "idx", "stride", "epoch", "epochs", "depth",
    "pid", "steps", "iteration", "iterations", "vocab_size", "ttl", "count",
)
FLOAT_NAMES = (
    "real", "imag", "alpha", "theta", "beta", "sigma", "gamma", "angle", "reward",
    "tau", "temperature",
)
STRING_NAMES = (
    "text", "txt", "password", "label", "prefix", "suffix", "desc", "description",
    "str", "pattern", "subject", "reason", "comment", "prompt", "sentence", "sep",
)
# fmt: on


def _guess_strategy_by_argname(name: str) -> st.SearchStrategy:
    """
    If all else fails, we try guessing a strategy based on common argument names.

    We wouldn't do this in builds() where strict correctness is required, but for
    the ghostwriter we accept "good guesses" since the user would otherwise have
    to change the strategy anyway - from `nothing()` - if we refused to guess.

    A "good guess" is _usually correct_, and _a reasonable mistake_ if not.
    The logic below is therefore based on a manual reading of the builtins and
    some standard-library docs, plus the analysis of about three hundred million
    arguments in https://github.com/HypothesisWorks/hypothesis/issues/3311
    """
    # Special-cased names
    if name in ("function", "func", "f"):
        return st.functions()
    if name in ("pred", "predicate"):
        return st.functions(returns=st.booleans(), pure=True)
    if name in ("iterable",):
        return st.iterables(st.integers()) | st.iterables(st.text())
    if name in ("list", "lst", "ls"):
        return st.lists(st.nothing())
    if name in ("object",):
        return st.builds(object)
    if "uuid" in name:
        return st.uuids().map(str)

    # Names which imply the value is a boolean
    if name.startswith("is_") or name in BOOL_NAMES:
        return st.booleans()

    # Names which imply that the value is a number, perhaps in a particular range
    if name in ("amount", "threshold", "number", "num"):
        return st.integers() | st.floats()

    if name in ("port",):
        return st.integers(0, 2**16 - 1)
    if (
        name.endswith("_size")
        or (name.endswith("size") and "_" not in name)
        or re.fullmatch(r"n(um)?_[a-z_]*s", name)
        or name in POSITIVE_INTEGER_NAMES
    ):
        return st.integers(min_value=0)
    if name in ("offset", "seed", "dim", "total", "priority"):
        return st.integers()

    if name in ("learning_rate", "dropout", "dropout_rate", "epsilon", "eps", "prob"):
        return st.floats(0, 1)
    if name in ("lat", "latitude"):
        return st.floats(-90, 90)
    if name in ("lon", "longitude"):
        return st.floats(-180, 180)
    if name in ("radius", "tol", "tolerance", "rate"):
        return st.floats(min_value=0)
    if name in FLOAT_NAMES:
        return st.floats()

    # Names which imply that the value is a string
    if name in ("host", "hostname"):
        return domains()
    if name in ("email",):
        return st.emails()
    if name in ("word", "slug", "api_key"):
        return st.from_regex(r"\w+", fullmatch=True)
    if name in ("char", "character"):
        return st.characters()

    if (
        "file" in name
        or "path" in name
        or name.endswith("_dir")
        or name in ("fname", "dir", "dirname", "directory", "folder")
    ):
        # Common names for filesystem paths: these are usually strings, but we
        # don't want to make strings more convenient than pathlib.Path.
        return st.nothing()

    if (
        name.endswith(("_name", "label"))
        or (name.endswith("name") and "_" not in name)
        or ("string" in name and "as" not in name)
        or name in STRING_NAMES
    ):
        return st.text()

    # Last clever idea: maybe we're looking a plural, and know the singular:
    if re.fullmatch(r"\w*[^s]s", name):
        elems = _guess_strategy_by_argname(name[:-1])
        if not elems.is_empty:
            return st.lists(elems)

    # And if all that failed, we'll return nothing() - the user will have to
    # fill this in by hand, and we'll leave a comment to that effect later.
    return st.nothing()


def _get_params_builtin_fn(func: Callable) -> list[inspect.Parameter]:
    if (
        isinstance(func, (types.BuiltinFunctionType, types.BuiltinMethodType))
        and hasattr(func, "__doc__")
        and isinstance(func.__doc__, str)
    ):
        # inspect.signature doesn't work on all builtin functions or methods.
        # In such cases, we can try to reconstruct simple signatures from the docstring.
        match = re.match(rf"^{func.__name__}\((.+?)\)", func.__doc__)
        if match is None:
            return []
        args = match.group(1).replace("[", "").replace("]", "")
        params = []
        # Even if the signature doesn't contain a /, we assume that arguments
        # are positional-only until shown otherwise - the / is often omitted.
        kind: inspect._ParameterKind = inspect.Parameter.POSITIONAL_ONLY
        for arg in args.split(", "):
            arg, *_ = arg.partition("=")
            arg = arg.strip()
            if arg == "/":
                kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                continue
            if arg.startswith("*") or arg == "...":
                kind = inspect.Parameter.KEYWORD_ONLY
                continue  # we omit *varargs, if there are any
            if _iskeyword(arg.lstrip("*")) or not arg.lstrip("*").isidentifier():
                break  # skip all subsequent params if this name is invalid
            params.append(inspect.Parameter(name=arg, kind=kind))
        return params
    return []


def _get_params_ufunc(func: Callable) -> list[inspect.Parameter]:
    if _is_probably_ufunc(func):
        # `inspect.signature` results vary for ufunc objects, but we can work out
        # what the required parameters would look like if it was reliable.
        # Note that we use args named a, b, c... to match the `operator` module,
        # rather than x1, x2, x3... like the Numpy docs.  Because they're pos-only
        # this doesn't make a runtime difference, and it's much nicer for use-cases
        # like `equivalent(numpy.add, operator.add)`.
        return [
            inspect.Parameter(name=name, kind=inspect.Parameter.POSITIONAL_ONLY)
            for name in ascii_lowercase[: func.nin]  # type: ignore
        ]
    return []


def _get_params(func: Callable) -> dict[str, inspect.Parameter]:
    """Get non-vararg parameters of `func` as an ordered dict."""
    try:
        params = list(get_signature(func).parameters.values())
    except Exception:
        if params := _get_params_ufunc(func):
            pass
        elif params := _get_params_builtin_fn(func):
            pass
        else:
            # If we haven't managed to recover a signature through the tricks above,
            # we're out of ideas and should just re-raise the exception.
            raise
    else:
        # If the params we got look like an uninformative placeholder, try fallbacks.
        P = inspect.Parameter
        placeholder = [("args", P.VAR_POSITIONAL), ("kwargs", P.VAR_KEYWORD)]
        if [(p.name, p.kind) for p in params] == placeholder:
            params = _get_params_ufunc(func) or _get_params_builtin_fn(func) or params
    return _params_to_dict(params)


def _params_to_dict(
    params: Iterable[inspect.Parameter],
) -> dict[str, inspect.Parameter]:
    var_param_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return OrderedDict((p.name, p) for p in params if p.kind not in var_param_kinds)


@contextlib.contextmanager
def _with_any_registered():
    # If the user has registered their own strategy for Any, leave it alone
    if Any in _global_type_lookup:
        yield
    # We usually want to force from_type(Any) to raise an error because we don't
    # have enough information to accurately resolve user intent, but in this case
    # we can treat it as a synonym for object - this is probably wrong, but you'll
    # get at least _some_ output to edit later.  We then reset everything in order
    # to avoid polluting the resolution logic in case you run tests later.
    else:
        try:
            _global_type_lookup[Any] = st.builds(object)
            yield
        finally:
            del _global_type_lookup[Any]
            st.from_type.__clear_cache()


def _get_strategies(
    *funcs: Callable, pass_result_to_next_func: bool = False
) -> dict[str, st.SearchStrategy]:
    """Return a dict of strategies for the union of arguments to `funcs`.

    If `pass_result_to_next_func` is True, assume that the result of each function
    is passed to the next, and therefore skip the first argument of all but the
    first function.

    This dict is used to construct our call to the `@given(...)` decorator.
    """
    assert funcs, "Must pass at least one function"
    given_strategies: dict[str, st.SearchStrategy] = {}
    for i, f in enumerate(funcs):
        params = _get_params(f)
        if pass_result_to_next_func and i >= 1:
            del params[next(iter(params))]
        hints = get_type_hints(f)
        docstring = getattr(f, "__doc__", None) or ""
        builder_args = {
            k: ... if k in hints else _strategy_for(v, docstring)
            for k, v in params.items()
        }
        with _with_any_registered():
            strat = st.builds(f, **builder_args).wrapped_strategy  # type: ignore

        if strat.args:
            raise NotImplementedError("Expected to pass everything as kwargs")

        for k, v in strat.kwargs.items():
            if _valid_syntax_repr(v)[1] == "nothing()" and k in hints:
                # e.g. from_type(Hashable) is OK but the unwrapped repr is not
                v = LazyStrategy(st.from_type, (hints[k],), {})
            if k in given_strategies:
                given_strategies[k] |= v
            else:
                given_strategies[k] = v

    # If there is only one function, we pass arguments to @given in the order of
    # that function's signature.  Otherwise, we use alphabetical order.
    if len(funcs) == 1:
        return {name: given_strategies[name] for name in _get_params(f)}
    return dict(sorted(given_strategies.items()))


def _assert_eq(style: str, a: str, b: str) -> str:
    if style == "unittest":
        return f"self.assertEqual({a}, {b})"
    assert style == "pytest"
    if a.isidentifier() and b.isidentifier():
        return f"assert {a} == {b}, ({a}, {b})"
    return f"assert {a} == {b}"


def _imports_for_object(obj):
    """Return the imports for `obj`, which may be empty for e.g. lambdas"""
    if type(obj) is getattr(types, "UnionType", object()):
        return {mod for mod, _ in set().union(*map(_imports_for_object, obj.__args__))}
    if isinstance(obj, (re.Pattern, re.Match)):
        return {"re"}
    if isinstance(obj, st.SearchStrategy):
        return _imports_for_strategy(obj)
    if isinstance(obj, getattr(sys.modules.get("numpy"), "dtype", ())):
        return {("numpy", "dtype")}
    try:
        if is_generic_type(obj):
            if isinstance(obj, TypeVar):
                return {(obj.__module__, obj.__name__)}
            with contextlib.suppress(Exception):
                return set().union(*map(_imports_for_object, obj.__args__))
        if (not callable(obj)) or obj.__name__ == "<lambda>":
            return set()
        name = _get_qualname(obj).split(".")[0]
        return {(_get_module(obj), name)}
    except Exception:
        return set()


def _imports_for_strategy(strategy):
    # If we have a lazy from_type strategy, because unwrapping it gives us an
    # error or invalid syntax, import that type and we're done.
    if isinstance(strategy, LazyStrategy):
        imports = {
            imp
            for arg in set(strategy._LazyStrategy__args)
            | set(strategy._LazyStrategy__kwargs.values())
            for imp in _imports_for_object(_strip_typevars(arg))
        }
        if re.match(r"from_(type|regex)\(", repr(strategy)):
            return imports
        elif _get_module(strategy.function).startswith("hypothesis.extra."):
            module = _get_module(strategy.function).replace("._array_helpers", ".numpy")
            return {(module, strategy.function.__name__)} | imports

    imports = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SmallSearchSpaceWarning)
        strategy = unwrap_strategies(strategy)

    # Get imports for s.map(f), s.filter(f), s.flatmap(f), including both s and f
    if isinstance(strategy, MappedStrategy):
        imports |= _imports_for_strategy(strategy.mapped_strategy)
        imports |= _imports_for_object(strategy.pack)
    if isinstance(strategy, FilteredStrategy):
        imports |= _imports_for_strategy(strategy.filtered_strategy)
        for f in strategy.flat_conditions:
            imports |= _imports_for_object(f)
    if isinstance(strategy, FlatMapStrategy):
        imports |= _imports_for_strategy(strategy.flatmapped_strategy)
        imports |= _imports_for_object(strategy.expand)

    # recurse through one_of to handle e.g. from_type(Optional[Foo])
    if isinstance(strategy, OneOfStrategy):
        for s in strategy.element_strategies:
            imports |= _imports_for_strategy(s)

    # get imports for the target of builds(), and recurse into the argument strategies
    if isinstance(strategy, BuildsStrategy):
        imports |= _imports_for_object(strategy.target)
        for s in strategy.args:
            imports |= _imports_for_strategy(s)
        for s in strategy.kwargs.values():
            imports |= _imports_for_strategy(s)

    if isinstance(strategy, SampledFromStrategy):
        for obj in strategy.elements:
            imports |= _imports_for_object(obj)

    if isinstance(strategy, ListStrategy):
        imports |= _imports_for_strategy(strategy.element_strategy)

    return imports


def _valid_syntax_repr(strategy):
    # For binary_op, we pass a variable name - so pass it right back again.
    if isinstance(strategy, str):
        return set(), strategy
    # Flatten and de-duplicate any one_of strategies, whether that's from resolving
    # a Union type or combining inputs to multiple functions.
    try:
        if isinstance(strategy, DeferredStrategy):
            strategy = strategy.wrapped_strategy
        if isinstance(strategy, OneOfStrategy):
            seen = set()
            elems = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SmallSearchSpaceWarning)
                strategy.element_strategies  # might warn on first access
            for s in strategy.element_strategies:
                if isinstance(s, SampledFromStrategy) and s.elements == (os.environ,):
                    continue
                if repr(s) not in seen:
                    elems.append(s)
                    seen.add(repr(s))
            strategy = st.one_of(elems or st.nothing())
        # Trivial special case because the wrapped repr for text() is terrible.
        if strategy == st.text().wrapped_strategy:
            return set(), "text()"
        # Remove any typevars; we don't exploit them so they're just clutter here
        if (
            isinstance(strategy, LazyStrategy)
            and strategy.function.__name__ == st.from_type.__name__
            and strategy._LazyStrategy__representation is None
        ):
            strategy._LazyStrategy__args = tuple(
                _strip_typevars(a) for a in strategy._LazyStrategy__args
            )
        # Return a syntactically-valid strategy repr, including fixing some
        # strategy reprs and replacing invalid syntax reprs with `"nothing()"`.
        # String-replace to hide the special case in from_type() for Decimal('snan')
        r = (
            repr(strategy)
            .replace(".filter(_can_hash)", "")
            .replace("hypothesis.strategies.", "")
        )
        # Replace <unknown> with ... in confusing lambdas
        r = re.sub(r"(lambda.*?: )(<unknown>)([,)])", r"\1...\3", r)
        compile(r, "<string>", "eval")
        # Finally, try to work out the imports we need for builds(), .map(),
        # .filter(), and .flatmap() to work without NameError
        imports = {i for i in _imports_for_strategy(strategy) if i[1] in r}
        return imports, r
    except (SyntaxError, RecursionError, InvalidArgument):
        return set(), "nothing()"


# When we ghostwrite for a module, we want to treat that as the __module__ for
# each function, rather than whichever internal file it was actually defined in.
KNOWN_FUNCTION_LOCATIONS: dict[object, str] = {}


def _get_module_helper(obj):
    # Get the __module__ attribute of the object, and return the first ancestor module
    # which contains the object; falling back to the literal __module__ if none do.
    # The goal is to show location from which obj should usually be accessed, rather
    # than what we assume is an internal submodule which defined it.
    module_name = obj.__module__

    # if "collections.abc" is used don't use the deprecated aliases in "collections"
    if module_name == "collections.abc":
        return module_name

    dots = [i for i, c in enumerate(module_name) if c == "."] + [None]
    for idx in dots:
        for candidate in (module_name[:idx].lstrip("_"), module_name[:idx]):
            if getattr(sys.modules.get(candidate), obj.__name__, None) is obj:
                KNOWN_FUNCTION_LOCATIONS[obj] = candidate
                return candidate
    return module_name


def _get_module(obj):
    if obj in KNOWN_FUNCTION_LOCATIONS:
        return KNOWN_FUNCTION_LOCATIONS[obj]
    try:
        return _get_module_helper(obj)
    except AttributeError:
        if not _is_probably_ufunc(obj):
            raise
    for module_name in sorted(sys.modules, key=lambda n: tuple(n.split("."))):
        if obj is getattr(sys.modules[module_name], obj.__name__, None):
            KNOWN_FUNCTION_LOCATIONS[obj] = module_name
            return module_name
    raise RuntimeError(f"Could not find module for ufunc {obj.__name__} ({obj!r}")


def _get_qualname(obj: Any, *, include_module: bool = False) -> str:
    # Replacing angle-brackets for objects defined in `.<locals>.`
    qname = getattr(obj, "__qualname__", obj.__name__)
    qname = qname.replace("<", "_").replace(">", "_").replace(" ", "")
    if include_module:
        return _get_module(obj) + "." + qname
    return qname


def _write_call(
    func: Callable, *pass_variables: str, except_: Except = Exception, assign: str = ""
) -> str:
    """Write a call to `func` with explicit and implicit arguments.

    >>> _write_call(sorted, "my_seq", "func")
    "builtins.sorted(my_seq, key=func, reverse=reverse)"

    >>> write_call(f, assign="var1")
    "var1 = f()"

    The fancy part is that we'll check the docstring for any known exceptions
    which `func` might raise, and catch-and-reject on them... *unless* they're
    subtypes of `except_`, which will be handled in an outer try-except block.
    """
    args = ", ".join(
        (
            (v or p.name)
            if p.kind is inspect.Parameter.POSITIONAL_ONLY
            else f"{p.name}={v or p.name}"
        )
        for v, p in zip_longest(pass_variables, _get_params(func).values())
    )
    call = f"{_get_qualname(func, include_module=True)}({args})"
    if assign:
        call = f"{assign} = {call}"
    raises = _exceptions_from_docstring(getattr(func, "__doc__", "") or "")
    exnames = [ex.__name__ for ex in raises if not issubclass(ex, except_)]
    if not exnames:
        return call
    return SUPPRESS_BLOCK.format(
        test_body=indent(call, prefix="    "),
        exceptions="(" + ", ".join(exnames) + ")" if len(exnames) > 1 else exnames[0],
    )


def _st_strategy_names(s: str) -> str:
    """Replace strategy name() with st.name().

    Uses a tricky re.sub() to avoid problems with frozensets() matching
    sets() too.
    """
    names = "|".join(sorted(st.__all__, key=len, reverse=True))
    return re.sub(pattern=rf"\b(?:{names})\b[^= ]", repl=r"st.\g<0>", string=s)


def _make_test_body(
    *funcs: Callable,
    ghost: str,
    test_body: str,
    except_: tuple[type[Exception], ...],
    assertions: str = "",
    style: str,
    given_strategies: Optional[Mapping[str, Union[str, st.SearchStrategy]]] = None,
    imports: Optional[ImportSet] = None,
    annotate: bool,
) -> tuple[ImportSet, str]:
    # A set of modules to import - we might add to this later.  The import code
    # is written later, so we can have one import section for multiple magic()
    # test functions.
    imports = (imports or set()) | {_get_module(f) for f in funcs}

    # Get strategies for all the arguments to each function we're testing.
    with _with_any_registered():
        given_strategies = given_strategies or _get_strategies(
            *funcs, pass_result_to_next_func=ghost in ("idempotent", "roundtrip")
        )
        reprs = [((k, *_valid_syntax_repr(v))) for k, v in given_strategies.items()]
        imports = imports.union(*(imp for _, imp, _ in reprs))
        given_args = ", ".join(f"{k}={v}" for k, _, v in reprs)
    given_args = _st_strategy_names(given_args)

    if except_:
        # Convert to strings, either builtin names or qualified names.
        imp, exc_string = _exception_string(except_)
        imports.update(imp)
        # And finally indent the existing test body into a try-except block
        # which catches these exceptions and calls `hypothesis.reject()`.
        test_body = SUPPRESS_BLOCK.format(
            test_body=indent(test_body, prefix="    "),
            exceptions=exc_string,
        )

    if assertions:
        test_body = f"{test_body}\n{assertions}"

    # Indent our test code to form the body of a function or method.
    argnames = ["self"] if style == "unittest" else []
    if annotate:
        argnames.extend(_annotate_args(given_strategies, funcs, imports))
    else:
        argnames.extend(given_strategies)

    body = TEMPLATE.format(
        given_args=given_args,
        test_kind=ghost,
        func_name="_".join(_get_qualname(f).replace(".", "_") for f in funcs),
        arg_names=", ".join(argnames),
        return_annotation=" -> None" if annotate else "",
        test_body=indent(test_body, prefix="    "),
    )

    # For unittest-style, indent method further into a class body
    if style == "unittest":
        imports.add("unittest")
        body = "class Test{}{}(unittest.TestCase):\n{}".format(
            ghost.title(),
            "".join(_get_qualname(f).replace(".", "").title() for f in funcs),
            indent(body, "    "),
        )

    return imports, body


def _annotate_args(
    argnames: Iterable[str], funcs: Iterable[Callable], imports: ImportSet
) -> Iterable[str]:
    arg_parameters: DefaultDict[str, set[Any]] = defaultdict(set)
    for func in funcs:
        try:
            params = tuple(get_signature(func, eval_str=True).parameters.values())
        except Exception:
            # don't add parameters if the annotations could not be evaluated
            pass
        else:
            for key, param in _params_to_dict(params).items():
                if param.annotation != inspect.Parameter.empty:
                    arg_parameters[key].add(param.annotation)

    for argname in argnames:
        parameters = arg_parameters.get(argname)
        annotation = _parameters_to_annotation_name(parameters, imports)
        if annotation is None:
            yield argname
        else:
            yield f"{argname}: {annotation}"


class _AnnotationData(NamedTuple):
    type_name: str
    imports: set[str]


def _parameters_to_annotation_name(
    parameters: Optional[Iterable[Any]], imports: ImportSet
) -> Optional[str]:
    if parameters is None:
        return None
    annotations = tuple(
        annotation
        for annotation in map(_parameter_to_annotation, parameters)
        if annotation is not None
    )
    if not annotations:
        return None
    if len(annotations) == 1:
        type_name, new_imports = annotations[0]
        imports.update(new_imports)
        return type_name
    joined = _join_generics(("typing.Union", {"typing"}), annotations)
    if joined is None:
        return None
    imports.update(joined.imports)
    return joined.type_name


def _join_generics(
    origin_type_data: Optional[tuple[str, set[str]]],
    annotations: Iterable[Optional[_AnnotationData]],
) -> Optional[_AnnotationData]:
    if origin_type_data is None:
        return None

    # because typing.Optional is converted to a Union, it also contains None
    # since typing.Optional only accepts one type variable, we need to remove it
    if origin_type_data is not None and origin_type_data[0] == "typing.Optional":
        annotations = (
            annotation
            for annotation in annotations
            if annotation is None or annotation.type_name != "None"
        )

    origin_type, imports = origin_type_data
    joined = _join_argument_annotations(annotations)
    if joined is None or not joined[0]:
        return None

    arg_types, new_imports = joined
    imports.update(new_imports)
    return _AnnotationData("{}[{}]".format(origin_type, ", ".join(arg_types)), imports)


def _join_argument_annotations(
    annotations: Iterable[Optional[_AnnotationData]],
) -> Optional[tuple[list[str], set[str]]]:
    imports: set[str] = set()
    arg_types: list[str] = []

    for annotation in annotations:
        if annotation is None:
            return None
        arg_types.append(annotation.type_name)
        imports.update(annotation.imports)

    return arg_types, imports


def _parameter_to_annotation(parameter: Any) -> Optional[_AnnotationData]:
    # if a ForwardRef could not be resolved
    if isinstance(parameter, str):
        return None

    if isinstance(parameter, ForwardRef):
        forwarded_value = parameter.__forward_value__
        if forwarded_value is None:
            return None
        return _parameter_to_annotation(forwarded_value)

    # the arguments of Callable are in a list
    if isinstance(parameter, list):
        joined = _join_argument_annotations(
            _parameter_to_annotation(param) for param in parameter
        )
        if joined is None:
            return None
        arg_type_names, new_imports = joined
        return _AnnotationData("[{}]".format(", ".join(arg_type_names)), new_imports)

    if isinstance(parameter, type):
        if parameter.__module__ == "builtins":
            return _AnnotationData(
                "None" if parameter.__name__ == "NoneType" else parameter.__name__,
                set(),
            )

        type_name = _get_qualname(parameter, include_module=True)

        # the types.UnionType does not support type arguments and needs to be translated
        if type_name == "types.UnionType":
            return _AnnotationData("typing.Union", {"typing"})
    else:
        if hasattr(parameter, "__module__") and hasattr(parameter, "__name__"):
            type_name = _get_qualname(parameter, include_module=True)
        else:
            type_name = str(parameter)

    if type_name.startswith("hypothesis.strategies."):
        return _AnnotationData(type_name.replace("hypothesis.strategies", "st"), set())

    origin_type = get_origin(parameter)

    # if not generic or no generic arguments
    if origin_type is None or origin_type == parameter:
        return _AnnotationData(type_name, set(type_name.rsplit(".", maxsplit=1)[:-1]))

    arg_types = get_args(parameter)
    if {type(a) for a in arg_types} == {TypeVar}:
        arg_types = ()

    # typing types get translated to classes that don't support generics
    origin_annotation: Optional[_AnnotationData]
    if type_name.startswith("typing."):
        try:
            new_type_name = type_name[: type_name.index("[")]
        except ValueError:
            new_type_name = type_name
        origin_annotation = _AnnotationData(new_type_name, {"typing"})
    else:
        origin_annotation = _parameter_to_annotation(origin_type)

    if arg_types:
        return _join_generics(
            origin_annotation,
            (_parameter_to_annotation(arg_type) for arg_type in arg_types),
        )
    return origin_annotation


def _are_annotations_used(*functions: Callable) -> bool:
    for function in functions:
        try:
            params = get_signature(function).parameters.values()
        except Exception:
            pass
        else:
            if any(param.annotation != inspect.Parameter.empty for param in params):
                return True
    return False


def _make_test(imports: ImportSet, body: str) -> str:
    # Discarding "builtins." and "__main__" probably isn't particularly useful
    # for user code, but important for making a good impression in demos.
    body = body.replace("builtins.", "").replace("__main__.", "")
    imports |= {("hypothesis", "given"), ("hypothesis", "strategies as st")}
    if "        reject()\n" in body:
        imports.add(("hypothesis", "reject"))

    do_not_import = {"builtins", "__main__", "hypothesis.strategies"}
    direct = {f"import {i}" for i in imports - do_not_import if isinstance(i, str)}
    from_imports = defaultdict(set)
    for module, name in {i for i in imports if isinstance(i, tuple)}:
        if not (module.startswith("hypothesis.strategies") and name in st.__all__):
            from_imports[module].add(name)
    from_ = {
        "from {} import {}".format(module, ", ".join(sorted(names)))
        for module, names in from_imports.items()
        if isinstance(module, str) and module not in do_not_import
    }
    header = IMPORT_SECTION.format(imports="\n".join(sorted(direct) + sorted(from_)))
    nothings = body.count("st.nothing()")
    if nothings == 1:
        header += "# TODO: replace st.nothing() with an appropriate strategy\n\n"
    elif nothings >= 1:
        header += "# TODO: replace st.nothing() with appropriate strategies\n\n"
    return black.format_str(header + body, mode=black.FileMode())


def _is_probably_ufunc(obj):
    # See https://numpy.org/doc/stable/reference/ufuncs.html - there doesn't seem
    # to be an upstream function to detect this, so we just guess.
    has_attributes = "nin nout nargs ntypes types identity signature".split()
    return callable(obj) and all(hasattr(obj, name) for name in has_attributes)


# If we have a pair of functions where one name matches the regex and the second
# is the result of formatting the template with matched groups, our magic()
# ghostwriter will write a roundtrip test for them.  Additional patterns welcome.
ROUNDTRIP_PAIRS = (
    # Defined prefix, shared postfix.  The easy cases.
    (r"write(.+)", "read{}"),
    (r"save(.+)", "load{}"),
    (r"dump(.+)", "load{}"),
    (r"to(.+)", "from{}"),
    # Known stem, maybe matching prefixes, maybe matching postfixes.
    (r"(.*)en(.+)", "{}de{}"),
    # Shared postfix, prefix only on "inverse" function
    (r"(.+)", "de{}"),
    (r"(?!safe)(.+)", "un{}"),  # safe_load / unsafe_load isn't a roundtrip
    # a2b_postfix and b2a_postfix.  Not a fan of this pattern, but it's pretty
    # common in code imitating an C API - see e.g. the stdlib binascii module.
    (r"(.+)2(.+?)(_.+)?", "{1}2{0}{2}"),
    # Common in e.g. the colorsys module
    (r"(.+)_to_(.+)", "{1}_to_{0}"),
    # Sockets patterns
    (r"(inet|if)_(.+)to(.+)", "{0}_{2}to{1}"),
    (r"(\w)to(\w)(.+)", "{1}to{0}{2}"),
    (r"send(.+)", "recv{}"),
    (r"send(.+)", "receive{}"),
)


def _get_testable_functions(thing: object) -> dict[str, Callable]:
    by_name = {}
    if callable(thing):
        funcs: list[Optional[Any]] = [thing]
    elif isinstance(thing, types.ModuleType):
        if hasattr(thing, "__all__"):
            funcs = [getattr(thing, name, None) for name in thing.__all__]
        elif hasattr(thing, "__package__"):
            pkg = thing.__package__
            funcs = [
                v
                for k, v in vars(thing).items()
                if callable(v)
                and not is_mock(v)
                and ((not pkg) or getattr(v, "__module__", pkg).startswith(pkg))
                and not k.startswith("_")
            ]
            if pkg and any(getattr(f, "__module__", pkg) == pkg for f in funcs):
                funcs = [f for f in funcs if getattr(f, "__module__", pkg) == pkg]
    else:
        raise InvalidArgument(f"Can't test non-module non-callable {thing!r}")

    for f in list(funcs):
        if inspect.isclass(f):
            funcs += [
                v.__get__(f)
                for k, v in vars(f).items()
                if hasattr(v, "__func__") and not is_mock(v) and not k.startswith("_")
            ]
    for f in funcs:
        try:
            if (
                (not is_mock(f))
                and callable(f)
                and _get_params(f)
                and not isinstance(f, enum.EnumMeta)
            ):
                if getattr(thing, "__name__", None):
                    if inspect.isclass(thing):
                        KNOWN_FUNCTION_LOCATIONS[f] = _get_module_helper(thing)
                    elif isinstance(thing, types.ModuleType):
                        KNOWN_FUNCTION_LOCATIONS[f] = thing.__name__
                try:
                    _get_params(f)
                    by_name[_get_qualname(f, include_module=True)] = f
                except Exception:
                    # usually inspect.signature on C code such as socket.inet_aton,
                    # or Pandas 'CallableDynamicDoc' object has no attr. '__name__'
                    pass
        except (TypeError, ValueError):
            pass
    return by_name


def magic(
    *modules_or_functions: Union[Callable, types.ModuleType],
    except_: Except = (),
    style: str = "pytest",
    annotate: Optional[bool] = None,
) -> str:
    """Guess which ghostwriters to use, for a module or collection of functions.

    As for all ghostwriters, the ``except_`` argument should be an
    :class:`python:Exception` or tuple of exceptions, and ``style`` may be either
    ``"pytest"`` to write test functions or ``"unittest"`` to write test methods
    and :class:`~python:unittest.TestCase`.

    After finding the public functions attached to any modules, the ``magic``
    ghostwriter looks for pairs of functions to pass to :func:`~roundtrip`,
    then checks for :func:`~binary_operation` and :func:`~ufunc` functions,
    and any others are passed to :func:`~fuzz`.

    For example, try :command:`hypothesis write gzip` on the command line!
    """
    except_ = _check_except(except_)
    _check_style(style)
    if not modules_or_functions:
        raise InvalidArgument("Must pass at least one function or module to test.")

    parts = []
    by_name = {}
    imports = set()

    for thing in modules_or_functions:
        by_name.update(found := _get_testable_functions(thing))
        if (not found) and isinstance(thing, types.ModuleType):
            msg = f"# Found no testable functions in {thing.__name__} (from {thing.__file__!r})"
            mods: list = []
            for k in sorted(sys.modules, key=len):
                if (
                    k.startswith(f"{thing.__name__}.")
                    and "._" not in k.removeprefix(thing.__name__)
                    and not k.startswith(tuple(f"{m}." for m in mods))
                    and _get_testable_functions(sys.modules[k])
                ):
                    mods.append(k)
            if mods:
                msg += (
                    f"\n# Try writing tests for submodules, e.g. by using:\n"
                    f"#     hypothesis write {' '.join(sorted(mods))}"
                )
            parts.append(msg)

    if not by_name:
        return "\n\n".join(parts)

    if annotate is None:
        annotate = _are_annotations_used(*by_name.values())

    def make_(how, *args, **kwargs):
        imp, body = how(*args, **kwargs, except_=except_, style=style)
        imports.update(imp)
        parts.append(body)

    # Look for pairs of functions that roundtrip, based on known naming patterns.
    for writename, readname in ROUNDTRIP_PAIRS:
        for name in sorted(by_name):
            match = re.fullmatch(writename, name.split(".")[-1])
            if match:
                inverse_name = readname.format(*match.groups())
                for other in sorted(
                    n for n in by_name if n.split(".")[-1] == inverse_name
                ):
                    make_(
                        _make_roundtrip_body,
                        (by_name.pop(name), by_name.pop(other)),
                        annotate=annotate,
                    )
                    break
                else:
                    try:
                        other_func = getattr(
                            sys.modules[_get_module(by_name[name])],
                            inverse_name,
                        )
                        _get_params(other_func)  # we want to skip if this fails
                    except Exception:
                        pass
                    else:
                        make_(
                            _make_roundtrip_body,
                            (by_name.pop(name), other_func),
                            annotate=annotate,
                        )

    # Look for equivalent functions: same name, all required arguments of any can
    # be found in all signatures, and if all have return-type annotations they match.
    names = defaultdict(list)
    for _, f in sorted(by_name.items()):
        names[_get_qualname(f)].append(f)
    for group in names.values():
        if len(group) >= 2 and len({frozenset(_get_params(f)) for f in group}) == 1:
            sentinel = object()
            returns = {get_type_hints(f).get("return", sentinel) for f in group}
            if len(returns - {sentinel}) <= 1:
                make_(_make_equiv_body, group, annotate=annotate)
                for f in group:
                    by_name.pop(_get_qualname(f, include_module=True))

    # Look for binary operators - functions with two identically-typed arguments,
    # and the same return type.  The latter restriction might be lifted later.
    for name, func in sorted(by_name.items()):
        hints = get_type_hints(func)
        hints.pop("return", None)
        params = _get_params(func)
        if (len(hints) == len(params) == 2) or (
            _get_module(func) == "operator"
            and "item" not in func.__name__
            and tuple(params) in [("a", "b"), ("x", "y")]
        ):
            a, b = hints.values() or [Any, Any]
            arg1, arg2 = params
            if a == b and len(arg1) == len(arg2) <= 3:
                # https://en.wikipedia.org/wiki/Distributive_property#Other_examples
                known = {
                    "mul": "add",
                    "matmul": "add",
                    "or_": "and_",
                    "and_": "or_",
                }.get(func.__name__, "")
                distributes_over = getattr(sys.modules[_get_module(func)], known, None)
                make_(
                    _make_binop_body,
                    func,
                    commutative=func.__name__ != "matmul",
                    distributes_over=distributes_over,
                    annotate=annotate,
                )
                del by_name[name]

    # Look for Numpy ufuncs or gufuncs, and write array-oriented tests for them.
    if "numpy" in sys.modules:
        for name, func in sorted(by_name.items()):
            if _is_probably_ufunc(func):
                make_(_make_ufunc_body, func, annotate=annotate)
                del by_name[name]

    # For all remaining callables, just write a fuzz-test.  In principle we could
    # guess at equivalence or idempotence; but it doesn't seem accurate enough to
    # be worth the trouble when it's so easy for the user to specify themselves.
    for _, f in sorted(by_name.items()):
        make_(
            _make_test_body,
            f,
            test_body=_write_call(f, except_=except_),
            ghost="fuzz",
            annotate=annotate,
        )
    return _make_test(imports, "\n".join(parts))


def fuzz(
    func: Callable,
    *,
    except_: Except = (),
    style: str = "pytest",
    annotate: Optional[bool] = None,
) -> str:
    """Write source code for a property-based test of ``func``.

    The resulting test checks that valid input only leads to expected exceptions.
    For example:

    .. code-block:: python

        from re import compile, error

        from hypothesis.extra import ghostwriter

        ghostwriter.fuzz(compile, except_=error)

    Gives:

    .. code-block:: python

        # This test code was written by the `hypothesis.extra.ghostwriter` module
        # and is provided under the Creative Commons Zero public domain dedication.
        import re

        from hypothesis import given, reject, strategies as st

        # TODO: replace st.nothing() with an appropriate strategy


        @given(pattern=st.nothing(), flags=st.just(0))
        def test_fuzz_compile(pattern, flags):
            try:
                re.compile(pattern=pattern, flags=flags)
            except re.error:
                reject()

    Note that it includes all the required imports.
    Because the ``pattern`` parameter doesn't have annotations or a default argument,
    you'll need to specify a strategy - for example :func:`~hypothesis.strategies.text`
    or :func:`~hypothesis.strategies.binary`.  After that, you have a test!
    """
    if not callable(func):
        raise InvalidArgument(f"Got non-callable {func=}")
    except_ = _check_except(except_)
    _check_style(style)

    if annotate is None:
        annotate = _are_annotations_used(func)

    imports, body = _make_test_body(
        func,
        test_body=_write_call(func, except_=except_),
        except_=except_,
        ghost="fuzz",
        style=style,
        annotate=annotate,
    )
    return _make_test(imports, body)


def idempotent(
    func: Callable,
    *,
    except_: Except = (),
    style: str = "pytest",
    annotate: Optional[bool] = None,
) -> str:
    """Write source code for a property-based test of ``func``.

    The resulting test checks that if you call ``func`` on it's own output,
    the result does not change.  For example:

    .. code-block:: python

        from typing import Sequence

        from hypothesis.extra import ghostwriter


        def timsort(seq: Sequence[int]) -> Sequence[int]:
            return sorted(seq)


        ghostwriter.idempotent(timsort)

    Gives:

    .. code-block:: python

        # This test code was written by the `hypothesis.extra.ghostwriter` module
        # and is provided under the Creative Commons Zero public domain dedication.

        from hypothesis import given, strategies as st


        @given(seq=st.one_of(st.binary(), st.binary().map(bytearray), st.lists(st.integers())))
        def test_idempotent_timsort(seq):
            result = timsort(seq=seq)
            repeat = timsort(seq=result)
            assert result == repeat, (result, repeat)
    """
    if not callable(func):
        raise InvalidArgument(f"Got non-callable {func=}")
    except_ = _check_except(except_)
    _check_style(style)

    if annotate is None:
        annotate = _are_annotations_used(func)

    imports, body = _make_test_body(
        func,
        test_body="result = {}\nrepeat = {}".format(
            _write_call(func, except_=except_),
            _write_call(func, "result", except_=except_),
        ),
        except_=except_,
        assertions=_assert_eq(style, "result", "repeat"),
        ghost="idempotent",
        style=style,
        annotate=annotate,
    )
    return _make_test(imports, body)


def _make_roundtrip_body(funcs, except_, style, annotate):
    first_param = next(iter(_get_params(funcs[0])))
    test_lines = [
        _write_call(funcs[0], assign="value0", except_=except_),
        *(
            _write_call(f, f"value{i}", assign=f"value{i + 1}", except_=except_)
            for i, f in enumerate(funcs[1:])
        ),
    ]
    return _make_test_body(
        *funcs,
        test_body="\n".join(test_lines),
        except_=except_,
        assertions=_assert_eq(style, first_param, f"value{len(funcs) - 1}"),
        ghost="roundtrip",
        style=style,
        annotate=annotate,
    )


def roundtrip(
    *funcs: Callable,
    except_: Except = (),
    style: str = "pytest",
    annotate: Optional[bool] = None,
) -> str:
    """Write source code for a property-based test of ``funcs``.

    The resulting test checks that if you call the first function, pass the result
    to the second (and so on), the final result is equal to the first input argument.

    This is a *very* powerful property to test, especially when the config options
    are varied along with the object to round-trip.  For example, try ghostwriting
    a test for :func:`python:json.dumps` - would you have thought of all that?

    .. code-block:: shell

        hypothesis write --roundtrip json.dumps json.loads
    """
    if not funcs:
        raise InvalidArgument("Round-trip of zero functions is meaningless.")
    for i, f in enumerate(funcs):
        if not callable(f):
            raise InvalidArgument(f"Got non-callable funcs[{i}]={f!r}")
    except_ = _check_except(except_)
    _check_style(style)

    if annotate is None:
        annotate = _are_annotations_used(*funcs)

    return _make_test(*_make_roundtrip_body(funcs, except_, style, annotate))


def _get_varnames(funcs):
    var_names = [f"result_{f.__name__}" for f in funcs]
    if len(set(var_names)) < len(var_names):
        var_names = [f"result_{f.__name__}_{_get_module(f)}" for f in funcs]
    if len(set(var_names)) < len(var_names):
        var_names = [f"result_{i}_{f.__name__}" for i, f in enumerate(funcs)]
    return var_names


def _make_equiv_body(funcs, except_, style, annotate):
    var_names = _get_varnames(funcs)
    test_lines = [
        _write_call(f, assign=vname, except_=except_)
        for vname, f in zip(var_names, funcs)
    ]
    assertions = "\n".join(
        _assert_eq(style, var_names[0], vname) for vname in var_names[1:]
    )

    return _make_test_body(
        *funcs,
        test_body="\n".join(test_lines),
        except_=except_,
        assertions=assertions,
        ghost="equivalent",
        style=style,
        annotate=annotate,
    )


EQUIV_FIRST_BLOCK = """
try:
{}
    exc_type = None
    target(1, label="input was valid")
{}except Exception as exc:
    exc_type = type(exc)
""".strip()

EQUIV_CHECK_BLOCK = """
if exc_type:
    with {ctx}(exc_type):
{check_raises}
else:
{call}
{compare}
""".rstrip()


def _make_equiv_errors_body(funcs, except_, style, annotate):
    var_names = _get_varnames(funcs)
    first, *rest = funcs
    first_call = _write_call(first, assign=var_names[0], except_=except_)
    extra_imports, suppress = _exception_string(except_)
    extra_imports.add(("hypothesis", "target"))
    catch = f"except {suppress}:\n    reject()\n" if suppress else ""
    test_lines = [EQUIV_FIRST_BLOCK.format(indent(first_call, prefix="    "), catch)]

    for vname, f in zip(var_names[1:], rest):
        if style == "pytest":
            ctx = "pytest.raises"
            extra_imports.add("pytest")
        else:
            assert style == "unittest"
            ctx = "self.assertRaises"
        block = EQUIV_CHECK_BLOCK.format(
            ctx=ctx,
            check_raises=indent(_write_call(f, except_=()), "        "),
            call=indent(_write_call(f, assign=vname, except_=()), "    "),
            compare=indent(_assert_eq(style, var_names[0], vname), "    "),
        )
        test_lines.append(block)

    imports, source_code = _make_test_body(
        *funcs,
        test_body="\n".join(test_lines),
        except_=(),
        ghost="equivalent",
        style=style,
        annotate=annotate,
    )
    return imports | extra_imports, source_code


def equivalent(
    *funcs: Callable,
    allow_same_errors: bool = False,
    except_: Except = (),
    style: str = "pytest",
    annotate: Optional[bool] = None,
) -> str:
    """Write source code for a property-based test of ``funcs``.

    The resulting test checks that calling each of the functions returns
    an equal value.  This can be used as a classic 'oracle', such as testing
    a fast sorting algorithm against the :func:`python:sorted` builtin, or
    for differential testing where none of the compared functions are fully
    trusted but any difference indicates a bug (e.g. running a function on
    different numbers of threads, or simply multiple times).

    The functions should have reasonably similar signatures, as only the
    common parameters will be passed the same arguments - any other parameters
    will be allowed to vary.

    If allow_same_errors is True, then the test will pass if calling each of
    the functions returns an equal value, *or* if the first function raises an
    exception and each of the others raises an exception of the same type.
    This relaxed mode can be useful for code synthesis projects.
    """
    if len(funcs) < 2:
        raise InvalidArgument("Need at least two functions to compare.")
    for i, f in enumerate(funcs):
        if not callable(f):
            raise InvalidArgument(f"Got non-callable funcs[{i}]={f!r}")
    check_type(bool, allow_same_errors, "allow_same_errors")
    except_ = _check_except(except_)
    _check_style(style)

    if annotate is None:
        annotate = _are_annotations_used(*funcs)

    if allow_same_errors and not any(issubclass(Exception, ex) for ex in except_):
        imports, source_code = _make_equiv_errors_body(funcs, except_, style, annotate)
    else:
        imports, source_code = _make_equiv_body(funcs, except_, style, annotate)
    return _make_test(imports, source_code)


X = TypeVar("X")
Y = TypeVar("Y")


def binary_operation(
    func: Callable[[X, X], Y],
    *,
    associative: bool = True,
    commutative: bool = True,
    identity: Union[X, EllipsisType, None] = ...,
    distributes_over: Optional[Callable[[X, X], X]] = None,
    except_: Except = (),
    style: str = "pytest",
    annotate: Optional[bool] = None,
) -> str:
    """Write property tests for the binary operation ``func``.

    While :wikipedia:`binary operations <Binary_operation>` are not particularly
    common, they have such nice properties to test that it seems a shame not to
    demonstrate them with a ghostwriter.  For an operator ``f``, test that:

    - if :wikipedia:`associative <Associative_property>`,
      ``f(a, f(b, c)) == f(f(a, b), c)``
    - if :wikipedia:`commutative <Commutative_property>`, ``f(a, b) == f(b, a)``
    - if :wikipedia:`identity <Identity_element>` is not None, ``f(a, identity) == a``
    - if :wikipedia:`distributes_over <Distributive_property>` is ``+``,
      ``f(a, b) + f(a, c) == f(a, b+c)``

    For example:

    .. code-block:: python

        ghostwriter.binary_operation(
            operator.mul,
            identity=1,
            distributes_over=operator.add,
            style="unittest",
        )
    """
    if not callable(func):
        raise InvalidArgument(f"Got non-callable {func=}")
    except_ = _check_except(except_)
    _check_style(style)
    check_type(bool, associative, "associative")
    check_type(bool, commutative, "commutative")
    if distributes_over is not None and not callable(distributes_over):
        raise InvalidArgument(
            f"{distributes_over=} must be an operation which "
            f"distributes over {func.__name__}"
        )
    if not any([associative, commutative, identity, distributes_over]):
        raise InvalidArgument(
            "You must select at least one property of the binary operation to test."
        )

    if annotate is None:
        annotate = _are_annotations_used(func)

    imports, body = _make_binop_body(
        func,
        associative=associative,
        commutative=commutative,
        identity=identity,
        distributes_over=distributes_over,
        except_=except_,
        style=style,
        annotate=annotate,
    )
    return _make_test(imports, body)


def _make_binop_body(
    func: Callable[[X, X], Y],
    *,
    associative: bool = True,
    commutative: bool = True,
    identity: Union[X, EllipsisType, None] = ...,
    distributes_over: Optional[Callable[[X, X], X]] = None,
    except_: tuple[type[Exception], ...],
    style: str,
    annotate: bool,
) -> tuple[ImportSet, str]:
    strategies = _get_strategies(func)
    operands, b = (strategies.pop(p) for p in list(_get_params(func))[:2])
    if repr(operands) != repr(b):
        operands |= b
    operands_name = func.__name__ + "_operands"

    all_imports = set()
    parts = []

    def maker(
        sub_property: str,
        args: str,
        body: str,
        right: Optional[str] = None,
    ) -> None:
        if right is None:
            assertions = ""
        else:
            body = f"{body}\n{right}"
            assertions = _assert_eq(style, "left", "right")
        imports, body = _make_test_body(
            func,
            test_body=body,
            ghost=sub_property + "_binary_operation",
            except_=except_,
            assertions=assertions,
            style=style,
            given_strategies={**strategies, **{n: operands_name for n in args}},
            annotate=annotate,
        )
        all_imports.update(imports)
        if style == "unittest":
            endline = "(unittest.TestCase):\n"
            body = body[body.index(endline) + len(endline) + 1 :]
        parts.append(body)

    if associative:
        maker(
            "associative",
            "abc",
            _write_call(func, "a", _write_call(func, "b", "c"), assign="left"),
            _write_call(
                func,
                _write_call(func, "a", "b"),
                "c",
                assign="right",
            ),
        )
    if commutative:
        maker(
            "commutative",
            "ab",
            _write_call(func, "a", "b", assign="left"),
            _write_call(func, "b", "a", assign="right"),
        )
    if identity is not None:
        # Guess that the identity element is the minimal example from our operands
        # strategy.  This is correct often enough to be worthwhile, and close enough
        # that it's a good starting point to edit much of the rest.
        if identity is ...:
            try:
                identity = find(operands, lambda x: True, settings=_quietly_settings)
            except Exception:
                identity = "identity element here"  # type: ignore
        # If the repr of this element is invalid Python, stringify it - this
        # can't be executed as-is, but at least makes it clear what should
        # happen.  E.g. type(None) -> <class 'NoneType'> -> quoted.
        try:
            # We don't actually execute this code object; we're just compiling
            # to check that the repr is syntactically valid.  HOWEVER, we're
            # going to output that code string into test code which will be
            # executed; so you still shouldn't ghostwrite for hostile code.
            compile(repr(identity), "<string>", "exec")
        except SyntaxError:
            identity = repr(identity)  # type: ignore
        identity_parts = [
            f"{identity = }",
            _assert_eq(
                style,
                "a",
                _write_call(func, "a", "identity"),
            ),
            _assert_eq(
                style,
                "a",
                _write_call(func, "identity", "a"),
            ),
        ]
        maker("identity", "a", "\n".join(identity_parts))
    if distributes_over:
        do = distributes_over
        dist_parts = [
            _write_call(func, "a", _write_call(do, "b", "c"), assign="left"),
            _write_call(
                do,
                _write_call(func, "a", "b"),
                _write_call(func, "a", "c"),
                assign="ldist",
            ),
            _assert_eq(style, "ldist", "left"),
            "\n",
            _write_call(func, _write_call(do, "a", "b"), "c", assign="right"),
            _write_call(
                do,
                _write_call(func, "a", "c"),
                _write_call(func, "b", "c"),
                assign="rdist",
            ),
            _assert_eq(style, "rdist", "right"),
        ]
        maker(do.__name__ + "_distributes_over", "abc", "\n".join(dist_parts))

    _, operands_repr = _valid_syntax_repr(operands)
    operands_repr = _st_strategy_names(operands_repr)
    classdef = ""
    if style == "unittest":
        classdef = f"class TestBinaryOperation{func.__name__}(unittest.TestCase):\n    "
    return (
        all_imports,
        classdef + f"{operands_name} = {operands_repr}\n" + "\n".join(parts),
    )


def ufunc(
    func: Callable,
    *,
    except_: Except = (),
    style: str = "pytest",
    annotate: Optional[bool] = None,
) -> str:
    """Write a property-based test for the :doc:`array ufunc <numpy:reference/ufuncs>` ``func``.

    The resulting test checks that your ufunc or :doc:`gufunc
    <numpy:reference/c-api/generalized-ufuncs>` has the expected broadcasting and dtype casting
    behaviour.  You will probably want to add extra assertions, but as with the other
    ghostwriters this gives you a great place to start.

    .. code-block:: shell

        hypothesis write numpy.matmul
    """
    if not _is_probably_ufunc(func):
        raise InvalidArgument(f"{func=} does not seem to be a ufunc")
    except_ = _check_except(except_)
    _check_style(style)

    if annotate is None:
        annotate = _are_annotations_used(func)

    return _make_test(
        *_make_ufunc_body(func, except_=except_, style=style, annotate=annotate)
    )


def _make_ufunc_body(func, *, except_, style, annotate):
    import hypothesis.extra.numpy as npst

    if func.signature is None:
        shapes = npst.mutually_broadcastable_shapes(num_shapes=func.nin)
    else:
        shapes = npst.mutually_broadcastable_shapes(signature=func.signature)
    shapes.function.__module__ = npst.__name__

    body = """
    input_shapes, expected_shape = shapes
    input_dtypes, expected_dtype = types.split("->")
    array_strats = [
        arrays(dtype=dtp, shape=shp, elements={{"allow_nan": True}})
        for dtp, shp in zip(input_dtypes, input_shapes)
    ]

    {array_names} = data.draw(st.tuples(*array_strats))
    result = {call}
    """.format(
        array_names=", ".join(ascii_lowercase[: func.nin]),
        call=_write_call(func, *ascii_lowercase[: func.nin], except_=except_),
    )
    assertions = "\n{shape_assert}\n{type_assert}".format(
        shape_assert=_assert_eq(style, "result.shape", "expected_shape"),
        type_assert=_assert_eq(style, "result.dtype.char", "expected_dtype"),
    )

    qname = _get_qualname(func, include_module=True)
    obj_sigs = ["O" in sig for sig in func.types]
    if all(obj_sigs) or not any(obj_sigs):
        types = f"sampled_from({qname}.types)"
    else:
        types = f"sampled_from([sig for sig in {qname}.types if 'O' not in sig])"

    return _make_test_body(
        func,
        test_body=dedent(body).strip(),
        except_=except_,
        assertions=assertions,
        ghost="ufunc" if func.signature is None else "gufunc",
        style=style,
        given_strategies={"data": st.data(), "shapes": shapes, "types": types},
        imports={("hypothesis.extra.numpy", "arrays")},
        annotate=annotate,
    )
