# mypy: allow-untyped-defs
import inspect
import itertools as itl
from typing_extensions import deprecated
from warnings import warn

from .conflict import ambiguities, AmbiguityWarning, ordering, super_signature
from .utils import expand_tuples
from .variadic import isvariadic, Variadic


__all__ = [
    "MDNotImplementedError",
    "ambiguity_warn",
    "halt_ordering",
    "restart_ordering",
    "variadic_signature_matches_iter",
    "variadic_signature_matches",
    "Dispatcher",
    "source",
    "MethodDispatcher",
    "str_signature",
    "warning_text",
]


class MDNotImplementedError(NotImplementedError):
    """A NotImplementedError for multiple dispatch"""


def ambiguity_warn(dispatcher, ambiguities):
    """Raise warning when ambiguity is detected
    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher
    See Also:
        Dispatcher.add
        warning_text
    """
    warn(warning_text(dispatcher.name, ambiguities), AmbiguityWarning)


@deprecated(
    "`halt_ordering` is deprecated, you can safely remove this call.",
    category=FutureWarning,
)
def halt_ordering():
    """Deprecated interface to temporarily disable ordering."""


@deprecated(
    "`restart_ordering` is deprecated, if you would like to eagerly order the dispatchers, "
    "you should call the `reorder()` method on each dispatcher.",
    category=FutureWarning,
)
def restart_ordering(on_ambiguity=ambiguity_warn):
    """Deprecated interface to temporarily resume ordering."""


def variadic_signature_matches_iter(types, full_signature):
    """Check if a set of input types matches a variadic signature.
    Notes
    -----
    The algorithm is as follows:
    Initialize the current signature to the first in the sequence
    For each type in `types`:
        If the current signature is variadic
            If the type matches the signature
                yield True
            Else
                Try to get the next signature
                If no signatures are left we can't possibly have a match
                    so yield False
        Else
            yield True if the type matches the current signature
            Get the next signature
    """
    sigiter = iter(full_signature)
    sig = next(sigiter)
    for typ in types:
        matches = issubclass(typ, sig)
        yield matches
        if not isvariadic(sig):
            # we're not matching a variadic argument, so move to the next
            # element in the signature
            sig = next(sigiter)
    else:
        try:
            sig = next(sigiter)
        except StopIteration:
            assert isvariadic(sig)
            yield True
        else:
            # We have signature items left over, so all of our arguments
            # haven't matched
            yield False


def variadic_signature_matches(types, full_signature):
    # No arguments always matches a variadic signature
    assert full_signature
    return all(variadic_signature_matches_iter(types, full_signature))


class Dispatcher:
    """Dispatch methods based on type signature
    Use ``dispatch`` to add implementations
    Examples
    --------
    >>> # xdoctest: +SKIP("bad import name")
    >>> from multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1
    >>> @dispatch(float)
    ... def f(x):
    ...     return x - 1
    >>> f(3)
    4
    >>> f(3.0)
    2.0
    """

    __slots__ = "__name__", "name", "funcs", "_ordering", "_cache", "doc"

    def __init__(self, name, doc=None):
        self.name = self.__name__ = name
        self.funcs = {}
        self.doc = doc

        self._cache = {}

    def register(self, *types, **kwargs):
        """register dispatcher with new implementation
        >>> # xdoctest: +SKIP
        >>> f = Dispatcher("f")
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1
        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1
        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]
        >>> f(1)
        2
        >>> f(1.0)
        0.0
        >>> f([1, 2, 3])
        [3, 2, 1]
        """

        def _df(func):
            self.add(types, func, **kwargs)
            return func

        return _df

    @classmethod
    def get_func_params(cls, func):
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return sig.parameters.values()

    @classmethod
    def get_func_annotations(cls, func):
        """get annotations of function positional parameters"""
        params = cls.get_func_params(func)
        if params:
            Parameter = inspect.Parameter

            params = (
                param
                for param in params
                if param.kind
                in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            )

            annotations = tuple(param.annotation for param in params)

            if all(ann is not Parameter.empty for ann in annotations):
                return annotations

    def add(self, signature, func):
        """Add new types/method pair to dispatcher
        >>> # xdoctest: +SKIP
        >>> D = Dispatcher("add")
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)
        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>
        >>> # When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        >>> # with a dispatcher/itself, and a set of ambiguous type signature pairs
        >>> # as inputs.  See ``ambiguity_warn`` for an example.
        """
        # Handle annotations
        if not signature:
            annotations = self.get_func_annotations(func)
            if annotations:
                signature = annotations

        # Handle union types
        if any(isinstance(typ, tuple) for typ in signature):
            for typs in expand_tuples(signature):
                self.add(typs, func)
            return

        new_signature = []

        for index, typ in enumerate(signature, start=1):
            if not isinstance(typ, (type, list)):
                str_sig = ", ".join(
                    c.__name__ if isinstance(c, type) else str(c) for c in signature
                )
                raise TypeError(
                    f"Tried to dispatch on non-type: {typ}\n"
                    f"In signature: <{str_sig}>\n"
                    f"In function: {self.name}"
                )

            # handle variadic signatures
            if isinstance(typ, list):
                if index != len(signature):
                    raise TypeError("Variadic signature must be the last element")

                if len(typ) != 1:
                    raise TypeError(
                        "Variadic signature must contain exactly one element. "
                        "To use a variadic union type place the desired types "
                        "inside of a tuple, e.g., [(int, str)]"
                    )
                new_signature.append(Variadic[typ[0]])
            else:
                new_signature.append(typ)

        self.funcs[tuple(new_signature)] = func
        self._cache.clear()

        try:
            del self._ordering
        except AttributeError:
            pass

    @property
    def ordering(self):
        try:
            return self._ordering
        except AttributeError:
            return self.reorder()

    def reorder(self, on_ambiguity=ambiguity_warn):
        self._ordering = od = ordering(self.funcs)
        amb = ambiguities(self.funcs)
        if amb:
            on_ambiguity(self, amb)
        return od

    def __call__(self, *args, **kwargs):
        types = tuple([type(arg) for arg in args])
        try:
            func = self._cache[types]
        except KeyError as e:
            func = self.dispatch(*types)
            if not func:
                raise NotImplementedError(
                    f"Could not find signature for {self.name}: <{str_signature(types)}>"
                ) from e
            self._cache[types] = func
        try:
            return func(*args, **kwargs)

        except MDNotImplementedError as e:
            funcs = self.dispatch_iter(*types)
            next(funcs)  # burn first
            for func in funcs:
                try:
                    return func(*args, **kwargs)
                except MDNotImplementedError:
                    pass

            raise NotImplementedError(
                "Matching functions for "
                f"{self.name}: <{str_signature(types)}> found, but none completed successfully",
            ) from e

    def __str__(self):
        return f"<dispatched {self.name}>"

    __repr__ = __str__

    def dispatch(self, *types):
        """Determine appropriate implementation for this type signature
        This method is internal.  Users should call this object as a function.
        Implementation resolution occurs within the ``__call__`` method.
        >>> # xdoctest: +SKIP
        >>> from multipledispatch import dispatch
        >>> @dispatch(int)
        ... def inc(x):
        ...     return x + 1
        >>> implementation = inc.dispatch(int)
        >>> implementation(3)
        4
        >>> print(inc.dispatch(float))
        None
        See Also:
          ``multipledispatch.conflict`` - module to determine resolution order
        """

        if types in self.funcs:
            return self.funcs[types]

        try:
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    def dispatch_iter(self, *types):
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):
                result = self.funcs[signature]
                yield result
            elif len(signature) and isvariadic(signature[-1]):
                if variadic_signature_matches(types, signature):
                    result = self.funcs[signature]
                    yield result

    @deprecated(
        "`resolve()` is deprecated, use `dispatch(*types)`", category=FutureWarning
    )
    def resolve(self, types):
        """Determine appropriate implementation for this type signature
        .. deprecated:: 0.4.4
            Use ``dispatch(*types)`` instead
        """
        return self.dispatch(*types)

    def __getstate__(self):
        return {"name": self.name, "funcs": self.funcs}

    def __setstate__(self, d):
        self.name = d["name"]
        self.funcs = d["funcs"]
        self._ordering = ordering(self.funcs)
        self._cache = {}

    @property
    def __doc__(self):
        docs = [f"Multiply dispatched method: {self.name}"]

        if self.doc:
            docs.append(self.doc)

        other = []
        for sig in self.ordering[::-1]:
            func = self.funcs[sig]
            if func.__doc__:
                s = f"Inputs: <{str_signature(sig)}>\n"
                s += "-" * len(s) + "\n"
                s += func.__doc__.strip()
                docs.append(s)
            else:
                other.append(str_signature(sig))

        if other:
            docs.append("Other signatures:\n    " + "\n    ".join(other))

        return "\n\n".join(docs)

    def _help(self, *args):
        return self.dispatch(*map(type, args)).__doc__

    def help(self, *args, **kwargs):
        """Print docstring for the function corresponding to inputs"""
        print(self._help(*args))

    def _source(self, *args):
        func = self.dispatch(*map(type, args))
        if not func:
            raise TypeError("No function found")
        return source(func)

    def source(self, *args, **kwargs):
        """Print source code for the function corresponding to inputs"""
        print(self._source(*args))


def source(func):
    s = f"File: {inspect.getsourcefile(func)}\n\n"
    s = s + inspect.getsource(func)
    return s


class MethodDispatcher(Dispatcher):
    """Dispatch methods based on type signature
    See Also:
        Dispatcher
    """

    __slots__ = ("obj", "cls")

    @classmethod
    def get_func_params(cls, func):
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return itl.islice(sig.parameters.values(), 1, None)

    def __get__(self, instance, owner):
        self.obj = instance
        self.cls = owner
        return self

    def __call__(self, *args, **kwargs):
        types = tuple([type(arg) for arg in args])
        func = self.dispatch(*types)
        if not func:
            raise NotImplementedError(
                f"Could not find signature for {self.name}: <{str_signature(types)}>"
            )
        return func(self.obj, *args, **kwargs)


def str_signature(sig):
    """String representation of type signature
    >>> str_signature((int, float))
    'int, float'
    """
    return ", ".join(cls.__name__ for cls in sig)


def warning_text(name, amb):
    """The text for ambiguity warnings"""
    text = f"\nAmbiguities exist in dispatched function {name}\n\n"
    text += "The following signatures may result in ambiguous behavior:\n"
    for pair in amb:
        text += "\t" + ", ".join("[" + str_signature(s) + "]" for s in pair) + "\n"
    text += "\n\nConsider making the following additions:\n\n"
    text += "\n\n".join(
        [
            "@dispatch(" + str_signature(super_signature(s)) + f")\ndef {name}(...)"
            for s in amb
        ]
    )
    return text
