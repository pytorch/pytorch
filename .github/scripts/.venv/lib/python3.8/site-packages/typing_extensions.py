import abc
import collections
import collections.abc
import operator
import sys
import types as _types
import typing

# After PEP 560, internal typing API was substantially reworked.
# This is especially important for Protocol class which uses internal APIs
# quite extensively.
PEP_560 = sys.version_info[:3] >= (3, 7, 0)

if PEP_560:
    GenericMeta = type
else:
    # 3.6
    from typing import GenericMeta, _type_vars  # noqa


# Please keep __all__ alphabetized within each category.
__all__ = [
    # Super-special typing primitives.
    'ClassVar',
    'Concatenate',
    'Final',
    'LiteralString',
    'ParamSpec',
    'Self',
    'Type',
    'TypeVarTuple',
    'Unpack',

    # ABCs (from collections.abc).
    'Awaitable',
    'AsyncIterator',
    'AsyncIterable',
    'Coroutine',
    'AsyncGenerator',
    'AsyncContextManager',
    'ChainMap',

    # Concrete collection types.
    'ContextManager',
    'Counter',
    'Deque',
    'DefaultDict',
    'OrderedDict',
    'TypedDict',

    # Structural checks, a.k.a. protocols.
    'SupportsIndex',

    # One-off things.
    'Annotated',
    'assert_never',
    'dataclass_transform',
    'final',
    'IntVar',
    'is_typeddict',
    'Literal',
    'NewType',
    'overload',
    'Protocol',
    'reveal_type',
    'runtime',
    'runtime_checkable',
    'Text',
    'TypeAlias',
    'TypeGuard',
    'TYPE_CHECKING',
    'Never',
    'NoReturn',
    'Required',
    'NotRequired',
]

if PEP_560:
    __all__.extend(["get_args", "get_origin", "get_type_hints"])

# The functions below are modified copies of typing internal helpers.
# They are needed by _ProtocolMeta and they provide support for PEP 646.


def _no_slots_copy(dct):
    dict_copy = dict(dct)
    if '__slots__' in dict_copy:
        for slot in dict_copy['__slots__']:
            dict_copy.pop(slot, None)
    return dict_copy


_marker = object()


def _check_generic(cls, parameters, elen=_marker):
    """Check correct count for parameters of a generic cls (internal helper).
    This gives a nice error message in case of count mismatch.
    """
    if not elen:
        raise TypeError(f"{cls} is not a generic class")
    if elen is _marker:
        if not hasattr(cls, "__parameters__") or not cls.__parameters__:
            raise TypeError(f"{cls} is not a generic class")
        elen = len(cls.__parameters__)
    alen = len(parameters)
    if alen != elen:
        if hasattr(cls, "__parameters__"):
            parameters = [p for p in cls.__parameters__ if not _is_unpack(p)]
            num_tv_tuples = sum(isinstance(p, TypeVarTuple) for p in parameters)
            if (num_tv_tuples > 0) and (alen >= elen - num_tv_tuples):
                return
        raise TypeError(f"Too {'many' if alen > elen else 'few'} parameters for {cls};"
                        f" actual {alen}, expected {elen}")


if sys.version_info >= (3, 10):
    def _should_collect_from_parameters(t):
        return isinstance(
            t, (typing._GenericAlias, _types.GenericAlias, _types.UnionType)
        )
elif sys.version_info >= (3, 9):
    def _should_collect_from_parameters(t):
        return isinstance(t, (typing._GenericAlias, _types.GenericAlias))
else:
    def _should_collect_from_parameters(t):
        return isinstance(t, typing._GenericAlias) and not t._special


def _collect_type_vars(types, typevar_types=None):
    """Collect all type variable contained in types in order of
    first appearance (lexicographic order). For example::

        _collect_type_vars((T, List[S, T])) == (T, S)
    """
    if typevar_types is None:
        typevar_types = typing.TypeVar
    tvars = []
    for t in types:
        if (
            isinstance(t, typevar_types) and
            t not in tvars and
            not _is_unpack(t)
        ):
            tvars.append(t)
        if _should_collect_from_parameters(t):
            tvars.extend([t for t in t.__parameters__ if t not in tvars])
    return tuple(tvars)


# 3.6.2+
if hasattr(typing, 'NoReturn'):
    NoReturn = typing.NoReturn
# 3.6.0-3.6.1
else:
    class _NoReturn(typing._FinalTypingBase, _root=True):
        """Special type indicating functions that never return.
        Example::

          from typing import NoReturn

          def stop() -> NoReturn:
              raise Exception('no way')

        This type is invalid in other positions, e.g., ``List[NoReturn]``
        will fail in static type checkers.
        """
        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError("NoReturn cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError("NoReturn cannot be used with issubclass().")

    NoReturn = _NoReturn(_root=True)

# Some unconstrained type variables.  These are used by the container types.
# (These are not for export.)
T = typing.TypeVar('T')  # Any type.
KT = typing.TypeVar('KT')  # Key type.
VT = typing.TypeVar('VT')  # Value type.
T_co = typing.TypeVar('T_co', covariant=True)  # Any type covariant containers.
T_contra = typing.TypeVar('T_contra', contravariant=True)  # Ditto contravariant.

ClassVar = typing.ClassVar

# On older versions of typing there is an internal class named "Final".
# 3.8+
if hasattr(typing, 'Final') and sys.version_info[:2] >= (3, 7):
    Final = typing.Final
# 3.7
elif sys.version_info[:2] >= (3, 7):
    class _FinalForm(typing._SpecialForm, _root=True):

        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            item = typing._type_check(parameters,
                                      f'{self._name} accepts only single type')
            return typing._GenericAlias(self, (item,))

    Final = _FinalForm('Final',
                       doc="""A special typing construct to indicate that a name
                       cannot be re-assigned or overridden in a subclass.
                       For example:

                           MAX_SIZE: Final = 9000
                           MAX_SIZE += 1  # Error reported by type checker

                           class Connection:
                               TIMEOUT: Final[int] = 10
                           class FastConnector(Connection):
                               TIMEOUT = 1  # Error reported by type checker

                       There is no runtime checking of these properties.""")
# 3.6
else:
    class _Final(typing._FinalTypingBase, _root=True):
        """A special typing construct to indicate that a name
        cannot be re-assigned or overridden in a subclass.
        For example:

            MAX_SIZE: Final = 9000
            MAX_SIZE += 1  # Error reported by type checker

            class Connection:
                TIMEOUT: Final[int] = 10
            class FastConnector(Connection):
                TIMEOUT = 1  # Error reported by type checker

        There is no runtime checking of these properties.
        """

        __slots__ = ('__type__',)

        def __init__(self, tp=None, **kwds):
            self.__type__ = tp

        def __getitem__(self, item):
            cls = type(self)
            if self.__type__ is None:
                return cls(typing._type_check(item,
                           f'{cls.__name__[1:]} accepts only single type.'),
                           _root=True)
            raise TypeError(f'{cls.__name__[1:]} cannot be further subscripted')

        def _eval_type(self, globalns, localns):
            new_tp = typing._eval_type(self.__type__, globalns, localns)
            if new_tp == self.__type__:
                return self
            return type(self)(new_tp, _root=True)

        def __repr__(self):
            r = super().__repr__()
            if self.__type__ is not None:
                r += f'[{typing._type_repr(self.__type__)}]'
            return r

        def __hash__(self):
            return hash((type(self).__name__, self.__type__))

        def __eq__(self, other):
            if not isinstance(other, _Final):
                return NotImplemented
            if self.__type__ is not None:
                return self.__type__ == other.__type__
            return self is other

    Final = _Final(_root=True)


if sys.version_info >= (3, 11):
    final = typing.final
else:
    # @final exists in 3.8+, but we backport it for all versions
    # before 3.11 to keep support for the __final__ attribute.
    # See https://bugs.python.org/issue46342
    def final(f):
        """This decorator can be used to indicate to type checkers that
        the decorated method cannot be overridden, and decorated class
        cannot be subclassed. For example:

            class Base:
                @final
                def done(self) -> None:
                    ...
            class Sub(Base):
                def done(self) -> None:  # Error reported by type checker
                    ...
            @final
            class Leaf:
                ...
            class Other(Leaf):  # Error reported by type checker
                ...

        There is no runtime checking of these properties. The decorator
        sets the ``__final__`` attribute to ``True`` on the decorated object
        to allow runtime introspection.
        """
        try:
            f.__final__ = True
        except (AttributeError, TypeError):
            # Skip the attribute silently if it is not writable.
            # AttributeError happens if the object has __slots__ or a
            # read-only property, TypeError if it's a builtin class.
            pass
        return f


def IntVar(name):
    return typing.TypeVar(name)


# 3.8+:
if hasattr(typing, 'Literal'):
    Literal = typing.Literal
# 3.7:
elif sys.version_info[:2] >= (3, 7):
    class _LiteralForm(typing._SpecialForm, _root=True):

        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            return typing._GenericAlias(self, parameters)

    Literal = _LiteralForm('Literal',
                           doc="""A type that can be used to indicate to type checkers
                           that the corresponding value has a value literally equivalent
                           to the provided parameter. For example:

                               var: Literal[4] = 4

                           The type checker understands that 'var' is literally equal to
                           the value 4 and no other value.

                           Literal[...] cannot be subclassed. There is no runtime
                           checking verifying that the parameter is actually a value
                           instead of a type.""")
# 3.6:
else:
    class _Literal(typing._FinalTypingBase, _root=True):
        """A type that can be used to indicate to type checkers that the
        corresponding value has a value literally equivalent to the
        provided parameter. For example:

            var: Literal[4] = 4

        The type checker understands that 'var' is literally equal to the
        value 4 and no other value.

        Literal[...] cannot be subclassed. There is no runtime checking
        verifying that the parameter is actually a value instead of a type.
        """

        __slots__ = ('__values__',)

        def __init__(self, values=None, **kwds):
            self.__values__ = values

        def __getitem__(self, values):
            cls = type(self)
            if self.__values__ is None:
                if not isinstance(values, tuple):
                    values = (values,)
                return cls(values, _root=True)
            raise TypeError(f'{cls.__name__[1:]} cannot be further subscripted')

        def _eval_type(self, globalns, localns):
            return self

        def __repr__(self):
            r = super().__repr__()
            if self.__values__ is not None:
                r += f'[{", ".join(map(typing._type_repr, self.__values__))}]'
            return r

        def __hash__(self):
            return hash((type(self).__name__, self.__values__))

        def __eq__(self, other):
            if not isinstance(other, _Literal):
                return NotImplemented
            if self.__values__ is not None:
                return self.__values__ == other.__values__
            return self is other

    Literal = _Literal(_root=True)


_overload_dummy = typing._overload_dummy  # noqa
overload = typing.overload


# This is not a real generic class.  Don't use outside annotations.
Type = typing.Type

# Various ABCs mimicking those in collections.abc.
# A few are simply re-exported for completeness.


class _ExtensionsGenericMeta(GenericMeta):
    def __subclasscheck__(self, subclass):
        """This mimics a more modern GenericMeta.__subclasscheck__() logic
        (that does not have problems with recursion) to work around interactions
        between collections, typing, and typing_extensions on older
        versions of Python, see https://github.com/python/typing/issues/501.
        """
        if self.__origin__ is not None:
            if sys._getframe(1).f_globals['__name__'] not in ['abc', 'functools']:
                raise TypeError("Parameterized generics cannot be used with class "
                                "or instance checks")
            return False
        if not self.__extra__:
            return super().__subclasscheck__(subclass)
        res = self.__extra__.__subclasshook__(subclass)
        if res is not NotImplemented:
            return res
        if self.__extra__ in subclass.__mro__:
            return True
        for scls in self.__extra__.__subclasses__():
            if isinstance(scls, GenericMeta):
                continue
            if issubclass(subclass, scls):
                return True
        return False


Awaitable = typing.Awaitable
Coroutine = typing.Coroutine
AsyncIterable = typing.AsyncIterable
AsyncIterator = typing.AsyncIterator

# 3.6.1+
if hasattr(typing, 'Deque'):
    Deque = typing.Deque
# 3.6.0
else:
    class Deque(collections.deque, typing.MutableSequence[T],
                metaclass=_ExtensionsGenericMeta,
                extra=collections.deque):
        __slots__ = ()

        def __new__(cls, *args, **kwds):
            if cls._gorg is Deque:
                return collections.deque(*args, **kwds)
            return typing._generic_new(collections.deque, cls, *args, **kwds)

ContextManager = typing.ContextManager
# 3.6.2+
if hasattr(typing, 'AsyncContextManager'):
    AsyncContextManager = typing.AsyncContextManager
# 3.6.0-3.6.1
else:
    from _collections_abc import _check_methods as _check_methods_in_mro  # noqa

    class AsyncContextManager(typing.Generic[T_co]):
        __slots__ = ()

        async def __aenter__(self):
            return self

        @abc.abstractmethod
        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        @classmethod
        def __subclasshook__(cls, C):
            if cls is AsyncContextManager:
                return _check_methods_in_mro(C, "__aenter__", "__aexit__")
            return NotImplemented

DefaultDict = typing.DefaultDict

# 3.7.2+
if hasattr(typing, 'OrderedDict'):
    OrderedDict = typing.OrderedDict
# 3.7.0-3.7.2
elif (3, 7, 0) <= sys.version_info[:3] < (3, 7, 2):
    OrderedDict = typing._alias(collections.OrderedDict, (KT, VT))
# 3.6
else:
    class OrderedDict(collections.OrderedDict, typing.MutableMapping[KT, VT],
                      metaclass=_ExtensionsGenericMeta,
                      extra=collections.OrderedDict):

        __slots__ = ()

        def __new__(cls, *args, **kwds):
            if cls._gorg is OrderedDict:
                return collections.OrderedDict(*args, **kwds)
            return typing._generic_new(collections.OrderedDict, cls, *args, **kwds)

# 3.6.2+
if hasattr(typing, 'Counter'):
    Counter = typing.Counter
# 3.6.0-3.6.1
else:
    class Counter(collections.Counter,
                  typing.Dict[T, int],
                  metaclass=_ExtensionsGenericMeta, extra=collections.Counter):

        __slots__ = ()

        def __new__(cls, *args, **kwds):
            if cls._gorg is Counter:
                return collections.Counter(*args, **kwds)
            return typing._generic_new(collections.Counter, cls, *args, **kwds)

# 3.6.1+
if hasattr(typing, 'ChainMap'):
    ChainMap = typing.ChainMap
elif hasattr(collections, 'ChainMap'):
    class ChainMap(collections.ChainMap, typing.MutableMapping[KT, VT],
                   metaclass=_ExtensionsGenericMeta,
                   extra=collections.ChainMap):

        __slots__ = ()

        def __new__(cls, *args, **kwds):
            if cls._gorg is ChainMap:
                return collections.ChainMap(*args, **kwds)
            return typing._generic_new(collections.ChainMap, cls, *args, **kwds)

# 3.6.1+
if hasattr(typing, 'AsyncGenerator'):
    AsyncGenerator = typing.AsyncGenerator
# 3.6.0
else:
    class AsyncGenerator(AsyncIterator[T_co], typing.Generic[T_co, T_contra],
                         metaclass=_ExtensionsGenericMeta,
                         extra=collections.abc.AsyncGenerator):
        __slots__ = ()

NewType = typing.NewType
Text = typing.Text
TYPE_CHECKING = typing.TYPE_CHECKING


def _gorg(cls):
    """This function exists for compatibility with old typing versions."""
    assert isinstance(cls, GenericMeta)
    if hasattr(cls, '_gorg'):
        return cls._gorg
    while cls.__origin__ is not None:
        cls = cls.__origin__
    return cls


_PROTO_WHITELIST = ['Callable', 'Awaitable',
                    'Iterable', 'Iterator', 'AsyncIterable', 'AsyncIterator',
                    'Hashable', 'Sized', 'Container', 'Collection', 'Reversible',
                    'ContextManager', 'AsyncContextManager']


def _get_protocol_attrs(cls):
    attrs = set()
    for base in cls.__mro__[:-1]:  # without object
        if base.__name__ in ('Protocol', 'Generic'):
            continue
        annotations = getattr(base, '__annotations__', {})
        for attr in list(base.__dict__.keys()) + list(annotations.keys()):
            if (not attr.startswith('_abc_') and attr not in (
                    '__abstractmethods__', '__annotations__', '__weakref__',
                    '_is_protocol', '_is_runtime_protocol', '__dict__',
                    '__args__', '__slots__',
                    '__next_in_mro__', '__parameters__', '__origin__',
                    '__orig_bases__', '__extra__', '__tree_hash__',
                    '__doc__', '__subclasshook__', '__init__', '__new__',
                    '__module__', '_MutableMapping__marker', '_gorg')):
                attrs.add(attr)
    return attrs


def _is_callable_members_only(cls):
    return all(callable(getattr(cls, attr, None)) for attr in _get_protocol_attrs(cls))


# 3.8+
if hasattr(typing, 'Protocol'):
    Protocol = typing.Protocol
# 3.7
elif PEP_560:

    def _no_init(self, *args, **kwargs):
        if type(self)._is_protocol:
            raise TypeError('Protocols cannot be instantiated')

    class _ProtocolMeta(abc.ABCMeta):
        # This metaclass is a bit unfortunate and exists only because of the lack
        # of __instancehook__.
        def __instancecheck__(cls, instance):
            # We need this method for situations where attributes are
            # assigned in __init__.
            if ((not getattr(cls, '_is_protocol', False) or
                 _is_callable_members_only(cls)) and
                    issubclass(instance.__class__, cls)):
                return True
            if cls._is_protocol:
                if all(hasattr(instance, attr) and
                       (not callable(getattr(cls, attr, None)) or
                        getattr(instance, attr) is not None)
                       for attr in _get_protocol_attrs(cls)):
                    return True
            return super().__instancecheck__(instance)

    class Protocol(metaclass=_ProtocolMeta):
        # There is quite a lot of overlapping code with typing.Generic.
        # Unfortunately it is hard to avoid this while these live in two different
        # modules. The duplicated code will be removed when Protocol is moved to typing.
        """Base class for protocol classes. Protocol classes are defined as::

            class Proto(Protocol):
                def meth(self) -> int:
                    ...

        Such classes are primarily used with static type checkers that recognize
        structural subtyping (static duck-typing), for example::

            class C:
                def meth(self) -> int:
                    return 0

            def func(x: Proto) -> int:
                return x.meth()

            func(C())  # Passes static type check

        See PEP 544 for details. Protocol classes decorated with
        @typing_extensions.runtime act as simple-minded runtime protocol that checks
        only the presence of given attributes, ignoring their type signatures.

        Protocol classes can be generic, they are defined as::

            class GenProto(Protocol[T]):
                def meth(self) -> T:
                    ...
        """
        __slots__ = ()
        _is_protocol = True

        def __new__(cls, *args, **kwds):
            if cls is Protocol:
                raise TypeError("Type Protocol cannot be instantiated; "
                                "it can only be used as a base class")
            return super().__new__(cls)

        @typing._tp_cache
        def __class_getitem__(cls, params):
            if not isinstance(params, tuple):
                params = (params,)
            if not params and cls is not typing.Tuple:
                raise TypeError(
                    f"Parameter list to {cls.__qualname__}[...] cannot be empty")
            msg = "Parameters to generic types must be types."
            params = tuple(typing._type_check(p, msg) for p in params)  # noqa
            if cls is Protocol:
                # Generic can only be subscripted with unique type variables.
                if not all(isinstance(p, typing.TypeVar) for p in params):
                    i = 0
                    while isinstance(params[i], typing.TypeVar):
                        i += 1
                    raise TypeError(
                        "Parameters to Protocol[...] must all be type variables."
                        f" Parameter {i + 1} is {params[i]}")
                if len(set(params)) != len(params):
                    raise TypeError(
                        "Parameters to Protocol[...] must all be unique")
            else:
                # Subscripting a regular Generic subclass.
                _check_generic(cls, params, len(cls.__parameters__))
            return typing._GenericAlias(cls, params)

        def __init_subclass__(cls, *args, **kwargs):
            tvars = []
            if '__orig_bases__' in cls.__dict__:
                error = typing.Generic in cls.__orig_bases__
            else:
                error = typing.Generic in cls.__bases__
            if error:
                raise TypeError("Cannot inherit from plain Generic")
            if '__orig_bases__' in cls.__dict__:
                tvars = typing._collect_type_vars(cls.__orig_bases__)
                # Look for Generic[T1, ..., Tn] or Protocol[T1, ..., Tn].
                # If found, tvars must be a subset of it.
                # If not found, tvars is it.
                # Also check for and reject plain Generic,
                # and reject multiple Generic[...] and/or Protocol[...].
                gvars = None
                for base in cls.__orig_bases__:
                    if (isinstance(base, typing._GenericAlias) and
                            base.__origin__ in (typing.Generic, Protocol)):
                        # for error messages
                        the_base = base.__origin__.__name__
                        if gvars is not None:
                            raise TypeError(
                                "Cannot inherit from Generic[...]"
                                " and/or Protocol[...] multiple types.")
                        gvars = base.__parameters__
                if gvars is None:
                    gvars = tvars
                else:
                    tvarset = set(tvars)
                    gvarset = set(gvars)
                    if not tvarset <= gvarset:
                        s_vars = ', '.join(str(t) for t in tvars if t not in gvarset)
                        s_args = ', '.join(str(g) for g in gvars)
                        raise TypeError(f"Some type variables ({s_vars}) are"
                                        f" not listed in {the_base}[{s_args}]")
                    tvars = gvars
            cls.__parameters__ = tuple(tvars)

            # Determine if this is a protocol or a concrete subclass.
            if not cls.__dict__.get('_is_protocol', None):
                cls._is_protocol = any(b is Protocol for b in cls.__bases__)

            # Set (or override) the protocol subclass hook.
            def _proto_hook(other):
                if not cls.__dict__.get('_is_protocol', None):
                    return NotImplemented
                if not getattr(cls, '_is_runtime_protocol', False):
                    if sys._getframe(2).f_globals['__name__'] in ['abc', 'functools']:
                        return NotImplemented
                    raise TypeError("Instance and class checks can only be used with"
                                    " @runtime protocols")
                if not _is_callable_members_only(cls):
                    if sys._getframe(2).f_globals['__name__'] in ['abc', 'functools']:
                        return NotImplemented
                    raise TypeError("Protocols with non-method members"
                                    " don't support issubclass()")
                if not isinstance(other, type):
                    # Same error as for issubclass(1, int)
                    raise TypeError('issubclass() arg 1 must be a class')
                for attr in _get_protocol_attrs(cls):
                    for base in other.__mro__:
                        if attr in base.__dict__:
                            if base.__dict__[attr] is None:
                                return NotImplemented
                            break
                        annotations = getattr(base, '__annotations__', {})
                        if (isinstance(annotations, typing.Mapping) and
                                attr in annotations and
                                isinstance(other, _ProtocolMeta) and
                                other._is_protocol):
                            break
                    else:
                        return NotImplemented
                return True
            if '__subclasshook__' not in cls.__dict__:
                cls.__subclasshook__ = _proto_hook

            # We have nothing more to do for non-protocols.
            if not cls._is_protocol:
                return

            # Check consistency of bases.
            for base in cls.__bases__:
                if not (base in (object, typing.Generic) or
                        base.__module__ == 'collections.abc' and
                        base.__name__ in _PROTO_WHITELIST or
                        isinstance(base, _ProtocolMeta) and base._is_protocol):
                    raise TypeError('Protocols can only inherit from other'
                                    f' protocols, got {repr(base)}')
            cls.__init__ = _no_init
# 3.6
else:
    from typing import _next_in_mro, _type_check  # noqa

    def _no_init(self, *args, **kwargs):
        if type(self)._is_protocol:
            raise TypeError('Protocols cannot be instantiated')

    class _ProtocolMeta(GenericMeta):
        """Internal metaclass for Protocol.

        This exists so Protocol classes can be generic without deriving
        from Generic.
        """
        def __new__(cls, name, bases, namespace,
                    tvars=None, args=None, origin=None, extra=None, orig_bases=None):
            # This is just a version copied from GenericMeta.__new__ that
            # includes "Protocol" special treatment. (Comments removed for brevity.)
            assert extra is None  # Protocols should not have extra
            if tvars is not None:
                assert origin is not None
                assert all(isinstance(t, typing.TypeVar) for t in tvars), tvars
            else:
                tvars = _type_vars(bases)
                gvars = None
                for base in bases:
                    if base is typing.Generic:
                        raise TypeError("Cannot inherit from plain Generic")
                    if (isinstance(base, GenericMeta) and
                            base.__origin__ in (typing.Generic, Protocol)):
                        if gvars is not None:
                            raise TypeError(
                                "Cannot inherit from Generic[...] or"
                                " Protocol[...] multiple times.")
                        gvars = base.__parameters__
                if gvars is None:
                    gvars = tvars
                else:
                    tvarset = set(tvars)
                    gvarset = set(gvars)
                    if not tvarset <= gvarset:
                        s_vars = ", ".join(str(t) for t in tvars if t not in gvarset)
                        s_args = ", ".join(str(g) for g in gvars)
                        cls_name = "Generic" if any(b.__origin__ is typing.Generic
                                                    for b in bases) else "Protocol"
                        raise TypeError(f"Some type variables ({s_vars}) are"
                                        f" not listed in {cls_name}[{s_args}]")
                    tvars = gvars

            initial_bases = bases
            if (extra is not None and type(extra) is abc.ABCMeta and
                    extra not in bases):
                bases = (extra,) + bases
            bases = tuple(_gorg(b) if isinstance(b, GenericMeta) else b
                          for b in bases)
            if any(isinstance(b, GenericMeta) and b is not typing.Generic for b in bases):
                bases = tuple(b for b in bases if b is not typing.Generic)
            namespace.update({'__origin__': origin, '__extra__': extra})
            self = super(GenericMeta, cls).__new__(cls, name, bases, namespace,
                                                   _root=True)
            super(GenericMeta, self).__setattr__('_gorg',
                                                 self if not origin else
                                                 _gorg(origin))
            self.__parameters__ = tvars
            self.__args__ = tuple(... if a is typing._TypingEllipsis else
                                  () if a is typing._TypingEmpty else
                                  a for a in args) if args else None
            self.__next_in_mro__ = _next_in_mro(self)
            if orig_bases is None:
                self.__orig_bases__ = initial_bases
            elif origin is not None:
                self._abc_registry = origin._abc_registry
                self._abc_cache = origin._abc_cache
            if hasattr(self, '_subs_tree'):
                self.__tree_hash__ = (hash(self._subs_tree()) if origin else
                                      super(GenericMeta, self).__hash__())
            return self

        def __init__(cls, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not cls.__dict__.get('_is_protocol', None):
                cls._is_protocol = any(b is Protocol or
                                       isinstance(b, _ProtocolMeta) and
                                       b.__origin__ is Protocol
                                       for b in cls.__bases__)
            if cls._is_protocol:
                for base in cls.__mro__[1:]:
                    if not (base in (object, typing.Generic) or
                            base.__module__ == 'collections.abc' and
                            base.__name__ in _PROTO_WHITELIST or
                            isinstance(base, typing.TypingMeta) and base._is_protocol or
                            isinstance(base, GenericMeta) and
                            base.__origin__ is typing.Generic):
                        raise TypeError(f'Protocols can only inherit from other'
                                        f' protocols, got {repr(base)}')

                cls.__init__ = _no_init

            def _proto_hook(other):
                if not cls.__dict__.get('_is_protocol', None):
                    return NotImplemented
                if not isinstance(other, type):
                    # Same error as for issubclass(1, int)
                    raise TypeError('issubclass() arg 1 must be a class')
                for attr in _get_protocol_attrs(cls):
                    for base in other.__mro__:
                        if attr in base.__dict__:
                            if base.__dict__[attr] is None:
                                return NotImplemented
                            break
                        annotations = getattr(base, '__annotations__', {})
                        if (isinstance(annotations, typing.Mapping) and
                                attr in annotations and
                                isinstance(other, _ProtocolMeta) and
                                other._is_protocol):
                            break
                    else:
                        return NotImplemented
                return True
            if '__subclasshook__' not in cls.__dict__:
                cls.__subclasshook__ = _proto_hook

        def __instancecheck__(self, instance):
            # We need this method for situations where attributes are
            # assigned in __init__.
            if ((not getattr(self, '_is_protocol', False) or
                    _is_callable_members_only(self)) and
                    issubclass(instance.__class__, self)):
                return True
            if self._is_protocol:
                if all(hasattr(instance, attr) and
                        (not callable(getattr(self, attr, None)) or
                         getattr(instance, attr) is not None)
                        for attr in _get_protocol_attrs(self)):
                    return True
            return super(GenericMeta, self).__instancecheck__(instance)

        def __subclasscheck__(self, cls):
            if self.__origin__ is not None:
                if sys._getframe(1).f_globals['__name__'] not in ['abc', 'functools']:
                    raise TypeError("Parameterized generics cannot be used with class "
                                    "or instance checks")
                return False
            if (self.__dict__.get('_is_protocol', None) and
                    not self.__dict__.get('_is_runtime_protocol', None)):
                if sys._getframe(1).f_globals['__name__'] in ['abc',
                                                              'functools',
                                                              'typing']:
                    return False
                raise TypeError("Instance and class checks can only be used with"
                                " @runtime protocols")
            if (self.__dict__.get('_is_runtime_protocol', None) and
                    not _is_callable_members_only(self)):
                if sys._getframe(1).f_globals['__name__'] in ['abc',
                                                              'functools',
                                                              'typing']:
                    return super(GenericMeta, self).__subclasscheck__(cls)
                raise TypeError("Protocols with non-method members"
                                " don't support issubclass()")
            return super(GenericMeta, self).__subclasscheck__(cls)

        @typing._tp_cache
        def __getitem__(self, params):
            # We also need to copy this from GenericMeta.__getitem__ to get
            # special treatment of "Protocol". (Comments removed for brevity.)
            if not isinstance(params, tuple):
                params = (params,)
            if not params and _gorg(self) is not typing.Tuple:
                raise TypeError(
                    f"Parameter list to {self.__qualname__}[...] cannot be empty")
            msg = "Parameters to generic types must be types."
            params = tuple(_type_check(p, msg) for p in params)
            if self in (typing.Generic, Protocol):
                if not all(isinstance(p, typing.TypeVar) for p in params):
                    raise TypeError(
                        f"Parameters to {repr(self)}[...] must all be type variables")
                if len(set(params)) != len(params):
                    raise TypeError(
                        f"Parameters to {repr(self)}[...] must all be unique")
                tvars = params
                args = params
            elif self in (typing.Tuple, typing.Callable):
                tvars = _type_vars(params)
                args = params
            elif self.__origin__ in (typing.Generic, Protocol):
                raise TypeError(f"Cannot subscript already-subscripted {repr(self)}")
            else:
                _check_generic(self, params, len(self.__parameters__))
                tvars = _type_vars(params)
                args = params

            prepend = (self,) if self.__origin__ is None else ()
            return self.__class__(self.__name__,
                                  prepend + self.__bases__,
                                  _no_slots_copy(self.__dict__),
                                  tvars=tvars,
                                  args=args,
                                  origin=self,
                                  extra=self.__extra__,
                                  orig_bases=self.__orig_bases__)

    class Protocol(metaclass=_ProtocolMeta):
        """Base class for protocol classes. Protocol classes are defined as::

          class Proto(Protocol):
              def meth(self) -> int:
                  ...

        Such classes are primarily used with static type checkers that recognize
        structural subtyping (static duck-typing), for example::

          class C:
              def meth(self) -> int:
                  return 0

          def func(x: Proto) -> int:
              return x.meth()

          func(C())  # Passes static type check

        See PEP 544 for details. Protocol classes decorated with
        @typing_extensions.runtime act as simple-minded runtime protocol that checks
        only the presence of given attributes, ignoring their type signatures.

        Protocol classes can be generic, they are defined as::

          class GenProto(Protocol[T]):
              def meth(self) -> T:
                  ...
        """
        __slots__ = ()
        _is_protocol = True

        def __new__(cls, *args, **kwds):
            if _gorg(cls) is Protocol:
                raise TypeError("Type Protocol cannot be instantiated; "
                                "it can be used only as a base class")
            return typing._generic_new(cls.__next_in_mro__, cls, *args, **kwds)


# 3.8+
if hasattr(typing, 'runtime_checkable'):
    runtime_checkable = typing.runtime_checkable
# 3.6-3.7
else:
    def runtime_checkable(cls):
        """Mark a protocol class as a runtime protocol, so that it
        can be used with isinstance() and issubclass(). Raise TypeError
        if applied to a non-protocol class.

        This allows a simple-minded structural check very similar to the
        one-offs in collections.abc such as Hashable.
        """
        if not isinstance(cls, _ProtocolMeta) or not cls._is_protocol:
            raise TypeError('@runtime_checkable can be only applied to protocol classes,'
                            f' got {cls!r}')
        cls._is_runtime_protocol = True
        return cls


# Exists for backwards compatibility.
runtime = runtime_checkable


# 3.8+
if hasattr(typing, 'SupportsIndex'):
    SupportsIndex = typing.SupportsIndex
# 3.6-3.7
else:
    @runtime_checkable
    class SupportsIndex(Protocol):
        __slots__ = ()

        @abc.abstractmethod
        def __index__(self) -> int:
            pass


if hasattr(typing, "Required"):
    # The standard library TypedDict in Python 3.8 does not store runtime information
    # about which (if any) keys are optional.  See https://bugs.python.org/issue38834
    # The standard library TypedDict in Python 3.9.0/1 does not honour the "total"
    # keyword with old-style TypedDict().  See https://bugs.python.org/issue42059
    # The standard library TypedDict below Python 3.11 does not store runtime
    # information about optional and required keys when using Required or NotRequired.
    TypedDict = typing.TypedDict
    _TypedDictMeta = typing._TypedDictMeta
    is_typeddict = typing.is_typeddict
else:
    def _check_fails(cls, other):
        try:
            if sys._getframe(1).f_globals['__name__'] not in ['abc',
                                                              'functools',
                                                              'typing']:
                # Typed dicts are only for static structural subtyping.
                raise TypeError('TypedDict does not support instance and class checks')
        except (AttributeError, ValueError):
            pass
        return False

    def _dict_new(*args, **kwargs):
        if not args:
            raise TypeError('TypedDict.__new__(): not enough arguments')
        _, args = args[0], args[1:]  # allow the "cls" keyword be passed
        return dict(*args, **kwargs)

    _dict_new.__text_signature__ = '($cls, _typename, _fields=None, /, **kwargs)'

    def _typeddict_new(*args, total=True, **kwargs):
        if not args:
            raise TypeError('TypedDict.__new__(): not enough arguments')
        _, args = args[0], args[1:]  # allow the "cls" keyword be passed
        if args:
            typename, args = args[0], args[1:]  # allow the "_typename" keyword be passed
        elif '_typename' in kwargs:
            typename = kwargs.pop('_typename')
            import warnings
            warnings.warn("Passing '_typename' as keyword argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        else:
            raise TypeError("TypedDict.__new__() missing 1 required positional "
                            "argument: '_typename'")
        if args:
            try:
                fields, = args  # allow the "_fields" keyword be passed
            except ValueError:
                raise TypeError('TypedDict.__new__() takes from 2 to 3 '
                                f'positional arguments but {len(args) + 2} '
                                'were given')
        elif '_fields' in kwargs and len(kwargs) == 1:
            fields = kwargs.pop('_fields')
            import warnings
            warnings.warn("Passing '_fields' as keyword argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        else:
            fields = None

        if fields is None:
            fields = kwargs
        elif kwargs:
            raise TypeError("TypedDict takes either a dict or keyword arguments,"
                            " but not both")

        ns = {'__annotations__': dict(fields)}
        try:
            # Setting correct module is necessary to make typed dict classes pickleable.
            ns['__module__'] = sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass

        return _TypedDictMeta(typename, (), ns, total=total)

    _typeddict_new.__text_signature__ = ('($cls, _typename, _fields=None,'
                                         ' /, *, total=True, **kwargs)')

    class _TypedDictMeta(type):
        def __init__(cls, name, bases, ns, total=True):
            super().__init__(name, bases, ns)

        def __new__(cls, name, bases, ns, total=True):
            # Create new typed dict class object.
            # This method is called directly when TypedDict is subclassed,
            # or via _typeddict_new when TypedDict is instantiated. This way
            # TypedDict supports all three syntaxes described in its docstring.
            # Subclasses and instances of TypedDict return actual dictionaries
            # via _dict_new.
            ns['__new__'] = _typeddict_new if name == 'TypedDict' else _dict_new
            tp_dict = super().__new__(cls, name, (dict,), ns)

            annotations = {}
            own_annotations = ns.get('__annotations__', {})
            msg = "TypedDict('Name', {f0: t0, f1: t1, ...}); each t must be a type"
            own_annotations = {
                n: typing._type_check(tp, msg) for n, tp in own_annotations.items()
            }
            required_keys = set()
            optional_keys = set()

            for base in bases:
                annotations.update(base.__dict__.get('__annotations__', {}))
                required_keys.update(base.__dict__.get('__required_keys__', ()))
                optional_keys.update(base.__dict__.get('__optional_keys__', ()))

            annotations.update(own_annotations)
            if PEP_560:
                for annotation_key, annotation_type in own_annotations.items():
                    annotation_origin = get_origin(annotation_type)
                    if annotation_origin is Annotated:
                        annotation_args = get_args(annotation_type)
                        if annotation_args:
                            annotation_type = annotation_args[0]
                            annotation_origin = get_origin(annotation_type)

                    if annotation_origin is Required:
                        required_keys.add(annotation_key)
                    elif annotation_origin is NotRequired:
                        optional_keys.add(annotation_key)
                    elif total:
                        required_keys.add(annotation_key)
                    else:
                        optional_keys.add(annotation_key)
            else:
                own_annotation_keys = set(own_annotations.keys())
                if total:
                    required_keys.update(own_annotation_keys)
                else:
                    optional_keys.update(own_annotation_keys)

            tp_dict.__annotations__ = annotations
            tp_dict.__required_keys__ = frozenset(required_keys)
            tp_dict.__optional_keys__ = frozenset(optional_keys)
            if not hasattr(tp_dict, '__total__'):
                tp_dict.__total__ = total
            return tp_dict

        __instancecheck__ = __subclasscheck__ = _check_fails

    TypedDict = _TypedDictMeta('TypedDict', (dict,), {})
    TypedDict.__module__ = __name__
    TypedDict.__doc__ = \
        """A simple typed name space. At runtime it is equivalent to a plain dict.

        TypedDict creates a dictionary type that expects all of its
        instances to have a certain set of keys, with each key
        associated with a value of a consistent type. This expectation
        is not checked at runtime but is only enforced by type checkers.
        Usage::

            class Point2D(TypedDict):
                x: int
                y: int
                label: str

            a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
            b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check

            assert Point2D(x=1, y=2, label='first') == dict(x=1, y=2, label='first')

        The type info can be accessed via the Point2D.__annotations__ dict, and
        the Point2D.__required_keys__ and Point2D.__optional_keys__ frozensets.
        TypedDict supports two additional equivalent forms::

            Point2D = TypedDict('Point2D', x=int, y=int, label=str)
            Point2D = TypedDict('Point2D', {'x': int, 'y': int, 'label': str})

        The class syntax is only supported in Python 3.6+, while two other
        syntax forms work for Python 2.7 and 3.2+
        """

    if hasattr(typing, "_TypedDictMeta"):
        _TYPEDDICT_TYPES = (typing._TypedDictMeta, _TypedDictMeta)
    else:
        _TYPEDDICT_TYPES = (_TypedDictMeta,)

    def is_typeddict(tp):
        """Check if an annotation is a TypedDict class

        For example::
            class Film(TypedDict):
                title: str
                year: int

            is_typeddict(Film)  # => True
            is_typeddict(Union[list, str])  # => False
        """
        return isinstance(tp, tuple(_TYPEDDICT_TYPES))

if hasattr(typing, "Required"):
    get_type_hints = typing.get_type_hints
elif PEP_560:
    import functools
    import types

    # replaces _strip_annotations()
    def _strip_extras(t):
        """Strips Annotated, Required and NotRequired from a given type."""
        if isinstance(t, _AnnotatedAlias):
            return _strip_extras(t.__origin__)
        if hasattr(t, "__origin__") and t.__origin__ in (Required, NotRequired):
            return _strip_extras(t.__args__[0])
        if isinstance(t, typing._GenericAlias):
            stripped_args = tuple(_strip_extras(a) for a in t.__args__)
            if stripped_args == t.__args__:
                return t
            return t.copy_with(stripped_args)
        if hasattr(types, "GenericAlias") and isinstance(t, types.GenericAlias):
            stripped_args = tuple(_strip_extras(a) for a in t.__args__)
            if stripped_args == t.__args__:
                return t
            return types.GenericAlias(t.__origin__, stripped_args)
        if hasattr(types, "UnionType") and isinstance(t, types.UnionType):
            stripped_args = tuple(_strip_extras(a) for a in t.__args__)
            if stripped_args == t.__args__:
                return t
            return functools.reduce(operator.or_, stripped_args)

        return t

    def get_type_hints(obj, globalns=None, localns=None, include_extras=False):
        """Return type hints for an object.

        This is often the same as obj.__annotations__, but it handles
        forward references encoded as string literals, adds Optional[t] if a
        default value equal to None is set and recursively replaces all
        'Annotated[T, ...]', 'Required[T]' or 'NotRequired[T]' with 'T'
        (unless 'include_extras=True').

        The argument may be a module, class, method, or function. The annotations
        are returned as a dictionary. For classes, annotations include also
        inherited members.

        TypeError is raised if the argument is not of a type that can contain
        annotations, and an empty dictionary is returned if no annotations are
        present.

        BEWARE -- the behavior of globalns and localns is counterintuitive
        (unless you are familiar with how eval() and exec() work).  The
        search order is locals first, then globals.

        - If no dict arguments are passed, an attempt is made to use the
          globals from obj (or the respective module's globals for classes),
          and these are also used as the locals.  If the object does not appear
          to have globals, an empty dictionary is used.

        - If one dict argument is passed, it is used for both globals and
          locals.

        - If two dict arguments are passed, they specify globals and
          locals, respectively.
        """
        if hasattr(typing, "Annotated"):
            hint = typing.get_type_hints(
                obj, globalns=globalns, localns=localns, include_extras=True
            )
        else:
            hint = typing.get_type_hints(obj, globalns=globalns, localns=localns)
        if include_extras:
            return hint
        return {k: _strip_extras(t) for k, t in hint.items()}


# Python 3.9+ has PEP 593 (Annotated)
if hasattr(typing, 'Annotated'):
    Annotated = typing.Annotated
    # Not exported and not a public API, but needed for get_origin() and get_args()
    # to work.
    _AnnotatedAlias = typing._AnnotatedAlias
# 3.7-3.8
elif PEP_560:
    class _AnnotatedAlias(typing._GenericAlias, _root=True):
        """Runtime representation of an annotated type.

        At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
        with extra annotations. The alias behaves like a normal typing alias,
        instantiating is the same as instantiating the underlying type, binding
        it to types is also the same.
        """
        def __init__(self, origin, metadata):
            if isinstance(origin, _AnnotatedAlias):
                metadata = origin.__metadata__ + metadata
                origin = origin.__origin__
            super().__init__(origin, origin)
            self.__metadata__ = metadata

        def copy_with(self, params):
            assert len(params) == 1
            new_type = params[0]
            return _AnnotatedAlias(new_type, self.__metadata__)

        def __repr__(self):
            return (f"typing_extensions.Annotated[{typing._type_repr(self.__origin__)}, "
                    f"{', '.join(repr(a) for a in self.__metadata__)}]")

        def __reduce__(self):
            return operator.getitem, (
                Annotated, (self.__origin__,) + self.__metadata__
            )

        def __eq__(self, other):
            if not isinstance(other, _AnnotatedAlias):
                return NotImplemented
            if self.__origin__ != other.__origin__:
                return False
            return self.__metadata__ == other.__metadata__

        def __hash__(self):
            return hash((self.__origin__, self.__metadata__))

    class Annotated:
        """Add context specific metadata to a type.

        Example: Annotated[int, runtime_check.Unsigned] indicates to the
        hypothetical runtime_check module that this type is an unsigned int.
        Every other consumer of this type can ignore this metadata and treat
        this type as int.

        The first argument to Annotated must be a valid type (and will be in
        the __origin__ field), the remaining arguments are kept as a tuple in
        the __extra__ field.

        Details:

        - It's an error to call `Annotated` with less than two arguments.
        - Nested Annotated are flattened::

            Annotated[Annotated[T, Ann1, Ann2], Ann3] == Annotated[T, Ann1, Ann2, Ann3]

        - Instantiating an annotated type is equivalent to instantiating the
        underlying type::

            Annotated[C, Ann1](5) == C(5)

        - Annotated can be used as a generic type alias::

            Optimized = Annotated[T, runtime.Optimize()]
            Optimized[int] == Annotated[int, runtime.Optimize()]

            OptimizedList = Annotated[List[T], runtime.Optimize()]
            OptimizedList[int] == Annotated[List[int], runtime.Optimize()]
        """

        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            raise TypeError("Type Annotated cannot be instantiated.")

        @typing._tp_cache
        def __class_getitem__(cls, params):
            if not isinstance(params, tuple) or len(params) < 2:
                raise TypeError("Annotated[...] should be used "
                                "with at least two arguments (a type and an "
                                "annotation).")
            allowed_special_forms = (ClassVar, Final)
            if get_origin(params[0]) in allowed_special_forms:
                origin = params[0]
            else:
                msg = "Annotated[t, ...]: t must be a type."
                origin = typing._type_check(params[0], msg)
            metadata = tuple(params[1:])
            return _AnnotatedAlias(origin, metadata)

        def __init_subclass__(cls, *args, **kwargs):
            raise TypeError(
                f"Cannot subclass {cls.__module__}.Annotated"
            )
# 3.6
else:

    def _is_dunder(name):
        """Returns True if name is a __dunder_variable_name__."""
        return len(name) > 4 and name.startswith('__') and name.endswith('__')

    # Prior to Python 3.7 types did not have `copy_with`. A lot of the equality
    # checks, argument expansion etc. are done on the _subs_tre. As a result we
    # can't provide a get_type_hints function that strips out annotations.

    class AnnotatedMeta(typing.GenericMeta):
        """Metaclass for Annotated"""

        def __new__(cls, name, bases, namespace, **kwargs):
            if any(b is not object for b in bases):
                raise TypeError("Cannot subclass " + str(Annotated))
            return super().__new__(cls, name, bases, namespace, **kwargs)

        @property
        def __metadata__(self):
            return self._subs_tree()[2]

        def _tree_repr(self, tree):
            cls, origin, metadata = tree
            if not isinstance(origin, tuple):
                tp_repr = typing._type_repr(origin)
            else:
                tp_repr = origin[0]._tree_repr(origin)
            metadata_reprs = ", ".join(repr(arg) for arg in metadata)
            return f'{cls}[{tp_repr}, {metadata_reprs}]'

        def _subs_tree(self, tvars=None, args=None):  # noqa
            if self is Annotated:
                return Annotated
            res = super()._subs_tree(tvars=tvars, args=args)
            # Flatten nested Annotated
            if isinstance(res[1], tuple) and res[1][0] is Annotated:
                sub_tp = res[1][1]
                sub_annot = res[1][2]
                return (Annotated, sub_tp, sub_annot + res[2])
            return res

        def _get_cons(self):
            """Return the class used to create instance of this type."""
            if self.__origin__ is None:
                raise TypeError("Cannot get the underlying type of a "
                                "non-specialized Annotated type.")
            tree = self._subs_tree()
            while isinstance(tree, tuple) and tree[0] is Annotated:
                tree = tree[1]
            if isinstance(tree, tuple):
                return tree[0]
            else:
                return tree

        @typing._tp_cache
        def __getitem__(self, params):
            if not isinstance(params, tuple):
                params = (params,)
            if self.__origin__ is not None:  # specializing an instantiated type
                return super().__getitem__(params)
            elif not isinstance(params, tuple) or len(params) < 2:
                raise TypeError("Annotated[...] should be instantiated "
                                "with at least two arguments (a type and an "
                                "annotation).")
            else:
                if (
                    isinstance(params[0], typing._TypingBase) and
                    type(params[0]).__name__ == "_ClassVar"
                ):
                    tp = params[0]
                else:
                    msg = "Annotated[t, ...]: t must be a type."
                    tp = typing._type_check(params[0], msg)
                metadata = tuple(params[1:])
            return self.__class__(
                self.__name__,
                self.__bases__,
                _no_slots_copy(self.__dict__),
                tvars=_type_vars((tp,)),
                # Metadata is a tuple so it won't be touched by _replace_args et al.
                args=(tp, metadata),
                origin=self,
            )

        def __call__(self, *args, **kwargs):
            cons = self._get_cons()
            result = cons(*args, **kwargs)
            try:
                result.__orig_class__ = self
            except AttributeError:
                pass
            return result

        def __getattr__(self, attr):
            # For simplicity we just don't relay all dunder names
            if self.__origin__ is not None and not _is_dunder(attr):
                return getattr(self._get_cons(), attr)
            raise AttributeError(attr)

        def __setattr__(self, attr, value):
            if _is_dunder(attr) or attr.startswith('_abc_'):
                super().__setattr__(attr, value)
            elif self.__origin__ is None:
                raise AttributeError(attr)
            else:
                setattr(self._get_cons(), attr, value)

        def __instancecheck__(self, obj):
            raise TypeError("Annotated cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError("Annotated cannot be used with issubclass().")

    class Annotated(metaclass=AnnotatedMeta):
        """Add context specific metadata to a type.

        Example: Annotated[int, runtime_check.Unsigned] indicates to the
        hypothetical runtime_check module that this type is an unsigned int.
        Every other consumer of this type can ignore this metadata and treat
        this type as int.

        The first argument to Annotated must be a valid type, the remaining
        arguments are kept as a tuple in the __metadata__ field.

        Details:

        - It's an error to call `Annotated` with less than two arguments.
        - Nested Annotated are flattened::

            Annotated[Annotated[T, Ann1, Ann2], Ann3] == Annotated[T, Ann1, Ann2, Ann3]

        - Instantiating an annotated type is equivalent to instantiating the
        underlying type::

            Annotated[C, Ann1](5) == C(5)

        - Annotated can be used as a generic type alias::

            Optimized = Annotated[T, runtime.Optimize()]
            Optimized[int] == Annotated[int, runtime.Optimize()]

            OptimizedList = Annotated[List[T], runtime.Optimize()]
            OptimizedList[int] == Annotated[List[int], runtime.Optimize()]
        """

# Python 3.8 has get_origin() and get_args() but those implementations aren't
# Annotated-aware, so we can't use those. Python 3.9's versions don't support
# ParamSpecArgs and ParamSpecKwargs, so only Python 3.10's versions will do.
if sys.version_info[:2] >= (3, 10):
    get_origin = typing.get_origin
    get_args = typing.get_args
# 3.7-3.9
elif PEP_560:
    try:
        # 3.9+
        from typing import _BaseGenericAlias
    except ImportError:
        _BaseGenericAlias = typing._GenericAlias
    try:
        # 3.9+
        from typing import GenericAlias
    except ImportError:
        GenericAlias = typing._GenericAlias

    def get_origin(tp):
        """Get the unsubscripted version of a type.

        This supports generic types, Callable, Tuple, Union, Literal, Final, ClassVar
        and Annotated. Return None for unsupported types. Examples::

            get_origin(Literal[42]) is Literal
            get_origin(int) is None
            get_origin(ClassVar[int]) is ClassVar
            get_origin(Generic) is Generic
            get_origin(Generic[T]) is Generic
            get_origin(Union[T, int]) is Union
            get_origin(List[Tuple[T, T]][int]) == list
            get_origin(P.args) is P
        """
        if isinstance(tp, _AnnotatedAlias):
            return Annotated
        if isinstance(tp, (typing._GenericAlias, GenericAlias, _BaseGenericAlias,
                           ParamSpecArgs, ParamSpecKwargs)):
            return tp.__origin__
        if tp is typing.Generic:
            return typing.Generic
        return None

    def get_args(tp):
        """Get type arguments with all substitutions performed.

        For unions, basic simplifications used by Union constructor are performed.
        Examples::
            get_args(Dict[str, int]) == (str, int)
            get_args(int) == ()
            get_args(Union[int, Union[T, int], str][int]) == (int, str)
            get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
            get_args(Callable[[], T][int]) == ([], int)
        """
        if isinstance(tp, _AnnotatedAlias):
            return (tp.__origin__,) + tp.__metadata__
        if isinstance(tp, (typing._GenericAlias, GenericAlias)):
            if getattr(tp, "_special", False):
                return ()
            res = tp.__args__
            if get_origin(tp) is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
            return res
        return ()


# 3.10+
if hasattr(typing, 'TypeAlias'):
    TypeAlias = typing.TypeAlias
# 3.9
elif sys.version_info[:2] >= (3, 9):
    class _TypeAliasForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    @_TypeAliasForm
    def TypeAlias(self, parameters):
        """Special marker indicating that an assignment should
        be recognized as a proper type alias definition by type
        checkers.

        For example::

            Predicate: TypeAlias = Callable[..., bool]

        It's invalid when used anywhere except as in the example above.
        """
        raise TypeError(f"{self} is not subscriptable")
# 3.7-3.8
elif sys.version_info[:2] >= (3, 7):
    class _TypeAliasForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    TypeAlias = _TypeAliasForm('TypeAlias',
                               doc="""Special marker indicating that an assignment should
                               be recognized as a proper type alias definition by type
                               checkers.

                               For example::

                                   Predicate: TypeAlias = Callable[..., bool]

                               It's invalid when used anywhere except as in the example
                               above.""")
# 3.6
else:
    class _TypeAliasMeta(typing.TypingMeta):
        """Metaclass for TypeAlias"""

        def __repr__(self):
            return 'typing_extensions.TypeAlias'

    class _TypeAliasBase(typing._FinalTypingBase, metaclass=_TypeAliasMeta, _root=True):
        """Special marker indicating that an assignment should
        be recognized as a proper type alias definition by type
        checkers.

        For example::

            Predicate: TypeAlias = Callable[..., bool]

        It's invalid when used anywhere except as in the example above.
        """
        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError("TypeAlias cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError("TypeAlias cannot be used with issubclass().")

        def __repr__(self):
            return 'typing_extensions.TypeAlias'

    TypeAlias = _TypeAliasBase(_root=True)


# Python 3.10+ has PEP 612
if hasattr(typing, 'ParamSpecArgs'):
    ParamSpecArgs = typing.ParamSpecArgs
    ParamSpecKwargs = typing.ParamSpecKwargs
# 3.6-3.9
else:
    class _Immutable:
        """Mixin to indicate that object should not be copied."""
        __slots__ = ()

        def __copy__(self):
            return self

        def __deepcopy__(self, memo):
            return self

    class ParamSpecArgs(_Immutable):
        """The args for a ParamSpec object.

        Given a ParamSpec object P, P.args is an instance of ParamSpecArgs.

        ParamSpecArgs objects have a reference back to their ParamSpec:

        P.args.__origin__ is P

        This type is meant for runtime introspection and has no special meaning to
        static type checkers.
        """
        def __init__(self, origin):
            self.__origin__ = origin

        def __repr__(self):
            return f"{self.__origin__.__name__}.args"

        def __eq__(self, other):
            if not isinstance(other, ParamSpecArgs):
                return NotImplemented
            return self.__origin__ == other.__origin__

    class ParamSpecKwargs(_Immutable):
        """The kwargs for a ParamSpec object.

        Given a ParamSpec object P, P.kwargs is an instance of ParamSpecKwargs.

        ParamSpecKwargs objects have a reference back to their ParamSpec:

        P.kwargs.__origin__ is P

        This type is meant for runtime introspection and has no special meaning to
        static type checkers.
        """
        def __init__(self, origin):
            self.__origin__ = origin

        def __repr__(self):
            return f"{self.__origin__.__name__}.kwargs"

        def __eq__(self, other):
            if not isinstance(other, ParamSpecKwargs):
                return NotImplemented
            return self.__origin__ == other.__origin__

# 3.10+
if hasattr(typing, 'ParamSpec'):
    ParamSpec = typing.ParamSpec
# 3.6-3.9
else:

    # Inherits from list as a workaround for Callable checks in Python < 3.9.2.
    class ParamSpec(list):
        """Parameter specification variable.

        Usage::

           P = ParamSpec('P')

        Parameter specification variables exist primarily for the benefit of static
        type checkers.  They are used to forward the parameter types of one
        callable to another callable, a pattern commonly found in higher order
        functions and decorators.  They are only valid when used in ``Concatenate``,
        or s the first argument to ``Callable``. In Python 3.10 and higher,
        they are also supported in user-defined Generics at runtime.
        See class Generic for more information on generic types.  An
        example for annotating a decorator::

           T = TypeVar('T')
           P = ParamSpec('P')

           def add_logging(f: Callable[P, T]) -> Callable[P, T]:
               '''A type-safe decorator to add logging to a function.'''
               def inner(*args: P.args, **kwargs: P.kwargs) -> T:
                   logging.info(f'{f.__name__} was called')
                   return f(*args, **kwargs)
               return inner

           @add_logging
           def add_two(x: float, y: float) -> float:
               '''Add two numbers together.'''
               return x + y

        Parameter specification variables defined with covariant=True or
        contravariant=True can be used to declare covariant or contravariant
        generic types.  These keyword arguments are valid, but their actual semantics
        are yet to be decided.  See PEP 612 for details.

        Parameter specification variables can be introspected. e.g.:

           P.__name__ == 'T'
           P.__bound__ == None
           P.__covariant__ == False
           P.__contravariant__ == False

        Note that only parameter specification variables defined in global scope can
        be pickled.
        """

        # Trick Generic __parameters__.
        __class__ = typing.TypeVar

        @property
        def args(self):
            return ParamSpecArgs(self)

        @property
        def kwargs(self):
            return ParamSpecKwargs(self)

        def __init__(self, name, *, bound=None, covariant=False, contravariant=False):
            super().__init__([self])
            self.__name__ = name
            self.__covariant__ = bool(covariant)
            self.__contravariant__ = bool(contravariant)
            if bound:
                self.__bound__ = typing._type_check(bound, 'Bound must be a type.')
            else:
                self.__bound__ = None

            # for pickling:
            try:
                def_mod = sys._getframe(1).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                def_mod = None
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod

        def __repr__(self):
            if self.__covariant__:
                prefix = '+'
            elif self.__contravariant__:
                prefix = '-'
            else:
                prefix = '~'
            return prefix + self.__name__

        def __hash__(self):
            return object.__hash__(self)

        def __eq__(self, other):
            return self is other

        def __reduce__(self):
            return self.__name__

        # Hack to get typing._type_check to pass.
        def __call__(self, *args, **kwargs):
            pass

        if not PEP_560:
            # Only needed in 3.6.
            def _get_type_vars(self, tvars):
                if self not in tvars:
                    tvars.append(self)


# 3.6-3.9
if not hasattr(typing, 'Concatenate'):
    # Inherits from list as a workaround for Callable checks in Python < 3.9.2.
    class _ConcatenateGenericAlias(list):

        # Trick Generic into looking into this for __parameters__.
        if PEP_560:
            __class__ = typing._GenericAlias
        else:
            __class__ = typing._TypingBase

        # Flag in 3.8.
        _special = False
        # Attribute in 3.6 and earlier.
        _gorg = typing.Generic

        def __init__(self, origin, args):
            super().__init__(args)
            self.__origin__ = origin
            self.__args__ = args

        def __repr__(self):
            _type_repr = typing._type_repr
            return (f'{_type_repr(self.__origin__)}'
                    f'[{", ".join(_type_repr(arg) for arg in self.__args__)}]')

        def __hash__(self):
            return hash((self.__origin__, self.__args__))

        # Hack to get typing._type_check to pass in Generic.
        def __call__(self, *args, **kwargs):
            pass

        @property
        def __parameters__(self):
            return tuple(
                tp for tp in self.__args__ if isinstance(tp, (typing.TypeVar, ParamSpec))
            )

        if not PEP_560:
            # Only required in 3.6.
            def _get_type_vars(self, tvars):
                if self.__origin__ and self.__parameters__:
                    typing._get_type_vars(self.__parameters__, tvars)


# 3.6-3.9
@typing._tp_cache
def _concatenate_getitem(self, parameters):
    if parameters == ():
        raise TypeError("Cannot take a Concatenate of no types.")
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    if not isinstance(parameters[-1], ParamSpec):
        raise TypeError("The last parameter to Concatenate should be a "
                        "ParamSpec variable.")
    msg = "Concatenate[arg, ...]: each arg must be a type."
    parameters = tuple(typing._type_check(p, msg) for p in parameters)
    return _ConcatenateGenericAlias(self, parameters)


# 3.10+
if hasattr(typing, 'Concatenate'):
    Concatenate = typing.Concatenate
    _ConcatenateGenericAlias = typing._ConcatenateGenericAlias # noqa
# 3.9
elif sys.version_info[:2] >= (3, 9):
    @_TypeAliasForm
    def Concatenate(self, parameters):
        """Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
        higher order function which adds, removes or transforms parameters of a
        callable.

        For example::

           Callable[Concatenate[int, P], int]

        See PEP 612 for detailed information.
        """
        return _concatenate_getitem(self, parameters)
# 3.7-8
elif sys.version_info[:2] >= (3, 7):
    class _ConcatenateForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            return _concatenate_getitem(self, parameters)

    Concatenate = _ConcatenateForm(
        'Concatenate',
        doc="""Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
        higher order function which adds, removes or transforms parameters of a
        callable.

        For example::

           Callable[Concatenate[int, P], int]

        See PEP 612 for detailed information.
        """)
# 3.6
else:
    class _ConcatenateAliasMeta(typing.TypingMeta):
        """Metaclass for Concatenate."""

        def __repr__(self):
            return 'typing_extensions.Concatenate'

    class _ConcatenateAliasBase(typing._FinalTypingBase,
                                metaclass=_ConcatenateAliasMeta,
                                _root=True):
        """Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
        higher order function which adds, removes or transforms parameters of a
        callable.

        For example::

           Callable[Concatenate[int, P], int]

        See PEP 612 for detailed information.
        """
        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError("Concatenate cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError("Concatenate cannot be used with issubclass().")

        def __repr__(self):
            return 'typing_extensions.Concatenate'

        def __getitem__(self, parameters):
            return _concatenate_getitem(self, parameters)

    Concatenate = _ConcatenateAliasBase(_root=True)

# 3.10+
if hasattr(typing, 'TypeGuard'):
    TypeGuard = typing.TypeGuard
# 3.9
elif sys.version_info[:2] >= (3, 9):
    class _TypeGuardForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    @_TypeGuardForm
    def TypeGuard(self, parameters):
        """Special typing form used to annotate the return type of a user-defined
        type guard function.  ``TypeGuard`` only accepts a single type argument.
        At runtime, functions marked this way should return a boolean.

        ``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static
        type checkers to determine a more precise type of an expression within a
        program's code flow.  Usually type narrowing is done by analyzing
        conditional code flow and applying the narrowing to a block of code.  The
        conditional expression here is sometimes referred to as a "type guard".

        Sometimes it would be convenient to use a user-defined boolean function
        as a type guard.  Such a function should use ``TypeGuard[...]`` as its
        return type to alert static type checkers to this intention.

        Using  ``-> TypeGuard`` tells the static type checker that for a given
        function:

        1. The return value is a boolean.
        2. If the return value is ``True``, the type of its argument
        is the type inside ``TypeGuard``.

        For example::

            def is_str(val: Union[str, float]):
                # "isinstance" type guard
                if isinstance(val, str):
                    # Type of ``val`` is narrowed to ``str``
                    ...
                else:
                    # Else, type of ``val`` is narrowed to ``float``.
                    ...

        Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower
        form of ``TypeA`` (it can even be a wider form) and this may lead to
        type-unsafe results.  The main reason is to allow for things like
        narrowing ``List[object]`` to ``List[str]`` even though the latter is not
        a subtype of the former, since ``List`` is invariant.  The responsibility of
        writing type-safe type guards is left to the user.

        ``TypeGuard`` also works with type variables.  For more information, see
        PEP 647 (User-Defined Type Guards).
        """
        item = typing._type_check(parameters, f'{self} accepts only single type.')
        return typing._GenericAlias(self, (item,))
# 3.7-3.8
elif sys.version_info[:2] >= (3, 7):
    class _TypeGuardForm(typing._SpecialForm, _root=True):

        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            item = typing._type_check(parameters,
                                      f'{self._name} accepts only a single type')
            return typing._GenericAlias(self, (item,))

    TypeGuard = _TypeGuardForm(
        'TypeGuard',
        doc="""Special typing form used to annotate the return type of a user-defined
        type guard function.  ``TypeGuard`` only accepts a single type argument.
        At runtime, functions marked this way should return a boolean.

        ``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static
        type checkers to determine a more precise type of an expression within a
        program's code flow.  Usually type narrowing is done by analyzing
        conditional code flow and applying the narrowing to a block of code.  The
        conditional expression here is sometimes referred to as a "type guard".

        Sometimes it would be convenient to use a user-defined boolean function
        as a type guard.  Such a function should use ``TypeGuard[...]`` as its
        return type to alert static type checkers to this intention.

        Using  ``-> TypeGuard`` tells the static type checker that for a given
        function:

        1. The return value is a boolean.
        2. If the return value is ``True``, the type of its argument
        is the type inside ``TypeGuard``.

        For example::

            def is_str(val: Union[str, float]):
                # "isinstance" type guard
                if isinstance(val, str):
                    # Type of ``val`` is narrowed to ``str``
                    ...
                else:
                    # Else, type of ``val`` is narrowed to ``float``.
                    ...

        Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower
        form of ``TypeA`` (it can even be a wider form) and this may lead to
        type-unsafe results.  The main reason is to allow for things like
        narrowing ``List[object]`` to ``List[str]`` even though the latter is not
        a subtype of the former, since ``List`` is invariant.  The responsibility of
        writing type-safe type guards is left to the user.

        ``TypeGuard`` also works with type variables.  For more information, see
        PEP 647 (User-Defined Type Guards).
        """)
# 3.6
else:
    class _TypeGuard(typing._FinalTypingBase, _root=True):
        """Special typing form used to annotate the return type of a user-defined
        type guard function.  ``TypeGuard`` only accepts a single type argument.
        At runtime, functions marked this way should return a boolean.

        ``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static
        type checkers to determine a more precise type of an expression within a
        program's code flow.  Usually type narrowing is done by analyzing
        conditional code flow and applying the narrowing to a block of code.  The
        conditional expression here is sometimes referred to as a "type guard".

        Sometimes it would be convenient to use a user-defined boolean function
        as a type guard.  Such a function should use ``TypeGuard[...]`` as its
        return type to alert static type checkers to this intention.

        Using  ``-> TypeGuard`` tells the static type checker that for a given
        function:

        1. The return value is a boolean.
        2. If the return value is ``True``, the type of its argument
        is the type inside ``TypeGuard``.

        For example::

            def is_str(val: Union[str, float]):
                # "isinstance" type guard
                if isinstance(val, str):
                    # Type of ``val`` is narrowed to ``str``
                    ...
                else:
                    # Else, type of ``val`` is narrowed to ``float``.
                    ...

        Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower
        form of ``TypeA`` (it can even be a wider form) and this may lead to
        type-unsafe results.  The main reason is to allow for things like
        narrowing ``List[object]`` to ``List[str]`` even though the latter is not
        a subtype of the former, since ``List`` is invariant.  The responsibility of
        writing type-safe type guards is left to the user.

        ``TypeGuard`` also works with type variables.  For more information, see
        PEP 647 (User-Defined Type Guards).
        """

        __slots__ = ('__type__',)

        def __init__(self, tp=None, **kwds):
            self.__type__ = tp

        def __getitem__(self, item):
            cls = type(self)
            if self.__type__ is None:
                return cls(typing._type_check(item,
                           f'{cls.__name__[1:]} accepts only a single type.'),
                           _root=True)
            raise TypeError(f'{cls.__name__[1:]} cannot be further subscripted')

        def _eval_type(self, globalns, localns):
            new_tp = typing._eval_type(self.__type__, globalns, localns)
            if new_tp == self.__type__:
                return self
            return type(self)(new_tp, _root=True)

        def __repr__(self):
            r = super().__repr__()
            if self.__type__ is not None:
                r += f'[{typing._type_repr(self.__type__)}]'
            return r

        def __hash__(self):
            return hash((type(self).__name__, self.__type__))

        def __eq__(self, other):
            if not isinstance(other, _TypeGuard):
                return NotImplemented
            if self.__type__ is not None:
                return self.__type__ == other.__type__
            return self is other

    TypeGuard = _TypeGuard(_root=True)


if sys.version_info[:2] >= (3, 7):
    # Vendored from cpython typing._SpecialFrom
    class _SpecialForm(typing._Final, _root=True):
        __slots__ = ('_name', '__doc__', '_getitem')

        def __init__(self, getitem):
            self._getitem = getitem
            self._name = getitem.__name__
            self.__doc__ = getitem.__doc__

        def __getattr__(self, item):
            if item in {'__name__', '__qualname__'}:
                return self._name

            raise AttributeError(item)

        def __mro_entries__(self, bases):
            raise TypeError(f"Cannot subclass {self!r}")

        def __repr__(self):
            return f'typing_extensions.{self._name}'

        def __reduce__(self):
            return self._name

        def __call__(self, *args, **kwds):
            raise TypeError(f"Cannot instantiate {self!r}")

        def __or__(self, other):
            return typing.Union[self, other]

        def __ror__(self, other):
            return typing.Union[other, self]

        def __instancecheck__(self, obj):
            raise TypeError(f"{self} cannot be used with isinstance()")

        def __subclasscheck__(self, cls):
            raise TypeError(f"{self} cannot be used with issubclass()")

        @typing._tp_cache
        def __getitem__(self, parameters):
            return self._getitem(self, parameters)


if hasattr(typing, "LiteralString"):
    LiteralString = typing.LiteralString
elif sys.version_info[:2] >= (3, 7):
    @_SpecialForm
    def LiteralString(self, params):
        """Represents an arbitrary literal string.

        Example::

          from typing_extensions import LiteralString

          def query(sql: LiteralString) -> ...:
              ...

          query("SELECT * FROM table")  # ok
          query(f"SELECT * FROM {input()}")  # not ok

        See PEP 675 for details.

        """
        raise TypeError(f"{self} is not subscriptable")
else:
    class _LiteralString(typing._FinalTypingBase, _root=True):
        """Represents an arbitrary literal string.

        Example::

          from typing_extensions import LiteralString

          def query(sql: LiteralString) -> ...:
              ...

          query("SELECT * FROM table")  # ok
          query(f"SELECT * FROM {input()}")  # not ok

        See PEP 675 for details.

        """

        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError(f"{self} cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError(f"{self} cannot be used with issubclass().")

    LiteralString = _LiteralString(_root=True)


if hasattr(typing, "Self"):
    Self = typing.Self
elif sys.version_info[:2] >= (3, 7):
    @_SpecialForm
    def Self(self, params):
        """Used to spell the type of "self" in classes.

        Example::

          from typing import Self

          class ReturnsSelf:
              def parse(self, data: bytes) -> Self:
                  ...
                  return self

        """

        raise TypeError(f"{self} is not subscriptable")
else:
    class _Self(typing._FinalTypingBase, _root=True):
        """Used to spell the type of "self" in classes.

        Example::

          from typing import Self

          class ReturnsSelf:
              def parse(self, data: bytes) -> Self:
                  ...
                  return self

        """

        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError(f"{self} cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError(f"{self} cannot be used with issubclass().")

    Self = _Self(_root=True)


if hasattr(typing, "Never"):
    Never = typing.Never
elif sys.version_info[:2] >= (3, 7):
    @_SpecialForm
    def Never(self, params):
        """The bottom type, a type that has no members.

        This can be used to define a function that should never be
        called, or a function that never returns::

            from typing_extensions import Never

            def never_call_me(arg: Never) -> None:
                pass

            def int_or_str(arg: int | str) -> None:
                never_call_me(arg)  # type checker error
                match arg:
                    case int():
                        print("It's an int")
                    case str():
                        print("It's a str")
                    case _:
                        never_call_me(arg)  # ok, arg is of type Never

        """

        raise TypeError(f"{self} is not subscriptable")
else:
    class _Never(typing._FinalTypingBase, _root=True):
        """The bottom type, a type that has no members.

        This can be used to define a function that should never be
        called, or a function that never returns::

            from typing_extensions import Never

            def never_call_me(arg: Never) -> None:
                pass

            def int_or_str(arg: int | str) -> None:
                never_call_me(arg)  # type checker error
                match arg:
                    case int():
                        print("It's an int")
                    case str():
                        print("It's a str")
                    case _:
                        never_call_me(arg)  # ok, arg is of type Never

        """

        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError(f"{self} cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError(f"{self} cannot be used with issubclass().")

    Never = _Never(_root=True)


if hasattr(typing, 'Required'):
    Required = typing.Required
    NotRequired = typing.NotRequired
elif sys.version_info[:2] >= (3, 9):
    class _ExtensionsSpecialForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    @_ExtensionsSpecialForm
    def Required(self, parameters):
        """A special typing construct to mark a key of a total=False TypedDict
        as required. For example:

            class Movie(TypedDict, total=False):
                title: Required[str]
                year: int

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )

        There is no runtime checking that a required key is actually provided
        when instantiating a related TypedDict.
        """
        item = typing._type_check(parameters, f'{self._name} accepts only single type')
        return typing._GenericAlias(self, (item,))

    @_ExtensionsSpecialForm
    def NotRequired(self, parameters):
        """A special typing construct to mark a key of a TypedDict as
        potentially missing. For example:

            class Movie(TypedDict):
                title: str
                year: NotRequired[int]

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )
        """
        item = typing._type_check(parameters, f'{self._name} accepts only single type')
        return typing._GenericAlias(self, (item,))

elif sys.version_info[:2] >= (3, 7):
    class _RequiredForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            item = typing._type_check(parameters,
                                      '{} accepts only single type'.format(self._name))
            return typing._GenericAlias(self, (item,))

    Required = _RequiredForm(
        'Required',
        doc="""A special typing construct to mark a key of a total=False TypedDict
        as required. For example:

            class Movie(TypedDict, total=False):
                title: Required[str]
                year: int

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )

        There is no runtime checking that a required key is actually provided
        when instantiating a related TypedDict.
        """)
    NotRequired = _RequiredForm(
        'NotRequired',
        doc="""A special typing construct to mark a key of a TypedDict as
        potentially missing. For example:

            class Movie(TypedDict):
                title: str
                year: NotRequired[int]

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )
        """)
else:
    # NOTE: Modeled after _Final's implementation when _FinalTypingBase available
    class _MaybeRequired(typing._FinalTypingBase, _root=True):
        __slots__ = ('__type__',)

        def __init__(self, tp=None, **kwds):
            self.__type__ = tp

        def __getitem__(self, item):
            cls = type(self)
            if self.__type__ is None:
                return cls(typing._type_check(item,
                           '{} accepts only single type.'.format(cls.__name__[1:])),
                           _root=True)
            raise TypeError('{} cannot be further subscripted'
                            .format(cls.__name__[1:]))

        def _eval_type(self, globalns, localns):
            new_tp = typing._eval_type(self.__type__, globalns, localns)
            if new_tp == self.__type__:
                return self
            return type(self)(new_tp, _root=True)

        def __repr__(self):
            r = super().__repr__()
            if self.__type__ is not None:
                r += '[{}]'.format(typing._type_repr(self.__type__))
            return r

        def __hash__(self):
            return hash((type(self).__name__, self.__type__))

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            if self.__type__ is not None:
                return self.__type__ == other.__type__
            return self is other

    class _Required(_MaybeRequired, _root=True):
        """A special typing construct to mark a key of a total=False TypedDict
        as required. For example:

            class Movie(TypedDict, total=False):
                title: Required[str]
                year: int

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )

        There is no runtime checking that a required key is actually provided
        when instantiating a related TypedDict.
        """

    class _NotRequired(_MaybeRequired, _root=True):
        """A special typing construct to mark a key of a TypedDict as
        potentially missing. For example:

            class Movie(TypedDict):
                title: str
                year: NotRequired[int]

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )
        """

    Required = _Required(_root=True)
    NotRequired = _NotRequired(_root=True)


if sys.version_info[:2] >= (3, 9):
    class _UnpackSpecialForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    class _UnpackAlias(typing._GenericAlias, _root=True):
        __class__ = typing.TypeVar

    @_UnpackSpecialForm
    def Unpack(self, parameters):
        """A special typing construct to unpack a variadic type. For example:

            Shape = TypeVarTuple('Shape')
            Batch = NewType('Batch', int)

            def add_batch_axis(
                x: Array[Unpack[Shape]]
            ) -> Array[Batch, Unpack[Shape]]: ...

        """
        item = typing._type_check(parameters, f'{self._name} accepts only single type')
        return _UnpackAlias(self, (item,))

    def _is_unpack(obj):
        return isinstance(obj, _UnpackAlias)

elif sys.version_info[:2] >= (3, 7):
    class _UnpackAlias(typing._GenericAlias, _root=True):
        __class__ = typing.TypeVar

    class _UnpackForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            item = typing._type_check(parameters,
                                      f'{self._name} accepts only single type')
            return _UnpackAlias(self, (item,))

    Unpack = _UnpackForm(
        'Unpack',
        doc="""A special typing construct to unpack a variadic type. For example:

            Shape = TypeVarTuple('Shape')
            Batch = NewType('Batch', int)

            def add_batch_axis(
                x: Array[Unpack[Shape]]
            ) -> Array[Batch, Unpack[Shape]]: ...

        """)

    def _is_unpack(obj):
        return isinstance(obj, _UnpackAlias)

else:
    # NOTE: Modeled after _Final's implementation when _FinalTypingBase available
    class _Unpack(typing._FinalTypingBase, _root=True):
        """A special typing construct to unpack a variadic type. For example:

            Shape = TypeVarTuple('Shape')
            Batch = NewType('Batch', int)

            def add_batch_axis(
                x: Array[Unpack[Shape]]
            ) -> Array[Batch, Unpack[Shape]]: ...

        """
        __slots__ = ('__type__',)
        __class__ = typing.TypeVar

        def __init__(self, tp=None, **kwds):
            self.__type__ = tp

        def __getitem__(self, item):
            cls = type(self)
            if self.__type__ is None:
                return cls(typing._type_check(item,
                           'Unpack accepts only single type.'),
                           _root=True)
            raise TypeError('Unpack cannot be further subscripted')

        def _eval_type(self, globalns, localns):
            new_tp = typing._eval_type(self.__type__, globalns, localns)
            if new_tp == self.__type__:
                return self
            return type(self)(new_tp, _root=True)

        def __repr__(self):
            r = super().__repr__()
            if self.__type__ is not None:
                r += '[{}]'.format(typing._type_repr(self.__type__))
            return r

        def __hash__(self):
            return hash((type(self).__name__, self.__type__))

        def __eq__(self, other):
            if not isinstance(other, _Unpack):
                return NotImplemented
            if self.__type__ is not None:
                return self.__type__ == other.__type__
            return self is other

        # For 3.6 only
        def _get_type_vars(self, tvars):
            self.__type__._get_type_vars(tvars)

    Unpack = _Unpack(_root=True)

    def _is_unpack(obj):
        return isinstance(obj, _Unpack)


class TypeVarTuple:
    """Type variable tuple.

    Usage::

        Ts = TypeVarTuple('Ts')

    In the same way that a normal type variable is a stand-in for a single
    type such as ``int``, a type variable *tuple* is a stand-in for a *tuple* type such as
    ``Tuple[int, str]``.

    Type variable tuples can be used in ``Generic`` declarations.
    Consider the following example::

        class Array(Generic[*Ts]): ...

    The ``Ts`` type variable tuple here behaves like ``tuple[T1, T2]``,
    where ``T1`` and ``T2`` are type variables. To use these type variables
    as type parameters of ``Array``, we must *unpack* the type variable tuple using
    the star operator: ``*Ts``. The signature of ``Array`` then behaves
    as if we had simply written ``class Array(Generic[T1, T2]): ...``.
    In contrast to ``Generic[T1, T2]``, however, ``Generic[*Shape]`` allows
    us to parameterise the class with an *arbitrary* number of type parameters.

    Type variable tuples can be used anywhere a normal ``TypeVar`` can.
    This includes class definitions, as shown above, as well as function
    signatures and variable annotations::

        class Array(Generic[*Ts]):

            def __init__(self, shape: Tuple[*Ts]):
                self._shape: Tuple[*Ts] = shape

            def get_shape(self) -> Tuple[*Ts]:
                return self._shape

        shape = (Height(480), Width(640))
        x: Array[Height, Width] = Array(shape)
        y = abs(x)  # Inferred type is Array[Height, Width]
        z = x + x   #        ...    is Array[Height, Width]
        x.get_shape()  #     ...    is tuple[Height, Width]

    """

    # Trick Generic __parameters__.
    __class__ = typing.TypeVar

    def __iter__(self):
        yield self.__unpacked__

    def __init__(self, name):
        self.__name__ = name

        # for pickling:
        try:
            def_mod = sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            def_mod = None
        if def_mod != 'typing_extensions':
            self.__module__ = def_mod

        self.__unpacked__ = Unpack[self]

    def __repr__(self):
        return self.__name__

    def __hash__(self):
        return object.__hash__(self)

    def __eq__(self, other):
        return self is other

    def __reduce__(self):
        return self.__name__

    def __init_subclass__(self, *args, **kwds):
        if '_root' not in kwds:
            raise TypeError("Cannot subclass special typing classes")

    if not PEP_560:
        # Only needed in 3.6.
        def _get_type_vars(self, tvars):
            if self not in tvars:
                tvars.append(self)


if hasattr(typing, "reveal_type"):
    reveal_type = typing.reveal_type
else:
    def reveal_type(__obj: T) -> T:
        """Reveal the inferred type of a variable.

        When a static type checker encounters a call to ``reveal_type()``,
        it will emit the inferred type of the argument::

            x: int = 1
            reveal_type(x)

        Running a static type checker (e.g., ``mypy``) on this example
        will produce output similar to 'Revealed type is "builtins.int"'.

        At runtime, the function prints the runtime type of the
        argument and returns it unchanged.

        """
        print(f"Runtime type is {type(__obj).__name__!r}", file=sys.stderr)
        return __obj


if hasattr(typing, "assert_never"):
    assert_never = typing.assert_never
else:
    def assert_never(__arg: Never) -> Never:
        """Assert to the type checker that a line of code is unreachable.

        Example::

            def int_or_str(arg: int | str) -> None:
                match arg:
                    case int():
                        print("It's an int")
                    case str():
                        print("It's a str")
                    case _:
                        assert_never(arg)

        If a type checker finds that a call to assert_never() is
        reachable, it will emit an error.

        At runtime, this throws an exception when called.

        """
        raise AssertionError("Expected code to be unreachable")


if hasattr(typing, 'dataclass_transform'):
    dataclass_transform = typing.dataclass_transform
else:
    def dataclass_transform(
        *,
        eq_default: bool = True,
        order_default: bool = False,
        kw_only_default: bool = False,
        field_descriptors: typing.Tuple[
            typing.Union[typing.Type[typing.Any], typing.Callable[..., typing.Any]],
            ...
        ] = (),
    ) -> typing.Callable[[T], T]:
        """Decorator that marks a function, class, or metaclass as providing
        dataclass-like behavior.

        Example:

            from typing_extensions import dataclass_transform

            _T = TypeVar("_T")

            # Used on a decorator function
            @dataclass_transform()
            def create_model(cls: type[_T]) -> type[_T]:
                ...
                return cls

            @create_model
            class CustomerModel:
                id: int
                name: str

            # Used on a base class
            @dataclass_transform()
            class ModelBase: ...

            class CustomerModel(ModelBase):
                id: int
                name: str

            # Used on a metaclass
            @dataclass_transform()
            class ModelMeta(type): ...

            class ModelBase(metaclass=ModelMeta): ...

            class CustomerModel(ModelBase):
                id: int
                name: str

        Each of the ``CustomerModel`` classes defined in this example will now
        behave similarly to a dataclass created with the ``@dataclasses.dataclass``
        decorator. For example, the type checker will synthesize an ``__init__``
        method.

        The arguments to this decorator can be used to customize this behavior:
        - ``eq_default`` indicates whether the ``eq`` parameter is assumed to be
          True or False if it is omitted by the caller.
        - ``order_default`` indicates whether the ``order`` parameter is
          assumed to be True or False if it is omitted by the caller.
        - ``kw_only_default`` indicates whether the ``kw_only`` parameter is
          assumed to be True or False if it is omitted by the caller.
        - ``field_descriptors`` specifies a static list of supported classes
          or functions, that describe fields, similar to ``dataclasses.field()``.

        At runtime, this decorator records its arguments in the
        ``__dataclass_transform__`` attribute on the decorated object.

        See PEP 681 for details.

        """
        def decorator(cls_or_fn):
            cls_or_fn.__dataclass_transform__ = {
                "eq_default": eq_default,
                "order_default": order_default,
                "kw_only_default": kw_only_default,
                "field_descriptors": field_descriptors,
            }
            return cls_or_fn
        return decorator


# We have to do some monkey patching to deal with the dual nature of
# Unpack/TypeVarTuple:
# - We want Unpack to be a kind of TypeVar so it gets accepted in
#   Generic[Unpack[Ts]]
# - We want it to *not* be treated as a TypeVar for the purposes of
#   counting generic parameters, so that when we subscript a generic,
#   the runtime doesn't try to substitute the Unpack with the subscripted type.
if not hasattr(typing, "TypeVarTuple"):
    typing._collect_type_vars = _collect_type_vars
    typing._check_generic = _check_generic
