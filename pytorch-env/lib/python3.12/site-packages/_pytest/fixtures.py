# mypy: allow-untyped-defs
from __future__ import annotations

import abc
from collections import defaultdict
from collections import deque
import dataclasses
import functools
import inspect
import os
from pathlib import Path
import sys
import types
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import OrderedDict
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings

import _pytest
from _pytest import nodes
from _pytest._code import getfslineno
from _pytest._code import Source
from _pytest._code.code import FormattedExcinfo
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import _PytestWrapper
from _pytest.compat import assert_never
from _pytest.compat import get_real_func
from _pytest.compat import get_real_method
from _pytest.compat import getfuncargnames
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_generator
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.compat import safe_getattr
from _pytest.compat import safe_isclass
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.deprecated import YIELD_FIXTURE
from _pytest.main import Session
from _pytest.mark import Mark
from _pytest.mark import ParameterSet
from _pytest.mark.structures import MarkDecorator
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import TEST_OUTCOME
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.scope import _ScopeName
from _pytest.scope import HIGH_SCOPES
from _pytest.scope import Scope


if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


if TYPE_CHECKING:
    from _pytest.python import CallSpec2
    from _pytest.python import Function
    from _pytest.python import Metafunc


# The value of the fixture -- return/yield of the fixture function (type variable).
FixtureValue = TypeVar("FixtureValue")
# The type of the fixture function (type variable).
FixtureFunction = TypeVar("FixtureFunction", bound=Callable[..., object])
# The type of a fixture function (type alias generic in fixture value).
_FixtureFunc = Union[
    Callable[..., FixtureValue], Callable[..., Generator[FixtureValue, None, None]]
]
# The type of FixtureDef.cached_result (type alias generic in fixture value).
_FixtureCachedResult = Union[
    Tuple[
        # The result.
        FixtureValue,
        # Cache key.
        object,
        None,
    ],
    Tuple[
        None,
        # Cache key.
        object,
        # The exception and the original traceback.
        Tuple[BaseException, Optional[types.TracebackType]],
    ],
]


@dataclasses.dataclass(frozen=True)
class PseudoFixtureDef(Generic[FixtureValue]):
    cached_result: _FixtureCachedResult[FixtureValue]
    _scope: Scope


def pytest_sessionstart(session: Session) -> None:
    session._fixturemanager = FixtureManager(session)


def get_scope_package(
    node: nodes.Item,
    fixturedef: FixtureDef[object],
) -> nodes.Node | None:
    from _pytest.python import Package

    for parent in node.iter_parents():
        if isinstance(parent, Package) and parent.nodeid == fixturedef.baseid:
            return parent
    return node.session


def get_scope_node(node: nodes.Node, scope: Scope) -> nodes.Node | None:
    import _pytest.python

    if scope is Scope.Function:
        # Type ignored because this is actually safe, see:
        # https://github.com/python/mypy/issues/4717
        return node.getparent(nodes.Item)  # type: ignore[type-abstract]
    elif scope is Scope.Class:
        return node.getparent(_pytest.python.Class)
    elif scope is Scope.Module:
        return node.getparent(_pytest.python.Module)
    elif scope is Scope.Package:
        return node.getparent(_pytest.python.Package)
    elif scope is Scope.Session:
        return node.getparent(_pytest.main.Session)
    else:
        assert_never(scope)


def getfixturemarker(obj: object) -> FixtureFunctionMarker | None:
    """Return fixturemarker or None if it doesn't exist or raised
    exceptions."""
    return cast(
        Optional[FixtureFunctionMarker],
        safe_getattr(obj, "_pytestfixturefunction", None),
    )


# Algorithm for sorting on a per-parametrized resource setup basis.
# It is called for Session scope first and performs sorting
# down to the lower scopes such as to minimize number of "high scope"
# setups and teardowns.


@dataclasses.dataclass(frozen=True)
class FixtureArgKey:
    argname: str
    param_index: int
    scoped_item_path: Path | None
    item_cls: type | None


_V = TypeVar("_V")
OrderedSet = Dict[_V, None]


def get_parametrized_fixture_argkeys(
    item: nodes.Item, scope: Scope
) -> Iterator[FixtureArgKey]:
    """Return list of keys for all parametrized arguments which match
    the specified scope."""
    assert scope is not Scope.Function

    try:
        callspec: CallSpec2 = item.callspec  # type: ignore[attr-defined]
    except AttributeError:
        return

    item_cls = None
    if scope is Scope.Session:
        scoped_item_path = None
    elif scope is Scope.Package:
        # Package key = module's directory.
        scoped_item_path = item.path.parent
    elif scope is Scope.Module:
        scoped_item_path = item.path
    elif scope is Scope.Class:
        scoped_item_path = item.path
        item_cls = item.cls  # type: ignore[attr-defined]
    else:
        assert_never(scope)

    for argname in callspec.indices:
        if callspec._arg2scope[argname] != scope:
            continue
        param_index = callspec.indices[argname]
        yield FixtureArgKey(argname, param_index, scoped_item_path, item_cls)


def reorder_items(items: Sequence[nodes.Item]) -> list[nodes.Item]:
    argkeys_by_item: dict[Scope, dict[nodes.Item, OrderedSet[FixtureArgKey]]] = {}
    items_by_argkey: dict[
        Scope, dict[FixtureArgKey, OrderedDict[nodes.Item, None]]
    ] = {}
    for scope in HIGH_SCOPES:
        scoped_argkeys_by_item = argkeys_by_item[scope] = {}
        scoped_items_by_argkey = items_by_argkey[scope] = defaultdict(OrderedDict)
        for item in items:
            argkeys = dict.fromkeys(get_parametrized_fixture_argkeys(item, scope))
            if argkeys:
                scoped_argkeys_by_item[item] = argkeys
                for argkey in argkeys:
                    scoped_items_by_argkey[argkey][item] = None

    items_set = dict.fromkeys(items)
    return list(
        reorder_items_atscope(
            items_set, argkeys_by_item, items_by_argkey, Scope.Session
        )
    )


def reorder_items_atscope(
    items: OrderedSet[nodes.Item],
    argkeys_by_item: Mapping[Scope, Mapping[nodes.Item, OrderedSet[FixtureArgKey]]],
    items_by_argkey: Mapping[
        Scope, Mapping[FixtureArgKey, OrderedDict[nodes.Item, None]]
    ],
    scope: Scope,
) -> OrderedSet[nodes.Item]:
    if scope is Scope.Function or len(items) < 3:
        return items

    scoped_items_by_argkey = items_by_argkey[scope]
    scoped_argkeys_by_item = argkeys_by_item[scope]

    ignore: set[FixtureArgKey] = set()
    items_deque = deque(items)
    items_done: OrderedSet[nodes.Item] = {}
    while items_deque:
        no_argkey_items: OrderedSet[nodes.Item] = {}
        slicing_argkey = None
        while items_deque:
            item = items_deque.popleft()
            if item in items_done or item in no_argkey_items:
                continue
            argkeys = dict.fromkeys(
                k for k in scoped_argkeys_by_item.get(item, ()) if k not in ignore
            )
            if not argkeys:
                no_argkey_items[item] = None
            else:
                slicing_argkey, _ = argkeys.popitem()
                # We don't have to remove relevant items from later in the
                # deque because they'll just be ignored.
                matching_items = [
                    i for i in scoped_items_by_argkey[slicing_argkey] if i in items
                ]
                for i in reversed(matching_items):
                    items_deque.appendleft(i)
                    # Fix items_by_argkey order.
                    for other_scope in HIGH_SCOPES:
                        other_scoped_items_by_argkey = items_by_argkey[other_scope]
                        for argkey in argkeys_by_item[other_scope].get(i, ()):
                            other_scoped_items_by_argkey[argkey][i] = None
                            other_scoped_items_by_argkey[argkey].move_to_end(
                                i, last=False
                            )
                break
        if no_argkey_items:
            reordered_no_argkey_items = reorder_items_atscope(
                no_argkey_items, argkeys_by_item, items_by_argkey, scope.next_lower()
            )
            items_done.update(reordered_no_argkey_items)
        if slicing_argkey is not None:
            ignore.add(slicing_argkey)
    return items_done


@dataclasses.dataclass(frozen=True)
class FuncFixtureInfo:
    """Fixture-related information for a fixture-requesting item (e.g. test
    function).

    This is used to examine the fixtures which an item requests statically
    (known during collection). This includes autouse fixtures, fixtures
    requested by the `usefixtures` marker, fixtures requested in the function
    parameters, and the transitive closure of these.

    An item may also request fixtures dynamically (using `request.getfixturevalue`);
    these are not reflected here.
    """

    __slots__ = ("argnames", "initialnames", "names_closure", "name2fixturedefs")

    # Fixture names that the item requests directly by function parameters.
    argnames: tuple[str, ...]
    # Fixture names that the item immediately requires. These include
    # argnames + fixture names specified via usefixtures and via autouse=True in
    # fixture definitions.
    initialnames: tuple[str, ...]
    # The transitive closure of the fixture names that the item requires.
    # Note: can't include dynamic dependencies (`request.getfixturevalue` calls).
    names_closure: list[str]
    # A map from a fixture name in the transitive closure to the FixtureDefs
    # matching the name which are applicable to this function.
    # There may be multiple overriding fixtures with the same name. The
    # sequence is ordered from furthest to closes to the function.
    name2fixturedefs: dict[str, Sequence[FixtureDef[Any]]]

    def prune_dependency_tree(self) -> None:
        """Recompute names_closure from initialnames and name2fixturedefs.

        Can only reduce names_closure, which means that the new closure will
        always be a subset of the old one. The order is preserved.

        This method is needed because direct parametrization may shadow some
        of the fixtures that were included in the originally built dependency
        tree. In this way the dependency tree can get pruned, and the closure
        of argnames may get reduced.
        """
        closure: set[str] = set()
        working_set = set(self.initialnames)
        while working_set:
            argname = working_set.pop()
            # Argname may be something not included in the original names_closure,
            # in which case we ignore it. This currently happens with pseudo
            # FixtureDefs which wrap 'get_direct_param_fixture_func(request)'.
            # So they introduce the new dependency 'request' which might have
            # been missing in the original tree (closure).
            if argname not in closure and argname in self.names_closure:
                closure.add(argname)
                if argname in self.name2fixturedefs:
                    working_set.update(self.name2fixturedefs[argname][-1].argnames)

        self.names_closure[:] = sorted(closure, key=self.names_closure.index)


class FixtureRequest(abc.ABC):
    """The type of the ``request`` fixture.

    A request object gives access to the requesting test context and has a
    ``param`` attribute in case the fixture is parametrized.
    """

    def __init__(
        self,
        pyfuncitem: Function,
        fixturename: str | None,
        arg2fixturedefs: dict[str, Sequence[FixtureDef[Any]]],
        fixture_defs: dict[str, FixtureDef[Any]],
        *,
        _ispytest: bool = False,
    ) -> None:
        check_ispytest(_ispytest)
        #: Fixture for which this request is being performed.
        self.fixturename: Final = fixturename
        self._pyfuncitem: Final = pyfuncitem
        # The FixtureDefs for each fixture name requested by this item.
        # Starts from the statically-known fixturedefs resolved during
        # collection. Dynamically requested fixtures (using
        # `request.getfixturevalue("foo")`) are added dynamically.
        self._arg2fixturedefs: Final = arg2fixturedefs
        # The evaluated argnames so far, mapping to the FixtureDef they resolved
        # to.
        self._fixture_defs: Final = fixture_defs
        # Notes on the type of `param`:
        # -`request.param` is only defined in parametrized fixtures, and will raise
        #   AttributeError otherwise. Python typing has no notion of "undefined", so
        #   this cannot be reflected in the type.
        # - Technically `param` is only (possibly) defined on SubRequest, not
        #   FixtureRequest, but the typing of that is still in flux so this cheats.
        # - In the future we might consider using a generic for the param type, but
        #   for now just using Any.
        self.param: Any

    @property
    def _fixturemanager(self) -> FixtureManager:
        return self._pyfuncitem.session._fixturemanager

    @property
    @abc.abstractmethod
    def _scope(self) -> Scope:
        raise NotImplementedError()

    @property
    def scope(self) -> _ScopeName:
        """Scope string, one of "function", "class", "module", "package", "session"."""
        return self._scope.value

    @abc.abstractmethod
    def _check_scope(
        self,
        requested_fixturedef: FixtureDef[object] | PseudoFixtureDef[object],
        requested_scope: Scope,
    ) -> None:
        raise NotImplementedError()

    @property
    def fixturenames(self) -> list[str]:
        """Names of all active fixtures in this request."""
        result = list(self._pyfuncitem.fixturenames)
        result.extend(set(self._fixture_defs).difference(result))
        return result

    @property
    @abc.abstractmethod
    def node(self):
        """Underlying collection node (depends on current request scope)."""
        raise NotImplementedError()

    @property
    def config(self) -> Config:
        """The pytest config object associated with this request."""
        return self._pyfuncitem.config

    @property
    def function(self):
        """Test function object if the request has a per-function scope."""
        if self.scope != "function":
            raise AttributeError(
                f"function not available in {self.scope}-scoped context"
            )
        return self._pyfuncitem.obj

    @property
    def cls(self):
        """Class (can be None) where the test function was collected."""
        if self.scope not in ("class", "function"):
            raise AttributeError(f"cls not available in {self.scope}-scoped context")
        clscol = self._pyfuncitem.getparent(_pytest.python.Class)
        if clscol:
            return clscol.obj

    @property
    def instance(self):
        """Instance (can be None) on which test function was collected."""
        if self.scope != "function":
            return None
        return getattr(self._pyfuncitem, "instance", None)

    @property
    def module(self):
        """Python module object where the test function was collected."""
        if self.scope not in ("function", "class", "module"):
            raise AttributeError(f"module not available in {self.scope}-scoped context")
        mod = self._pyfuncitem.getparent(_pytest.python.Module)
        assert mod is not None
        return mod.obj

    @property
    def path(self) -> Path:
        """Path where the test function was collected."""
        if self.scope not in ("function", "class", "module", "package"):
            raise AttributeError(f"path not available in {self.scope}-scoped context")
        return self._pyfuncitem.path

    @property
    def keywords(self) -> MutableMapping[str, Any]:
        """Keywords/markers dictionary for the underlying node."""
        node: nodes.Node = self.node
        return node.keywords

    @property
    def session(self) -> Session:
        """Pytest session object."""
        return self._pyfuncitem.session

    @abc.abstractmethod
    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        """Add finalizer/teardown function to be called without arguments after
        the last test within the requesting test context finished execution."""
        raise NotImplementedError()

    def applymarker(self, marker: str | MarkDecorator) -> None:
        """Apply a marker to a single test function invocation.

        This method is useful if you don't want to have a keyword/marker
        on all function invocations.

        :param marker:
            An object created by a call to ``pytest.mark.NAME(...)``.
        """
        self.node.add_marker(marker)

    def raiseerror(self, msg: str | None) -> NoReturn:
        """Raise a FixtureLookupError exception.

        :param msg:
            An optional custom error message.
        """
        raise FixtureLookupError(None, self, msg)

    def getfixturevalue(self, argname: str) -> Any:
        """Dynamically run a named fixture function.

        Declaring fixtures via function argument is recommended where possible.
        But if you can only decide whether to use another fixture at test
        setup time, you may use this function to retrieve it inside a fixture
        or test function body.

        This method can be used during the test setup phase or the test run
        phase, but during the test teardown phase a fixture's value may not
        be available.

        :param argname:
            The fixture name.
        :raises pytest.FixtureLookupError:
            If the given fixture could not be found.
        """
        # Note that in addition to the use case described in the docstring,
        # getfixturevalue() is also called by pytest itself during item and fixture
        # setup to evaluate the fixtures that are requested statically
        # (using function parameters, autouse, etc).

        fixturedef = self._get_active_fixturedef(argname)
        assert fixturedef.cached_result is not None, (
            f'The fixture value for "{argname}" is not available.  '
            "This can happen when the fixture has already been torn down."
        )
        return fixturedef.cached_result[0]

    def _iter_chain(self) -> Iterator[SubRequest]:
        """Yield all SubRequests in the chain, from self up.

        Note: does *not* yield the TopRequest.
        """
        current = self
        while isinstance(current, SubRequest):
            yield current
            current = current._parent_request

    def _get_active_fixturedef(
        self, argname: str
    ) -> FixtureDef[object] | PseudoFixtureDef[object]:
        if argname == "request":
            cached_result = (self, [0], None)
            return PseudoFixtureDef(cached_result, Scope.Function)

        # If we already finished computing a fixture by this name in this item,
        # return it.
        fixturedef = self._fixture_defs.get(argname)
        if fixturedef is not None:
            self._check_scope(fixturedef, fixturedef._scope)
            return fixturedef

        # Find the appropriate fixturedef.
        fixturedefs = self._arg2fixturedefs.get(argname, None)
        if fixturedefs is None:
            # We arrive here because of a dynamic call to
            # getfixturevalue(argname) which was naturally
            # not known at parsing/collection time.
            fixturedefs = self._fixturemanager.getfixturedefs(argname, self._pyfuncitem)
            if fixturedefs is not None:
                self._arg2fixturedefs[argname] = fixturedefs
        # No fixtures defined with this name.
        if fixturedefs is None:
            raise FixtureLookupError(argname, self)
        # The are no fixtures with this name applicable for the function.
        if not fixturedefs:
            raise FixtureLookupError(argname, self)
        # A fixture may override another fixture with the same name, e.g. a
        # fixture in a module can override a fixture in a conftest, a fixture in
        # a class can override a fixture in the module, and so on.
        # An overriding fixture can request its own name (possibly indirectly);
        # in this case it gets the value of the fixture it overrides, one level
        # up.
        # Check how many `argname`s deep we are, and take the next one.
        # `fixturedefs` is sorted from furthest to closest, so use negative
        # indexing to go in reverse.
        index = -1
        for request in self._iter_chain():
            if request.fixturename == argname:
                index -= 1
        # If already consumed all of the available levels, fail.
        if -index > len(fixturedefs):
            raise FixtureLookupError(argname, self)
        fixturedef = fixturedefs[index]

        # Prepare a SubRequest object for calling the fixture.
        try:
            callspec = self._pyfuncitem.callspec
        except AttributeError:
            callspec = None
        if callspec is not None and argname in callspec.params:
            param = callspec.params[argname]
            param_index = callspec.indices[argname]
            # The parametrize invocation scope overrides the fixture's scope.
            scope = callspec._arg2scope[argname]
        else:
            param = NOTSET
            param_index = 0
            scope = fixturedef._scope
            self._check_fixturedef_without_param(fixturedef)
        self._check_scope(fixturedef, scope)
        subrequest = SubRequest(
            self, scope, param, param_index, fixturedef, _ispytest=True
        )

        # Make sure the fixture value is cached, running it if it isn't
        fixturedef.execute(request=subrequest)

        self._fixture_defs[argname] = fixturedef
        return fixturedef

    def _check_fixturedef_without_param(self, fixturedef: FixtureDef[object]) -> None:
        """Check that this request is allowed to execute this fixturedef without
        a param."""
        funcitem = self._pyfuncitem
        has_params = fixturedef.params is not None
        fixtures_not_supported = getattr(funcitem, "nofuncargs", False)
        if has_params and fixtures_not_supported:
            msg = (
                f"{funcitem.name} does not support fixtures, maybe unittest.TestCase subclass?\n"
                f"Node id: {funcitem.nodeid}\n"
                f"Function type: {type(funcitem).__name__}"
            )
            fail(msg, pytrace=False)
        if has_params:
            frame = inspect.stack()[3]
            frameinfo = inspect.getframeinfo(frame[0])
            source_path = absolutepath(frameinfo.filename)
            source_lineno = frameinfo.lineno
            try:
                source_path_str = str(source_path.relative_to(funcitem.config.rootpath))
            except ValueError:
                source_path_str = str(source_path)
            location = getlocation(fixturedef.func, funcitem.config.rootpath)
            msg = (
                "The requested fixture has no parameter defined for test:\n"
                f"    {funcitem.nodeid}\n\n"
                f"Requested fixture '{fixturedef.argname}' defined in:\n"
                f"{location}\n\n"
                f"Requested here:\n"
                f"{source_path_str}:{source_lineno}"
            )
            fail(msg, pytrace=False)

    def _get_fixturestack(self) -> list[FixtureDef[Any]]:
        values = [request._fixturedef for request in self._iter_chain()]
        values.reverse()
        return values


@final
class TopRequest(FixtureRequest):
    """The type of the ``request`` fixture in a test function."""

    def __init__(self, pyfuncitem: Function, *, _ispytest: bool = False) -> None:
        super().__init__(
            fixturename=None,
            pyfuncitem=pyfuncitem,
            arg2fixturedefs=pyfuncitem._fixtureinfo.name2fixturedefs.copy(),
            fixture_defs={},
            _ispytest=_ispytest,
        )

    @property
    def _scope(self) -> Scope:
        return Scope.Function

    def _check_scope(
        self,
        requested_fixturedef: FixtureDef[object] | PseudoFixtureDef[object],
        requested_scope: Scope,
    ) -> None:
        # TopRequest always has function scope so always valid.
        pass

    @property
    def node(self):
        return self._pyfuncitem

    def __repr__(self) -> str:
        return f"<FixtureRequest for {self.node!r}>"

    def _fillfixtures(self) -> None:
        item = self._pyfuncitem
        for argname in item.fixturenames:
            if argname not in item.funcargs:
                item.funcargs[argname] = self.getfixturevalue(argname)

    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        self.node.addfinalizer(finalizer)


@final
class SubRequest(FixtureRequest):
    """The type of the ``request`` fixture in a fixture function requested
    (transitively) by a test function."""

    def __init__(
        self,
        request: FixtureRequest,
        scope: Scope,
        param: Any,
        param_index: int,
        fixturedef: FixtureDef[object],
        *,
        _ispytest: bool = False,
    ) -> None:
        super().__init__(
            pyfuncitem=request._pyfuncitem,
            fixturename=fixturedef.argname,
            fixture_defs=request._fixture_defs,
            arg2fixturedefs=request._arg2fixturedefs,
            _ispytest=_ispytest,
        )
        self._parent_request: Final[FixtureRequest] = request
        self._scope_field: Final = scope
        self._fixturedef: Final[FixtureDef[object]] = fixturedef
        if param is not NOTSET:
            self.param = param
        self.param_index: Final = param_index

    def __repr__(self) -> str:
        return f"<SubRequest {self.fixturename!r} for {self._pyfuncitem!r}>"

    @property
    def _scope(self) -> Scope:
        return self._scope_field

    @property
    def node(self):
        scope = self._scope
        if scope is Scope.Function:
            # This might also be a non-function Item despite its attribute name.
            node: nodes.Node | None = self._pyfuncitem
        elif scope is Scope.Package:
            node = get_scope_package(self._pyfuncitem, self._fixturedef)
        else:
            node = get_scope_node(self._pyfuncitem, scope)
        if node is None and scope is Scope.Class:
            # Fallback to function item itself.
            node = self._pyfuncitem
        assert node, f'Could not obtain a node for scope "{scope}" for function {self._pyfuncitem!r}'
        return node

    def _check_scope(
        self,
        requested_fixturedef: FixtureDef[object] | PseudoFixtureDef[object],
        requested_scope: Scope,
    ) -> None:
        if isinstance(requested_fixturedef, PseudoFixtureDef):
            return
        if self._scope > requested_scope:
            # Try to report something helpful.
            argname = requested_fixturedef.argname
            fixture_stack = "\n".join(
                self._format_fixturedef_line(fixturedef)
                for fixturedef in self._get_fixturestack()
            )
            requested_fixture = self._format_fixturedef_line(requested_fixturedef)
            fail(
                f"ScopeMismatch: You tried to access the {requested_scope.value} scoped "
                f"fixture {argname} with a {self._scope.value} scoped request object. "
                f"Requesting fixture stack:\n{fixture_stack}\n"
                f"Requested fixture:\n{requested_fixture}",
                pytrace=False,
            )

    def _format_fixturedef_line(self, fixturedef: FixtureDef[object]) -> str:
        factory = fixturedef.func
        path, lineno = getfslineno(factory)
        if isinstance(path, Path):
            path = bestrelpath(self._pyfuncitem.session.path, path)
        signature = inspect.signature(factory)
        return f"{path}:{lineno + 1}:  def {factory.__name__}{signature}"

    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        self._fixturedef.addfinalizer(finalizer)


@final
class FixtureLookupError(LookupError):
    """Could not return a requested fixture (missing or invalid)."""

    def __init__(
        self, argname: str | None, request: FixtureRequest, msg: str | None = None
    ) -> None:
        self.argname = argname
        self.request = request
        self.fixturestack = request._get_fixturestack()
        self.msg = msg

    def formatrepr(self) -> FixtureLookupErrorRepr:
        tblines: list[str] = []
        addline = tblines.append
        stack = [self.request._pyfuncitem.obj]
        stack.extend(map(lambda x: x.func, self.fixturestack))
        msg = self.msg
        if msg is not None:
            # The last fixture raise an error, let's present
            # it at the requesting side.
            stack = stack[:-1]
        for function in stack:
            fspath, lineno = getfslineno(function)
            try:
                lines, _ = inspect.getsourcelines(get_real_func(function))
            except (OSError, IndexError, TypeError):
                error_msg = "file %s, line %s: source code not available"
                addline(error_msg % (fspath, lineno + 1))
            else:
                addline(f"file {fspath}, line {lineno + 1}")
                for i, line in enumerate(lines):
                    line = line.rstrip()
                    addline("  " + line)
                    if line.lstrip().startswith("def"):
                        break

        if msg is None:
            fm = self.request._fixturemanager
            available = set()
            parent = self.request._pyfuncitem.parent
            assert parent is not None
            for name, fixturedefs in fm._arg2fixturedefs.items():
                faclist = list(fm._matchfactories(fixturedefs, parent))
                if faclist:
                    available.add(name)
            if self.argname in available:
                msg = (
                    f" recursive dependency involving fixture '{self.argname}' detected"
                )
            else:
                msg = f"fixture '{self.argname}' not found"
            msg += "\n available fixtures: {}".format(", ".join(sorted(available)))
            msg += "\n use 'pytest --fixtures [testpath]' for help on them."

        return FixtureLookupErrorRepr(fspath, lineno, tblines, msg, self.argname)


class FixtureLookupErrorRepr(TerminalRepr):
    def __init__(
        self,
        filename: str | os.PathLike[str],
        firstlineno: int,
        tblines: Sequence[str],
        errorstring: str,
        argname: str | None,
    ) -> None:
        self.tblines = tblines
        self.errorstring = errorstring
        self.filename = filename
        self.firstlineno = firstlineno
        self.argname = argname

    def toterminal(self, tw: TerminalWriter) -> None:
        # tw.line("FixtureLookupError: %s" %(self.argname), red=True)
        for tbline in self.tblines:
            tw.line(tbline.rstrip())
        lines = self.errorstring.split("\n")
        if lines:
            tw.line(
                f"{FormattedExcinfo.fail_marker}       {lines[0].strip()}",
                red=True,
            )
            for line in lines[1:]:
                tw.line(
                    f"{FormattedExcinfo.flow_marker}       {line.strip()}",
                    red=True,
                )
        tw.line()
        tw.line("%s:%d" % (os.fspath(self.filename), self.firstlineno + 1))


def call_fixture_func(
    fixturefunc: _FixtureFunc[FixtureValue], request: FixtureRequest, kwargs
) -> FixtureValue:
    if is_generator(fixturefunc):
        fixturefunc = cast(
            Callable[..., Generator[FixtureValue, None, None]], fixturefunc
        )
        generator = fixturefunc(**kwargs)
        try:
            fixture_result = next(generator)
        except StopIteration:
            raise ValueError(f"{request.fixturename} did not yield a value") from None
        finalizer = functools.partial(_teardown_yield_fixture, fixturefunc, generator)
        request.addfinalizer(finalizer)
    else:
        fixturefunc = cast(Callable[..., FixtureValue], fixturefunc)
        fixture_result = fixturefunc(**kwargs)
    return fixture_result


def _teardown_yield_fixture(fixturefunc, it) -> None:
    """Execute the teardown of a fixture function by advancing the iterator
    after the yield and ensure the iteration ends (if not it means there is
    more than one yield in the function)."""
    try:
        next(it)
    except StopIteration:
        pass
    else:
        fs, lineno = getfslineno(fixturefunc)
        fail(
            f"fixture function has more than one 'yield':\n\n"
            f"{Source(fixturefunc).indent()}\n"
            f"{fs}:{lineno + 1}",
            pytrace=False,
        )


def _eval_scope_callable(
    scope_callable: Callable[[str, Config], _ScopeName],
    fixture_name: str,
    config: Config,
) -> _ScopeName:
    try:
        # Type ignored because there is no typing mechanism to specify
        # keyword arguments, currently.
        result = scope_callable(fixture_name=fixture_name, config=config)  # type: ignore[call-arg]
    except Exception as e:
        raise TypeError(
            f"Error evaluating {scope_callable} while defining fixture '{fixture_name}'.\n"
            "Expected a function with the signature (*, fixture_name, config)"
        ) from e
    if not isinstance(result, str):
        fail(
            f"Expected {scope_callable} to return a 'str' while defining fixture '{fixture_name}', but it returned:\n"
            f"{result!r}",
            pytrace=False,
        )
    return result


@final
class FixtureDef(Generic[FixtureValue]):
    """A container for a fixture definition.

    Note: At this time, only explicitly documented fields and methods are
    considered public stable API.
    """

    def __init__(
        self,
        config: Config,
        baseid: str | None,
        argname: str,
        func: _FixtureFunc[FixtureValue],
        scope: Scope | _ScopeName | Callable[[str, Config], _ScopeName] | None,
        params: Sequence[object] | None,
        ids: tuple[object | None, ...] | Callable[[Any], object | None] | None = None,
        *,
        _ispytest: bool = False,
    ) -> None:
        check_ispytest(_ispytest)
        # The "base" node ID for the fixture.
        #
        # This is a node ID prefix. A fixture is only available to a node (e.g.
        # a `Function` item) if the fixture's baseid is a nodeid of a parent of
        # node.
        #
        # For a fixture found in a Collector's object (e.g. a `Module`s module,
        # a `Class`'s class), the baseid is the Collector's nodeid.
        #
        # For a fixture found in a conftest plugin, the baseid is the conftest's
        # directory path relative to the rootdir.
        #
        # For other plugins, the baseid is the empty string (always matches).
        self.baseid: Final = baseid or ""
        # Whether the fixture was found from a node or a conftest in the
        # collection tree. Will be false for fixtures defined in non-conftest
        # plugins.
        self.has_location: Final = baseid is not None
        # The fixture factory function.
        self.func: Final = func
        # The name by which the fixture may be requested.
        self.argname: Final = argname
        if scope is None:
            scope = Scope.Function
        elif callable(scope):
            scope = _eval_scope_callable(scope, argname, config)
        if isinstance(scope, str):
            scope = Scope.from_user(
                scope, descr=f"Fixture '{func.__name__}'", where=baseid
            )
        self._scope: Final = scope
        # If the fixture is directly parametrized, the parameter values.
        self.params: Final = params
        # If the fixture is directly parametrized, a tuple of explicit IDs to
        # assign to the parameter values, or a callable to generate an ID given
        # a parameter value.
        self.ids: Final = ids
        # The names requested by the fixtures.
        self.argnames: Final = getfuncargnames(func, name=argname)
        # If the fixture was executed, the current value of the fixture.
        # Can change if the fixture is executed with different parameters.
        self.cached_result: _FixtureCachedResult[FixtureValue] | None = None
        self._finalizers: Final[list[Callable[[], object]]] = []

    @property
    def scope(self) -> _ScopeName:
        """Scope string, one of "function", "class", "module", "package", "session"."""
        return self._scope.value

    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        self._finalizers.append(finalizer)

    def finish(self, request: SubRequest) -> None:
        exceptions: list[BaseException] = []
        while self._finalizers:
            fin = self._finalizers.pop()
            try:
                fin()
            except BaseException as e:
                exceptions.append(e)
        node = request.node
        node.ihook.pytest_fixture_post_finalizer(fixturedef=self, request=request)
        # Even if finalization fails, we invalidate the cached fixture
        # value and remove all finalizers because they may be bound methods
        # which will keep instances alive.
        self.cached_result = None
        self._finalizers.clear()
        if len(exceptions) == 1:
            raise exceptions[0]
        elif len(exceptions) > 1:
            msg = f'errors while tearing down fixture "{self.argname}" of {node}'
            raise BaseExceptionGroup(msg, exceptions[::-1])

    def execute(self, request: SubRequest) -> FixtureValue:
        """Return the value of this fixture, executing it if not cached."""
        # Ensure that the dependent fixtures requested by this fixture are loaded.
        # This needs to be done before checking if we have a cached value, since
        # if a dependent fixture has their cache invalidated, e.g. due to
        # parametrization, they finalize themselves and fixtures depending on it
        # (which will likely include this fixture) setting `self.cached_result = None`.
        # See #4871
        requested_fixtures_that_should_finalize_us = []
        for argname in self.argnames:
            fixturedef = request._get_active_fixturedef(argname)
            # Saves requested fixtures in a list so we later can add our finalizer
            # to them, ensuring that if a requested fixture gets torn down we get torn
            # down first. This is generally handled by SetupState, but still currently
            # needed when this fixture is not parametrized but depends on a parametrized
            # fixture.
            if not isinstance(fixturedef, PseudoFixtureDef):
                requested_fixtures_that_should_finalize_us.append(fixturedef)

        # Check for (and return) cached value/exception.
        if self.cached_result is not None:
            request_cache_key = self.cache_key(request)
            cache_key = self.cached_result[1]
            try:
                # Attempt to make a normal == check: this might fail for objects
                # which do not implement the standard comparison (like numpy arrays -- #6497).
                cache_hit = bool(request_cache_key == cache_key)
            except (ValueError, RuntimeError):
                # If the comparison raises, use 'is' as fallback.
                cache_hit = request_cache_key is cache_key

            if cache_hit:
                if self.cached_result[2] is not None:
                    exc, exc_tb = self.cached_result[2]
                    raise exc.with_traceback(exc_tb)
                else:
                    result = self.cached_result[0]
                    return result
            # We have a previous but differently parametrized fixture instance
            # so we need to tear it down before creating a new one.
            self.finish(request)
            assert self.cached_result is None

        # Add finalizer to requested fixtures we saved previously.
        # We make sure to do this after checking for cached value to avoid
        # adding our finalizer multiple times. (#12135)
        finalizer = functools.partial(self.finish, request=request)
        for parent_fixture in requested_fixtures_that_should_finalize_us:
            parent_fixture.addfinalizer(finalizer)

        ihook = request.node.ihook
        try:
            # Setup the fixture, run the code in it, and cache the value
            # in self.cached_result
            result = ihook.pytest_fixture_setup(fixturedef=self, request=request)
        finally:
            # schedule our finalizer, even if the setup failed
            request.node.addfinalizer(finalizer)

        return result

    def cache_key(self, request: SubRequest) -> object:
        return getattr(request, "param", None)

    def __repr__(self) -> str:
        return f"<FixtureDef argname={self.argname!r} scope={self.scope!r} baseid={self.baseid!r}>"


def resolve_fixture_function(
    fixturedef: FixtureDef[FixtureValue], request: FixtureRequest
) -> _FixtureFunc[FixtureValue]:
    """Get the actual callable that can be called to obtain the fixture
    value."""
    fixturefunc = fixturedef.func
    # The fixture function needs to be bound to the actual
    # request.instance so that code working with "fixturedef" behaves
    # as expected.
    instance = request.instance
    if instance is not None:
        # Handle the case where fixture is defined not in a test class, but some other class
        # (for example a plugin class with a fixture), see #2270.
        if hasattr(fixturefunc, "__self__") and not isinstance(
            instance,
            fixturefunc.__self__.__class__,
        ):
            return fixturefunc
        fixturefunc = getimfunc(fixturedef.func)
        if fixturefunc != fixturedef.func:
            fixturefunc = fixturefunc.__get__(instance)
    return fixturefunc


def pytest_fixture_setup(
    fixturedef: FixtureDef[FixtureValue], request: SubRequest
) -> FixtureValue:
    """Execution of fixture setup."""
    kwargs = {}
    for argname in fixturedef.argnames:
        kwargs[argname] = request.getfixturevalue(argname)

    fixturefunc = resolve_fixture_function(fixturedef, request)
    my_cache_key = fixturedef.cache_key(request)
    try:
        result = call_fixture_func(fixturefunc, request, kwargs)
    except TEST_OUTCOME as e:
        if isinstance(e, skip.Exception):
            # The test requested a fixture which caused a skip.
            # Don't show the fixture as the skip location, as then the user
            # wouldn't know which test skipped.
            e._use_item_location = True
        fixturedef.cached_result = (None, my_cache_key, (e, e.__traceback__))
        raise
    fixturedef.cached_result = (result, my_cache_key, None)
    return result


def wrap_function_to_error_out_if_called_directly(
    function: FixtureFunction,
    fixture_marker: FixtureFunctionMarker,
) -> FixtureFunction:
    """Wrap the given fixture function so we can raise an error about it being called directly,
    instead of used as an argument in a test function."""
    name = fixture_marker.name or function.__name__
    message = (
        f'Fixture "{name}" called directly. Fixtures are not meant to be called directly,\n'
        "but are created automatically when test functions request them as parameters.\n"
        "See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and\n"
        "https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code."
    )

    @functools.wraps(function)
    def result(*args, **kwargs):
        fail(message, pytrace=False)

    # Keep reference to the original function in our own custom attribute so we don't unwrap
    # further than this point and lose useful wrappings like @mock.patch (#3774).
    result.__pytest_wrapped__ = _PytestWrapper(function)  # type: ignore[attr-defined]

    return cast(FixtureFunction, result)


@final
@dataclasses.dataclass(frozen=True)
class FixtureFunctionMarker:
    scope: _ScopeName | Callable[[str, Config], _ScopeName]
    params: tuple[object, ...] | None
    autouse: bool = False
    ids: tuple[object | None, ...] | Callable[[Any], object | None] | None = None
    name: str | None = None

    _ispytest: dataclasses.InitVar[bool] = False

    def __post_init__(self, _ispytest: bool) -> None:
        check_ispytest(_ispytest)

    def __call__(self, function: FixtureFunction) -> FixtureFunction:
        if inspect.isclass(function):
            raise ValueError("class fixtures not supported (maybe in the future)")

        if getattr(function, "_pytestfixturefunction", False):
            raise ValueError(
                f"@pytest.fixture is being applied more than once to the same function {function.__name__!r}"
            )

        if hasattr(function, "pytestmark"):
            warnings.warn(MARKED_FIXTURE, stacklevel=2)

        function = wrap_function_to_error_out_if_called_directly(function, self)

        name = self.name or function.__name__
        if name == "request":
            location = getlocation(function)
            fail(
                f"'request' is a reserved word for fixtures, use another name:\n  {location}",
                pytrace=False,
            )

        # Type ignored because https://github.com/python/mypy/issues/2087.
        function._pytestfixturefunction = self  # type: ignore[attr-defined]
        return function


@overload
def fixture(
    fixture_function: FixtureFunction,
    *,
    scope: _ScopeName | Callable[[str, Config], _ScopeName] = ...,
    params: Iterable[object] | None = ...,
    autouse: bool = ...,
    ids: Sequence[object | None] | Callable[[Any], object | None] | None = ...,
    name: str | None = ...,
) -> FixtureFunction: ...


@overload
def fixture(
    fixture_function: None = ...,
    *,
    scope: _ScopeName | Callable[[str, Config], _ScopeName] = ...,
    params: Iterable[object] | None = ...,
    autouse: bool = ...,
    ids: Sequence[object | None] | Callable[[Any], object | None] | None = ...,
    name: str | None = None,
) -> FixtureFunctionMarker: ...


def fixture(
    fixture_function: FixtureFunction | None = None,
    *,
    scope: _ScopeName | Callable[[str, Config], _ScopeName] = "function",
    params: Iterable[object] | None = None,
    autouse: bool = False,
    ids: Sequence[object | None] | Callable[[Any], object | None] | None = None,
    name: str | None = None,
) -> FixtureFunctionMarker | FixtureFunction:
    """Decorator to mark a fixture factory function.

    This decorator can be used, with or without parameters, to define a
    fixture function.

    The name of the fixture function can later be referenced to cause its
    invocation ahead of running tests: test modules or classes can use the
    ``pytest.mark.usefixtures(fixturename)`` marker.

    Test functions can directly use fixture names as input arguments in which
    case the fixture instance returned from the fixture function will be
    injected.

    Fixtures can provide their values to test functions using ``return`` or
    ``yield`` statements. When using ``yield`` the code block after the
    ``yield`` statement is executed as teardown code regardless of the test
    outcome, and must yield exactly once.

    :param scope:
        The scope for which this fixture is shared; one of ``"function"``
        (default), ``"class"``, ``"module"``, ``"package"`` or ``"session"``.

        This parameter may also be a callable which receives ``(fixture_name, config)``
        as parameters, and must return a ``str`` with one of the values mentioned above.

        See :ref:`dynamic scope` in the docs for more information.

    :param params:
        An optional list of parameters which will cause multiple invocations
        of the fixture function and all of the tests using it. The current
        parameter is available in ``request.param``.

    :param autouse:
        If True, the fixture func is activated for all tests that can see it.
        If False (the default), an explicit reference is needed to activate
        the fixture.

    :param ids:
        Sequence of ids each corresponding to the params so that they are
        part of the test id. If no ids are provided they will be generated
        automatically from the params.

    :param name:
        The name of the fixture. This defaults to the name of the decorated
        function. If a fixture is used in the same module in which it is
        defined, the function name of the fixture will be shadowed by the
        function arg that requests the fixture; one way to resolve this is to
        name the decorated function ``fixture_<fixturename>`` and then use
        ``@pytest.fixture(name='<fixturename>')``.
    """
    fixture_marker = FixtureFunctionMarker(
        scope=scope,
        params=tuple(params) if params is not None else None,
        autouse=autouse,
        ids=None if ids is None else ids if callable(ids) else tuple(ids),
        name=name,
        _ispytest=True,
    )

    # Direct decoration.
    if fixture_function:
        return fixture_marker(fixture_function)

    return fixture_marker


def yield_fixture(
    fixture_function=None,
    *args,
    scope="function",
    params=None,
    autouse=False,
    ids=None,
    name=None,
):
    """(Return a) decorator to mark a yield-fixture factory function.

    .. deprecated:: 3.0
        Use :py:func:`pytest.fixture` directly instead.
    """
    warnings.warn(YIELD_FIXTURE, stacklevel=2)
    return fixture(
        fixture_function,
        *args,
        scope=scope,
        params=params,
        autouse=autouse,
        ids=ids,
        name=name,
    )


@fixture(scope="session")
def pytestconfig(request: FixtureRequest) -> Config:
    """Session-scoped fixture that returns the session's :class:`pytest.Config`
    object.

    Example::

        def test_foo(pytestconfig):
            if pytestconfig.get_verbosity() > 0:
                ...

    """
    return request.config


def pytest_addoption(parser: Parser) -> None:
    parser.addini(
        "usefixtures",
        type="args",
        default=[],
        help="List of default fixtures to be used with this project",
    )
    group = parser.getgroup("general")
    group.addoption(
        "--fixtures",
        "--funcargs",
        action="store_true",
        dest="showfixtures",
        default=False,
        help="Show available fixtures, sorted by plugin appearance "
        "(fixtures with leading '_' are only shown with '-v')",
    )
    group.addoption(
        "--fixtures-per-test",
        action="store_true",
        dest="show_fixtures_per_test",
        default=False,
        help="Show fixtures per test",
    )


def pytest_cmdline_main(config: Config) -> int | ExitCode | None:
    if config.option.showfixtures:
        showfixtures(config)
        return 0
    if config.option.show_fixtures_per_test:
        show_fixtures_per_test(config)
        return 0
    return None


def _get_direct_parametrize_args(node: nodes.Node) -> set[str]:
    """Return all direct parametrization arguments of a node, so we don't
    mistake them for fixtures.

    Check https://github.com/pytest-dev/pytest/issues/5036.

    These things are done later as well when dealing with parametrization
    so this could be improved.
    """
    parametrize_argnames: set[str] = set()
    for marker in node.iter_markers(name="parametrize"):
        if not marker.kwargs.get("indirect", False):
            p_argnames, _ = ParameterSet._parse_parametrize_args(
                *marker.args, **marker.kwargs
            )
            parametrize_argnames.update(p_argnames)
    return parametrize_argnames


def deduplicate_names(*seqs: Iterable[str]) -> tuple[str, ...]:
    """De-duplicate the sequence of names while keeping the original order."""
    # Ideally we would use a set, but it does not preserve insertion order.
    return tuple(dict.fromkeys(name for seq in seqs for name in seq))


class FixtureManager:
    """pytest fixture definitions and information is stored and managed
    from this class.

    During collection fm.parsefactories() is called multiple times to parse
    fixture function definitions into FixtureDef objects and internal
    data structures.

    During collection of test functions, metafunc-mechanics instantiate
    a FuncFixtureInfo object which is cached per node/func-name.
    This FuncFixtureInfo object is later retrieved by Function nodes
    which themselves offer a fixturenames attribute.

    The FuncFixtureInfo object holds information about fixtures and FixtureDefs
    relevant for a particular function. An initial list of fixtures is
    assembled like this:

    - ini-defined usefixtures
    - autouse-marked fixtures along the collection chain up from the function
    - usefixtures markers at module/class/function level
    - test function funcargs

    Subsequently the funcfixtureinfo.fixturenames attribute is computed
    as the closure of the fixtures needed to setup the initial fixtures,
    i.e. fixtures needed by fixture functions themselves are appended
    to the fixturenames list.

    Upon the test-setup phases all fixturenames are instantiated, retrieved
    by a lookup of their FuncFixtureInfo.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.config: Config = session.config
        # Maps a fixture name (argname) to all of the FixtureDefs in the test
        # suite/plugins defined with this name. Populated by parsefactories().
        # TODO: The order of the FixtureDefs list of each arg is significant,
        #       explain.
        self._arg2fixturedefs: Final[dict[str, list[FixtureDef[Any]]]] = {}
        self._holderobjseen: Final[set[object]] = set()
        # A mapping from a nodeid to a list of autouse fixtures it defines.
        self._nodeid_autousenames: Final[dict[str, list[str]]] = {
            "": self.config.getini("usefixtures"),
        }
        session.config.pluginmanager.register(self, "funcmanage")

    def getfixtureinfo(
        self,
        node: nodes.Item,
        func: Callable[..., object] | None,
        cls: type | None,
    ) -> FuncFixtureInfo:
        """Calculate the :class:`FuncFixtureInfo` for an item.

        If ``func`` is None, or if the item sets an attribute
        ``nofuncargs = True``, then ``func`` is not examined at all.

        :param node:
            The item requesting the fixtures.
        :param func:
            The item's function.
        :param cls:
            If the function is a method, the method's class.
        """
        if func is not None and not getattr(node, "nofuncargs", False):
            argnames = getfuncargnames(func, name=node.name, cls=cls)
        else:
            argnames = ()
        usefixturesnames = self._getusefixturesnames(node)
        autousenames = self._getautousenames(node)
        initialnames = deduplicate_names(autousenames, usefixturesnames, argnames)

        direct_parametrize_args = _get_direct_parametrize_args(node)

        names_closure, arg2fixturedefs = self.getfixtureclosure(
            parentnode=node,
            initialnames=initialnames,
            ignore_args=direct_parametrize_args,
        )

        return FuncFixtureInfo(argnames, initialnames, names_closure, arg2fixturedefs)

    def pytest_plugin_registered(self, plugin: _PluggyPlugin, plugin_name: str) -> None:
        # Fixtures defined in conftest plugins are only visible to within the
        # conftest's directory. This is unlike fixtures in non-conftest plugins
        # which have global visibility. So for conftests, construct the base
        # nodeid from the plugin name (which is the conftest path).
        if plugin_name and plugin_name.endswith("conftest.py"):
            # Note: we explicitly do *not* use `plugin.__file__` here -- The
            # difference is that plugin_name has the correct capitalization on
            # case-insensitive systems (Windows) and other normalization issues
            # (issue #11816).
            conftestpath = absolutepath(plugin_name)
            try:
                nodeid = str(conftestpath.parent.relative_to(self.config.rootpath))
            except ValueError:
                nodeid = ""
            if nodeid == ".":
                nodeid = ""
            if os.sep != nodes.SEP:
                nodeid = nodeid.replace(os.sep, nodes.SEP)
        else:
            nodeid = None

        self.parsefactories(plugin, nodeid)

    def _getautousenames(self, node: nodes.Node) -> Iterator[str]:
        """Return the names of autouse fixtures applicable to node."""
        for parentnode in node.listchain():
            basenames = self._nodeid_autousenames.get(parentnode.nodeid)
            if basenames:
                yield from basenames

    def _getusefixturesnames(self, node: nodes.Item) -> Iterator[str]:
        """Return the names of usefixtures fixtures applicable to node."""
        for mark in node.iter_markers(name="usefixtures"):
            yield from mark.args

    def getfixtureclosure(
        self,
        parentnode: nodes.Node,
        initialnames: tuple[str, ...],
        ignore_args: AbstractSet[str],
    ) -> tuple[list[str], dict[str, Sequence[FixtureDef[Any]]]]:
        # Collect the closure of all fixtures, starting with the given
        # fixturenames as the initial set.  As we have to visit all
        # factory definitions anyway, we also return an arg2fixturedefs
        # mapping so that the caller can reuse it and does not have
        # to re-discover fixturedefs again for each fixturename
        # (discovering matching fixtures for a given name/node is expensive).

        fixturenames_closure = list(initialnames)

        arg2fixturedefs: dict[str, Sequence[FixtureDef[Any]]] = {}
        lastlen = -1
        while lastlen != len(fixturenames_closure):
            lastlen = len(fixturenames_closure)
            for argname in fixturenames_closure:
                if argname in ignore_args:
                    continue
                if argname in arg2fixturedefs:
                    continue
                fixturedefs = self.getfixturedefs(argname, parentnode)
                if fixturedefs:
                    arg2fixturedefs[argname] = fixturedefs
                    for arg in fixturedefs[-1].argnames:
                        if arg not in fixturenames_closure:
                            fixturenames_closure.append(arg)

        def sort_by_scope(arg_name: str) -> Scope:
            try:
                fixturedefs = arg2fixturedefs[arg_name]
            except KeyError:
                return Scope.Function
            else:
                return fixturedefs[-1]._scope

        fixturenames_closure.sort(key=sort_by_scope, reverse=True)
        return fixturenames_closure, arg2fixturedefs

    def pytest_generate_tests(self, metafunc: Metafunc) -> None:
        """Generate new tests based on parametrized fixtures used by the given metafunc"""

        def get_parametrize_mark_argnames(mark: Mark) -> Sequence[str]:
            args, _ = ParameterSet._parse_parametrize_args(*mark.args, **mark.kwargs)
            return args

        for argname in metafunc.fixturenames:
            # Get the FixtureDefs for the argname.
            fixture_defs = metafunc._arg2fixturedefs.get(argname)
            if not fixture_defs:
                # Will raise FixtureLookupError at setup time if not parametrized somewhere
                # else (e.g @pytest.mark.parametrize)
                continue

            # If the test itself parametrizes using this argname, give it
            # precedence.
            if any(
                argname in get_parametrize_mark_argnames(mark)
                for mark in metafunc.definition.iter_markers("parametrize")
            ):
                continue

            # In the common case we only look at the fixture def with the
            # closest scope (last in the list). But if the fixture overrides
            # another fixture, while requesting the super fixture, keep going
            # in case the super fixture is parametrized (#1953).
            for fixturedef in reversed(fixture_defs):
                # Fixture is parametrized, apply it and stop.
                if fixturedef.params is not None:
                    metafunc.parametrize(
                        argname,
                        fixturedef.params,
                        indirect=True,
                        scope=fixturedef.scope,
                        ids=fixturedef.ids,
                    )
                    break

                # Not requesting the overridden super fixture, stop.
                if argname not in fixturedef.argnames:
                    break

                # Try next super fixture, if any.

    def pytest_collection_modifyitems(self, items: list[nodes.Item]) -> None:
        # Separate parametrized setups.
        items[:] = reorder_items(items)

    def _register_fixture(
        self,
        *,
        name: str,
        func: _FixtureFunc[object],
        nodeid: str | None,
        scope: Scope | _ScopeName | Callable[[str, Config], _ScopeName] = "function",
        params: Sequence[object] | None = None,
        ids: tuple[object | None, ...] | Callable[[Any], object | None] | None = None,
        autouse: bool = False,
    ) -> None:
        """Register a fixture

        :param name:
            The fixture's name.
        :param func:
            The fixture's implementation function.
        :param nodeid:
            The visibility of the fixture. The fixture will be available to the
            node with this nodeid and its children in the collection tree.
            None means that the fixture is visible to the entire collection tree,
            e.g. a fixture defined for general use in a plugin.
        :param scope:
            The fixture's scope.
        :param params:
            The fixture's parametrization params.
        :param ids:
            The fixture's IDs.
        :param autouse:
            Whether this is an autouse fixture.
        """
        fixture_def = FixtureDef(
            config=self.config,
            baseid=nodeid,
            argname=name,
            func=func,
            scope=scope,
            params=params,
            ids=ids,
            _ispytest=True,
        )

        faclist = self._arg2fixturedefs.setdefault(name, [])
        if fixture_def.has_location:
            faclist.append(fixture_def)
        else:
            # fixturedefs with no location are at the front
            # so this inserts the current fixturedef after the
            # existing fixturedefs from external plugins but
            # before the fixturedefs provided in conftests.
            i = len([f for f in faclist if not f.has_location])
            faclist.insert(i, fixture_def)
        if autouse:
            self._nodeid_autousenames.setdefault(nodeid or "", []).append(name)

    @overload
    def parsefactories(
        self,
        node_or_obj: nodes.Node,
    ) -> None:
        raise NotImplementedError()

    @overload
    def parsefactories(
        self,
        node_or_obj: object,
        nodeid: str | None,
    ) -> None:
        raise NotImplementedError()

    def parsefactories(
        self,
        node_or_obj: nodes.Node | object,
        nodeid: str | NotSetType | None = NOTSET,
    ) -> None:
        """Collect fixtures from a collection node or object.

        Found fixtures are parsed into `FixtureDef`s and saved.

        If `node_or_object` is a collection node (with an underlying Python
        object), the node's object is traversed and the node's nodeid is used to
        determine the fixtures' visibility. `nodeid` must not be specified in
        this case.

        If `node_or_object` is an object (e.g. a plugin), the object is
        traversed and the given `nodeid` is used to determine the fixtures'
        visibility. `nodeid` must be specified in this case; None and "" mean
        total visibility.
        """
        if nodeid is not NOTSET:
            holderobj = node_or_obj
        else:
            assert isinstance(node_or_obj, nodes.Node)
            holderobj = cast(object, node_or_obj.obj)  # type: ignore[attr-defined]
            assert isinstance(node_or_obj.nodeid, str)
            nodeid = node_or_obj.nodeid
        if holderobj in self._holderobjseen:
            return

        # Avoid accessing `@property` (and other descriptors) when iterating fixtures.
        if not safe_isclass(holderobj) and not isinstance(holderobj, types.ModuleType):
            holderobj_tp: object = type(holderobj)
        else:
            holderobj_tp = holderobj

        self._holderobjseen.add(holderobj)
        for name in dir(holderobj):
            # The attribute can be an arbitrary descriptor, so the attribute
            # access below can raise. safe_getattr() ignores such exceptions.
            obj_ub = safe_getattr(holderobj_tp, name, None)
            marker = getfixturemarker(obj_ub)
            if not isinstance(marker, FixtureFunctionMarker):
                # Magic globals  with __getattr__ might have got us a wrong
                # fixture attribute.
                continue

            # OK we know it is a fixture -- now safe to look up on the _instance_.
            obj = getattr(holderobj, name)

            if marker.name:
                name = marker.name

            # During fixture definition we wrap the original fixture function
            # to issue a warning if called directly, so here we unwrap it in
            # order to not emit the warning when pytest itself calls the
            # fixture function.
            func = get_real_method(obj, holderobj)

            self._register_fixture(
                name=name,
                nodeid=nodeid,
                func=func,
                scope=marker.scope,
                params=marker.params,
                ids=marker.ids,
                autouse=marker.autouse,
            )

    def getfixturedefs(
        self, argname: str, node: nodes.Node
    ) -> Sequence[FixtureDef[Any]] | None:
        """Get FixtureDefs for a fixture name which are applicable
        to a given node.

        Returns None if there are no fixtures at all defined with the given
        name. (This is different from the case in which there are fixtures
        with the given name, but none applicable to the node. In this case,
        an empty result is returned).

        :param argname: Name of the fixture to search for.
        :param node: The requesting Node.
        """
        try:
            fixturedefs = self._arg2fixturedefs[argname]
        except KeyError:
            return None
        return tuple(self._matchfactories(fixturedefs, node))

    def _matchfactories(
        self, fixturedefs: Iterable[FixtureDef[Any]], node: nodes.Node
    ) -> Iterator[FixtureDef[Any]]:
        parentnodeids = {n.nodeid for n in node.iter_parents()}
        for fixturedef in fixturedefs:
            if fixturedef.baseid in parentnodeids:
                yield fixturedef


def show_fixtures_per_test(config: Config) -> int | ExitCode:
    from _pytest.main import wrap_session

    return wrap_session(config, _show_fixtures_per_test)


_PYTEST_DIR = Path(_pytest.__file__).parent


def _pretty_fixture_path(invocation_dir: Path, func) -> str:
    loc = Path(getlocation(func, invocation_dir))
    prefix = Path("...", "_pytest")
    try:
        return str(prefix / loc.relative_to(_PYTEST_DIR))
    except ValueError:
        return bestrelpath(invocation_dir, loc)


def _show_fixtures_per_test(config: Config, session: Session) -> None:
    import _pytest.config

    session.perform_collect()
    invocation_dir = config.invocation_params.dir
    tw = _pytest.config.create_terminal_writer(config)
    verbose = config.get_verbosity()

    def get_best_relpath(func) -> str:
        loc = getlocation(func, invocation_dir)
        return bestrelpath(invocation_dir, Path(loc))

    def write_fixture(fixture_def: FixtureDef[object]) -> None:
        argname = fixture_def.argname
        if verbose <= 0 and argname.startswith("_"):
            return
        prettypath = _pretty_fixture_path(invocation_dir, fixture_def.func)
        tw.write(f"{argname}", green=True)
        tw.write(f" -- {prettypath}", yellow=True)
        tw.write("\n")
        fixture_doc = inspect.getdoc(fixture_def.func)
        if fixture_doc:
            write_docstring(
                tw,
                fixture_doc.split("\n\n", maxsplit=1)[0]
                if verbose <= 0
                else fixture_doc,
            )
        else:
            tw.line("    no docstring available", red=True)

    def write_item(item: nodes.Item) -> None:
        # Not all items have _fixtureinfo attribute.
        info: FuncFixtureInfo | None = getattr(item, "_fixtureinfo", None)
        if info is None or not info.name2fixturedefs:
            # This test item does not use any fixtures.
            return
        tw.line()
        tw.sep("-", f"fixtures used by {item.name}")
        # TODO: Fix this type ignore.
        tw.sep("-", f"({get_best_relpath(item.function)})")  # type: ignore[attr-defined]
        # dict key not used in loop but needed for sorting.
        for _, fixturedefs in sorted(info.name2fixturedefs.items()):
            assert fixturedefs is not None
            if not fixturedefs:
                continue
            # Last item is expected to be the one used by the test item.
            write_fixture(fixturedefs[-1])

    for session_item in session.items:
        write_item(session_item)


def showfixtures(config: Config) -> int | ExitCode:
    from _pytest.main import wrap_session

    return wrap_session(config, _showfixtures_main)


def _showfixtures_main(config: Config, session: Session) -> None:
    import _pytest.config

    session.perform_collect()
    invocation_dir = config.invocation_params.dir
    tw = _pytest.config.create_terminal_writer(config)
    verbose = config.get_verbosity()

    fm = session._fixturemanager

    available = []
    seen: set[tuple[str, str]] = set()

    for argname, fixturedefs in fm._arg2fixturedefs.items():
        assert fixturedefs is not None
        if not fixturedefs:
            continue
        for fixturedef in fixturedefs:
            loc = getlocation(fixturedef.func, invocation_dir)
            if (fixturedef.argname, loc) in seen:
                continue
            seen.add((fixturedef.argname, loc))
            available.append(
                (
                    len(fixturedef.baseid),
                    fixturedef.func.__module__,
                    _pretty_fixture_path(invocation_dir, fixturedef.func),
                    fixturedef.argname,
                    fixturedef,
                )
            )

    available.sort()
    currentmodule = None
    for baseid, module, prettypath, argname, fixturedef in available:
        if currentmodule != module:
            if not module.startswith("_pytest."):
                tw.line()
                tw.sep("-", f"fixtures defined from {module}")
                currentmodule = module
        if verbose <= 0 and argname.startswith("_"):
            continue
        tw.write(f"{argname}", green=True)
        if fixturedef.scope != "function":
            tw.write(f" [{fixturedef.scope} scope]", cyan=True)
        tw.write(f" -- {prettypath}", yellow=True)
        tw.write("\n")
        doc = inspect.getdoc(fixturedef.func)
        if doc:
            write_docstring(
                tw, doc.split("\n\n", maxsplit=1)[0] if verbose <= 0 else doc
            )
        else:
            tw.line("    no docstring available", red=True)
        tw.line()


def write_docstring(tw: TerminalWriter, doc: str, indent: str = "    ") -> None:
    for line in doc.split("\n"):
        tw.line(indent + line)
