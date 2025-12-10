# Notes about NetworkX namespace objects set up here:
#
# nx.utils.backends.backends:
#   dict keyed by backend name to the backend entry point object.
#   Filled using ``_get_backends("networkx.backends")`` during import of this module.
#
# nx.utils.backends.backend_info:
#   dict keyed by backend name to the metadata returned by the function indicated
#   by the "networkx.backend_info" entry point.
#   Created as an empty dict while importing this module, but later filled using
#   ``_set_configs_from_environment()`` at end of importing ``networkx/__init__.py``.
#
# nx.config:
#   Config object for NetworkX config setting. Created using
#   ``_set_configs_from_environment()`` at end of importing ``networkx/__init__.py``.
#
# private dicts:
#   nx.utils.backends._loaded_backends:
#       dict used to memoize loaded backends. Keyed by backend name to loaded backends.
#
#   nx.utils.backends._registered_algorithms:
#       dict of all the dispatchable functions in networkx, keyed by _dispatchable
#       function name to the wrapped function object.

import inspect
import itertools
import logging
import os
import typing
import warnings
from functools import partial
from importlib.metadata import entry_points

import networkx as nx

from .configs import BackendPriorities, Config, NetworkXConfig
from .decorators import argmap

__all__ = ["_dispatchable"]

_logger = logging.getLogger(__name__)
FAILED_TO_CONVERT = "FAILED_TO_CONVERT"


def _get_backends(group, *, load_and_call=False):
    """
    Retrieve NetworkX ``backends`` and ``backend_info`` from the entry points.

    Parameters
    -----------
    group : str
        The entry_point to be retrieved.
    load_and_call : bool, optional
        If True, load and call the backend. Defaults to False.

    Returns
    --------
    dict
        A dictionary mapping backend names to their respective backend objects.

    Notes
    ------
    If a backend is defined more than once, a warning is issued.
    If a backend name is not a valid Python identifier, the backend is
    ignored and a warning is issued.
    The "nx_loopback" backend is removed if it exists, as it is only available during testing.
    A warning is displayed if an error occurs while loading a backend.
    """
    items = entry_points(group=group)
    rv = {}
    for ep in items:
        if not ep.name.isidentifier():
            warnings.warn(
                f"networkx backend name is not a valid identifier: {ep.name!r}. Ignoring.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif ep.name in rv:
            warnings.warn(
                f"networkx backend defined more than once: {ep.name}",
                RuntimeWarning,
                stacklevel=2,
            )
        elif load_and_call:
            try:
                rv[ep.name] = ep.load()()
            except Exception as exc:
                warnings.warn(
                    f"Error encountered when loading info for backend {ep.name}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            rv[ep.name] = ep
    rv.pop("nx_loopback", None)
    return rv


# Note: "networkx" is in `backend_info` but ignored in `backends` and `config.backends`.
# It is valid to use "networkx" as a backend argument and in `config.backend_priority`.
# If we make "networkx" a "proper" backend, put it in `backends` and `config.backends`.
backends = _get_backends("networkx.backends")

# Use _set_configs_from_environment() below to fill backend_info dict as
# the last step in importing networkx
backend_info = {}

# Load and cache backends on-demand
_loaded_backends = {}  # type: ignore[var-annotated]
_registered_algorithms = {}


# Get default configuration from environment variables at import time
def _comma_sep_to_list(string):
    return [x_strip for x in string.strip().split(",") if (x_strip := x.strip())]


def _set_configs_from_environment():
    """Initialize ``config.backend_priority``, load backend_info and config.

    This gets default values from environment variables (see ``nx.config`` for details).
    This function is run at the very end of importing networkx. It is run at this time
    to avoid loading backend_info before the rest of networkx is imported in case a
    backend uses networkx for its backend_info (e.g. subclassing the Config class.)
    """
    # backend_info is defined above as empty dict. Fill it after import finishes.
    backend_info.update(_get_backends("networkx.backend_info", load_and_call=True))
    backend_info.update(
        (backend, {}) for backend in backends.keys() - backend_info.keys()
    )

    # set up config based on backend_info and environment
    backend_config = {}
    for backend, info in backend_info.items():
        if "default_config" not in info:
            cfg = Config()
        else:
            cfg = info["default_config"]
            if not isinstance(cfg, Config):
                cfg = Config(**cfg)
        backend_config[backend] = cfg
    backend_config = Config(**backend_config)
    # Setting doc of backends_config type is not setting doc of Config
    # Config has __new__ method that returns instance with a unique type!
    type(backend_config).__doc__ = "All installed NetworkX backends and their configs."

    backend_priority = BackendPriorities(algos=[], generators=[], classes=[])

    config = NetworkXConfig(
        backend_priority=backend_priority,
        backends=backend_config,
        cache_converted_graphs=bool(
            os.environ.get("NETWORKX_CACHE_CONVERTED_GRAPHS", True)
        ),
        fallback_to_nx=bool(os.environ.get("NETWORKX_FALLBACK_TO_NX", False)),
        warnings_to_ignore=set(
            _comma_sep_to_list(os.environ.get("NETWORKX_WARNINGS_TO_IGNORE", ""))
        ),
    )

    # Add "networkx" item to backend_info now b/c backend_config is set up
    backend_info["networkx"] = {}

    # NETWORKX_BACKEND_PRIORITY is the same as NETWORKX_BACKEND_PRIORITY_ALGOS
    priorities = {
        key[26:].lower(): val
        for key, val in os.environ.items()
        if key.startswith("NETWORKX_BACKEND_PRIORITY_")
    }
    backend_priority = config.backend_priority
    backend_priority.algos = (
        _comma_sep_to_list(priorities.pop("algos"))
        if "algos" in priorities
        else _comma_sep_to_list(
            os.environ.get(
                "NETWORKX_BACKEND_PRIORITY",
                os.environ.get("NETWORKX_AUTOMATIC_BACKENDS", ""),
            )
        )
    )
    backend_priority.generators = _comma_sep_to_list(priorities.pop("generators", ""))
    for key in sorted(priorities):
        backend_priority[key] = _comma_sep_to_list(priorities[key])

    return config


def _do_nothing():
    """This does nothing at all, yet it helps turn ``_dispatchable`` into functions.

    Use this with the ``argmap`` decorator to turn ``self`` into a function. It results
    in some small additional overhead compared to calling ``_dispatchable`` directly,
    but ``argmap`` has the property that it can stack with other ``argmap``
    decorators "for free". Being a function is better for REPRs and type-checkers.
    """


def _always_run(name, args, kwargs):
    return True


def _load_backend(backend_name):
    if backend_name in _loaded_backends:
        return _loaded_backends[backend_name]
    if backend_name not in backends:
        raise ImportError(f"'{backend_name}' backend is not installed")
    rv = _loaded_backends[backend_name] = backends[backend_name].load()
    if not hasattr(rv, "can_run"):
        rv.can_run = _always_run
    if not hasattr(rv, "should_run"):
        rv.should_run = _always_run
    return rv


class _dispatchable:
    _is_testing = False

    def __new__(
        cls,
        func=None,
        *,
        name=None,
        graphs="G",
        edge_attrs=None,
        node_attrs=None,
        preserve_edge_attrs=False,
        preserve_node_attrs=False,
        preserve_graph_attrs=False,
        preserve_all_attrs=False,
        mutates_input=False,
        returns_graph=False,
        implemented_by_nx=True,
    ):
        """A decorator function that is used to redirect the execution of ``func``
        function to its backend implementation.

        This decorator allows the function to dispatch to different backend
        implementations based on the input graph types, and also manages the
        extra keywords ``backend`` and ``**backend_kwargs``.
        Usage can be any of the following decorator forms:

        - ``@_dispatchable``
        - ``@_dispatchable()``
        - ``@_dispatchable(name="override_name")``
        - ``@_dispatchable(graphs="graph_var_name")``
        - ``@_dispatchable(edge_attrs="weight")``
        - ``@_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})``
            with 0 and 1 giving the position in the signature function for graph
            objects. When ``edge_attrs`` is a dict, keys are keyword names and values
            are defaults.

        Parameters
        ----------
        func : callable, optional (default: None)
            The function to be decorated. If None, ``_dispatchable`` returns a
            partial object that can be used to decorate a function later. If ``func``
            is a callable, returns a new callable object that dispatches to a backend
            function based on input graph types.

        name : str, optional (default: name of `func`)
            The dispatch name for the function. It defaults to the name of `func`,
            but can be set manually to avoid conflicts in the global dispatch
            namespace. A common pattern is to prefix the function name with its
            module or submodule to make it unique. For example:

                - ``@_dispatchable(name="tournament_is_strongly_connected")``
                  resolves conflict between ``nx.tournament.is_strongly_connected``
                  and ``nx.is_strongly_connected``.
                - ``@_dispatchable(name="approximate_node_connectivity")``
                  resolves conflict between ``nx.approximation.node_connectivity``
                  and ``nx.connectivity.node_connectivity``.

        graphs : str or dict or None, optional (default: "G")
            If a string, the parameter name of the graph, which must be the first
            argument of the wrapped function. If more than one graph is required
            for the function (or if the graph is not the first argument), provide
            a dict keyed by graph parameter name to the value parameter position.
            A question mark in the name indicates an optional argument.
            For example, ``@_dispatchable(graphs={"G": 0, "auxiliary?": 4})``
            indicates the 0th parameter ``G`` of the function is a required graph,
            and the 4th parameter ``auxiliary?`` is an optional graph.
            To indicate that an argument is a list of graphs, do ``"[graphs]"``.
            Use ``graphs=None``, if *no* arguments are NetworkX graphs such as for
            graph generators, readers, and conversion functions.

        edge_attrs : str or dict, optional (default: None)
            ``edge_attrs`` holds information about edge attribute arguments
            and default values for those edge attributes.
            If a string, ``edge_attrs`` holds the function argument name that
            indicates a single edge attribute to include in the converted graph.
            The default value for this attribute is 1. To indicate that an argument
            is a list of attributes (all with default value 1), use e.g. ``"[attrs]"``.
            If a dict, ``edge_attrs`` holds a dict keyed by argument names, with
            values that are either the default value or, if a string, the argument
            name that indicates the default value.
            If None, function does not use edge attributes.

        node_attrs : str or dict, optional
            Like ``edge_attrs``, but for node attributes.

        preserve_edge_attrs : bool or str or dict, optional (default: False)
            If bool, whether to preserve all edge attributes.
            If a string, the parameter name that may indicate (with ``True`` or a
            callable argument) whether all edge attributes should be preserved
            when converting graphs to a backend graph type.
            If a dict of form ``{graph_name: {attr: default}}``, indicate
            pre-determined edge attributes (and defaults) to preserve for the
            indicated input graph.

        preserve_node_attrs : bool or str or dict, optional (default: False)
            Like ``preserve_edge_attrs``, but for node attributes.

        preserve_graph_attrs : bool or set, optional (default: False)
            If bool, whether to preserve all graph attributes.
            If set, which input graph arguments to preserve graph attributes.

        preserve_all_attrs : bool, optional (default: False)
            Whether to preserve all edge, node and graph attributes.
            If True, this overrides all the other preserve_*_attrs.

        mutates_input : bool or dict, optional (default: False)
            If bool, whether the function mutates an input graph argument.
            If dict of ``{arg_name: arg_pos}``, name and position of bool arguments
            that indicate whether an input graph will be mutated, and ``arg_name``
            may begin with ``"not "`` to negate the logic (for example, ``"not copy"``
            means we mutate the input graph when the ``copy`` argument is False).
            By default, dispatching doesn't convert input graphs to a different
            backend for functions that mutate input graphs.

        returns_graph : bool, optional (default: False)
            Whether the function can return or yield a graph object. By default,
            dispatching doesn't convert input graphs to a different backend for
            functions that return graphs.

        implemented_by_nx : bool, optional (default: True)
            Whether the function is implemented by NetworkX. If it is not, then the
            function is included in NetworkX only as an API to dispatch to backends.
            Default is True.
        """
        if func is None:
            return partial(
                _dispatchable,
                name=name,
                graphs=graphs,
                edge_attrs=edge_attrs,
                node_attrs=node_attrs,
                preserve_edge_attrs=preserve_edge_attrs,
                preserve_node_attrs=preserve_node_attrs,
                preserve_graph_attrs=preserve_graph_attrs,
                preserve_all_attrs=preserve_all_attrs,
                mutates_input=mutates_input,
                returns_graph=returns_graph,
                implemented_by_nx=implemented_by_nx,
            )
        if isinstance(func, str):
            raise TypeError("'name' and 'graphs' must be passed by keyword") from None
        # If name not provided, use the name of the function
        if name is None:
            name = func.__name__

        self = object.__new__(cls)

        # standard function-wrapping stuff
        # __annotations__ not used
        self.__name__ = func.__name__
        # self.__doc__ = func.__doc__  # __doc__ handled as cached property
        self.__defaults__ = func.__defaults__
        # Add `backend=` keyword argument to allow backend choice at call-time
        if func.__kwdefaults__:
            self.__kwdefaults__ = {**func.__kwdefaults__, "backend": None}
        else:
            self.__kwdefaults__ = {"backend": None}
        self.__module__ = func.__module__
        self.__qualname__ = func.__qualname__
        self.__dict__.update(func.__dict__)
        self.__wrapped__ = func

        # Supplement docstring with backend info; compute and cache when needed
        self._orig_doc = func.__doc__
        self._cached_doc = None

        self.orig_func = func
        self.name = name
        self.edge_attrs = edge_attrs
        self.node_attrs = node_attrs
        self.preserve_edge_attrs = preserve_edge_attrs or preserve_all_attrs
        self.preserve_node_attrs = preserve_node_attrs or preserve_all_attrs
        self.preserve_graph_attrs = preserve_graph_attrs or preserve_all_attrs
        self.mutates_input = mutates_input
        # Keep `returns_graph` private for now, b/c we may extend info on return types
        self._returns_graph = returns_graph

        if edge_attrs is not None and not isinstance(edge_attrs, str | dict):
            raise TypeError(
                f"Bad type for edge_attrs: {type(edge_attrs)}. Expected str or dict."
            ) from None
        if node_attrs is not None and not isinstance(node_attrs, str | dict):
            raise TypeError(
                f"Bad type for node_attrs: {type(node_attrs)}. Expected str or dict."
            ) from None
        if not isinstance(self.preserve_edge_attrs, bool | str | dict):
            raise TypeError(
                f"Bad type for preserve_edge_attrs: {type(self.preserve_edge_attrs)}."
                " Expected bool, str, or dict."
            ) from None
        if not isinstance(self.preserve_node_attrs, bool | str | dict):
            raise TypeError(
                f"Bad type for preserve_node_attrs: {type(self.preserve_node_attrs)}."
                " Expected bool, str, or dict."
            ) from None
        if not isinstance(self.preserve_graph_attrs, bool | set):
            raise TypeError(
                f"Bad type for preserve_graph_attrs: {type(self.preserve_graph_attrs)}."
                " Expected bool or set."
            ) from None
        if not isinstance(self.mutates_input, bool | dict):
            raise TypeError(
                f"Bad type for mutates_input: {type(self.mutates_input)}."
                " Expected bool or dict."
            ) from None
        if not isinstance(self._returns_graph, bool):
            raise TypeError(
                f"Bad type for returns_graph: {type(self._returns_graph)}."
                " Expected bool."
            ) from None

        if isinstance(graphs, str):
            graphs = {graphs: 0}
        elif graphs is None:
            pass
        elif not isinstance(graphs, dict):
            raise TypeError(
                f"Bad type for graphs: {type(graphs)}. Expected str or dict."
            ) from None
        elif len(graphs) == 0:
            raise KeyError("'graphs' must contain at least one variable name") from None

        # This dict comprehension is complicated for better performance; equivalent shown below.
        self.optional_graphs = set()
        self.list_graphs = set()
        if graphs is None:
            self.graphs = {}
        else:
            self.graphs = {
                self.optional_graphs.add(val := k[:-1]) or val
                if (last := k[-1]) == "?"
                else self.list_graphs.add(val := k[1:-1]) or val
                if last == "]"
                else k: v
                for k, v in graphs.items()
            }
        # The above is equivalent to:
        # self.optional_graphs = {k[:-1] for k in graphs if k[-1] == "?"}
        # self.list_graphs = {k[1:-1] for k in graphs if k[-1] == "]"}
        # self.graphs = {k[:-1] if k[-1] == "?" else k: v for k, v in graphs.items()}

        # Compute and cache the signature on-demand
        self._sig = None

        # Which backends implement this function?
        self.backends = {
            backend
            for backend, info in backend_info.items()
            if "functions" in info and name in info["functions"]
        }
        if implemented_by_nx:
            self.backends.add("networkx")

        if name in _registered_algorithms:
            raise KeyError(
                f"Algorithm already exists in dispatch namespace: {name}. "
                "Fix by assigning a unique `name=` in the `@_dispatchable` decorator."
            ) from None
        # Use the `argmap` decorator to turn `self` into a function. This does result
        # in small additional overhead compared to calling `_dispatchable` directly,
        # but `argmap` has the property that it can stack with other `argmap`
        # decorators "for free". Being a function is better for REPRs and type-checkers.
        # It also allows `_dispatchable` to be used on class methods, since functions
        # define `__get__`. Without using `argmap`, we would need to define `__get__`.
        self = argmap(_do_nothing)(self)
        _registered_algorithms[name] = self
        return self

    @property
    def __doc__(self):
        """If the cached documentation exists, it is returned.
        Otherwise, the documentation is generated using _make_doc() method,
        cached, and then returned."""

        rv = self._cached_doc
        if rv is None:
            rv = self._cached_doc = self._make_doc()
        return rv

    @__doc__.setter
    def __doc__(self, val):
        """Sets the original documentation to the given value and resets the
        cached documentation."""

        self._orig_doc = val
        self._cached_doc = None

    @property
    def __signature__(self):
        """Return the signature of the original function, with the addition of
        the `backend` and `backend_kwargs` parameters."""

        if self._sig is None:
            sig = inspect.signature(self.orig_func)
            # `backend` is now a reserved argument used by dispatching.
            # assert "backend" not in sig.parameters
            if not any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                sig = sig.replace(
                    parameters=[
                        *sig.parameters.values(),
                        inspect.Parameter(
                            "backend", inspect.Parameter.KEYWORD_ONLY, default=None
                        ),
                        inspect.Parameter(
                            "backend_kwargs", inspect.Parameter.VAR_KEYWORD
                        ),
                    ]
                )
            else:
                *parameters, var_keyword = sig.parameters.values()
                sig = sig.replace(
                    parameters=[
                        *parameters,
                        inspect.Parameter(
                            "backend", inspect.Parameter.KEYWORD_ONLY, default=None
                        ),
                        var_keyword,
                    ]
                )
            self._sig = sig
        return self._sig

    # Fast, simple path if no backends are installed
    def _call_if_no_backends_installed(self, /, *args, backend=None, **kwargs):
        """Returns the result of the original function (no backends installed)."""
        if backend is not None and backend != "networkx":
            raise ImportError(f"'{backend}' backend is not installed")
        if "networkx" not in self.backends:
            raise NotImplementedError(
                f"'{self.name}' is not implemented by 'networkx' backend. "
                "This function is included in NetworkX as an API to dispatch to "
                "other backends."
            )
        return self.orig_func(*args, **kwargs)

    # Dispatch to backends based on inputs, `backend=` arg, or configuration
    def _call_if_any_backends_installed(self, /, *args, backend=None, **kwargs):
        """Returns the result of the original function, or the backend function if
        the backend is specified and that backend implements `func`."""
        # Use `backend_name` in this function instead of `backend`.
        # This is purely for aesthetics and to make it easier to search for this
        # variable since "backend" is used in many comments and log/error messages.
        backend_name = backend
        if backend_name is not None and backend_name not in backend_info:
            raise ImportError(f"'{backend_name}' backend is not installed")

        graphs_resolved = {}
        for gname, pos in self.graphs.items():
            if pos < len(args):
                if gname in kwargs:
                    raise TypeError(f"{self.name}() got multiple values for {gname!r}")
                graph = args[pos]
            elif gname in kwargs:
                graph = kwargs[gname]
            elif gname not in self.optional_graphs:
                raise TypeError(
                    f"{self.name}() missing required graph argument: {gname}"
                )
            else:
                continue
            if graph is None:
                if gname not in self.optional_graphs:
                    raise TypeError(
                        f"{self.name}() required graph argument {gname!r} is None; must be a graph"
                    )
            else:
                graphs_resolved[gname] = graph

        # Alternative to the above that does not check duplicated args or missing required graphs.
        # graphs_resolved = {
        #     gname: graph
        #     for gname, pos in self.graphs.items()
        #     if (graph := args[pos] if pos < len(args) else kwargs.get(gname)) is not None
        # }

        # Check if any graph comes from a backend
        if self.list_graphs:
            # Make sure we don't lose values by consuming an iterator
            args = list(args)
            for gname in self.list_graphs & graphs_resolved.keys():
                list_of_graphs = list(graphs_resolved[gname])
                graphs_resolved[gname] = list_of_graphs
                if gname in kwargs:
                    kwargs[gname] = list_of_graphs
                else:
                    args[self.graphs[gname]] = list_of_graphs

            graph_backend_names = {
                getattr(g, "__networkx_backend__", None)
                for gname, g in graphs_resolved.items()
                if gname not in self.list_graphs
            }
            for gname in self.list_graphs & graphs_resolved.keys():
                graph_backend_names.update(
                    getattr(g, "__networkx_backend__", None)
                    for g in graphs_resolved[gname]
                )
        else:
            graph_backend_names = {
                getattr(g, "__networkx_backend__", None)
                for g in graphs_resolved.values()
            }

        backend_priority = nx.config.backend_priority.get(
            self.name,
            nx.config.backend_priority.classes
            if self.name.endswith("__new__")
            else nx.config.backend_priority.generators
            if self._returns_graph
            else nx.config.backend_priority.algos,
        )
        fallback_to_nx = nx.config.fallback_to_nx and "networkx" in self.backends
        if self._is_testing and backend_priority and backend_name is None:
            # Special path if we are running networkx tests with a backend.
            # This even runs for (and handles) functions that mutate input graphs.
            return self._convert_and_call_for_tests(
                backend_priority[0],
                args,
                kwargs,
                fallback_to_nx=fallback_to_nx,
            )

        graph_backend_names.discard(None)
        if backend_name is not None:
            # Must run with the given backend.
            # `can_run` only used for better log and error messages.
            # Check `mutates_input` for logging, not behavior.
            backend_kwarg_msg = (
                "No other backends will be attempted, because the backend was "
                f"specified with the `backend='{backend_name}'` keyword argument."
            )
            extra_message = (
                f"'{backend_name}' backend raised NotImplementedError when calling "
                f"'{self.name}'. {backend_kwarg_msg}"
            )
            if not graph_backend_names or graph_backend_names == {backend_name}:
                # All graphs are backend graphs--no need to convert!
                if self._can_backend_run(backend_name, args, kwargs):
                    return self._call_with_backend(
                        backend_name, args, kwargs, extra_message=extra_message
                    )
                if self._does_backend_have(backend_name):
                    extra = " for the given arguments"
                else:
                    extra = ""
                raise NotImplementedError(
                    f"'{self.name}' is not implemented by '{backend_name}' backend"
                    f"{extra}. {backend_kwarg_msg}"
                )
            if self._can_convert(backend_name, graph_backend_names):
                if self._can_backend_run(backend_name, args, kwargs):
                    if self._will_call_mutate_input(args, kwargs):
                        _logger.debug(
                            "'%s' will mutate an input graph. This prevents automatic conversion "
                            "to, and use of, backends listed in `nx.config.backend_priority`. "
                            "Using backend specified by the "
                            "`backend='%s'` keyword argument. This may change behavior by not "
                            "mutating inputs.",
                            self.name,
                            backend_name,
                        )
                        mutations = []
                    else:
                        mutations = None
                    rv = self._convert_and_call(
                        backend_name,
                        graph_backend_names,
                        args,
                        kwargs,
                        extra_message=extra_message,
                        mutations=mutations,
                    )
                    if mutations:
                        for cache, key in mutations:
                            # If the call mutates inputs, then remove all inputs gotten
                            # from cache. We do this after all conversions (and call) so
                            # that a graph can be gotten from a cache multiple times.
                            cache.pop(key, None)
                    return rv
                if self._does_backend_have(backend_name):
                    extra = " for the given arguments"
                else:
                    extra = ""
                raise NotImplementedError(
                    f"'{self.name}' is not implemented by '{backend_name}' backend"
                    f"{extra}. {backend_kwarg_msg}"
                )
            if len(graph_backend_names) == 1:
                maybe_s = ""
                graph_backend_names = f"'{next(iter(graph_backend_names))}'"
            else:
                maybe_s = "s"
            raise TypeError(
                f"'{self.name}' is unable to convert graph from backend{maybe_s} "
                f"{graph_backend_names} to '{backend_name}' backend, which was "
                f"specified with the `backend='{backend_name}'` keyword argument. "
                f"{backend_kwarg_msg}"
            )

        if self._will_call_mutate_input(args, kwargs):
            # The current behavior for functions that mutate input graphs:
            #
            # 1. If backend is specified by `backend=` keyword, use it (done above).
            # 2. If inputs are from one backend, try to use it.
            # 3. If all input graphs are instances of `nx.Graph`, then run with the
            #    default "networkx" implementation.
            #
            # Do not automatically convert if a call will mutate inputs, because doing
            # so would change behavior. Hence, we should fail if there are multiple input
            # backends or if the input backend does not implement the function. However,
            # we offer a way for backends to circumvent this if they do not implement
            # this function: we will fall back to the default "networkx" implementation
            # without using conversions if all input graphs are subclasses of `nx.Graph`.
            mutate_msg = (
                "conversions between backends (if configured) will not be attempted "
                "because the original input graph would not be mutated. Using the "
                "backend keyword e.g. `backend='some_backend'` will force conversions "
                "and not mutate the original input graph."
            )
            fallback_msg = (
                "This call will mutate inputs, so fall back to 'networkx' "
                "backend (without converting) since all input graphs are "
                "instances of nx.Graph and are hopefully compatible."
            )
            if len(graph_backend_names) == 1:
                [backend_name] = graph_backend_names
                msg_template = (
                    f"Backend '{backend_name}' does not implement '{self.name}'%s. "
                    f"This call will mutate an input, so automatic {mutate_msg}"
                )
                # `can_run` is only used for better log and error messages
                try:
                    if self._can_backend_run(backend_name, args, kwargs):
                        return self._call_with_backend(
                            backend_name,
                            args,
                            kwargs,
                            extra_message=msg_template % " with these arguments",
                        )
                except NotImplementedError as exc:
                    if all(isinstance(g, nx.Graph) for g in graphs_resolved.values()):
                        _logger.debug(
                            "Backend '%s' raised when calling '%s': %s. %s",
                            backend_name,
                            self.name,
                            exc,
                            fallback_msg,
                        )
                    else:
                        raise
                else:
                    if fallback_to_nx and all(
                        # Consider dropping the `isinstance` check here to allow
                        # duck-type graphs, but let's wait for a backend to ask us.
                        isinstance(g, nx.Graph)
                        for g in graphs_resolved.values()
                    ):
                        # Log that we are falling back to networkx
                        _logger.debug(
                            "Backend '%s' can't run '%s'. %s",
                            backend_name,
                            self.name,
                            fallback_msg,
                        )
                    else:
                        if self._does_backend_have(backend_name):
                            extra = " with these arguments"
                        else:
                            extra = ""
                        raise NotImplementedError(msg_template % extra)
            elif fallback_to_nx and all(
                # Consider dropping the `isinstance` check here to allow
                # duck-type graphs, but let's wait for a backend to ask us.
                isinstance(g, nx.Graph)
                for g in graphs_resolved.values()
            ):
                # Log that we are falling back to networkx
                _logger.debug(
                    "'%s' was called with inputs from multiple backends: %s. %s",
                    self.name,
                    graph_backend_names,
                    fallback_msg,
                )
            else:
                raise RuntimeError(
                    f"'{self.name}' will mutate an input, but it was called with "
                    f"inputs from multiple backends: {graph_backend_names}. "
                    f"Automatic {mutate_msg}"
                )
            # At this point, no backends are available to handle the call with
            # the input graph types, but if the input graphs are compatible
            # nx.Graph instances, fall back to networkx without converting.
            return self.orig_func(*args, **kwargs)

        # We may generalize fallback configuration as e.g. `nx.config.backend_fallback`
        if fallback_to_nx or not graph_backend_names:
            # Use "networkx" by default if there are no inputs from backends.
            # For example, graph generators should probably return NetworkX graphs
            # instead of raising NotImplementedError.
            backend_fallback = ["networkx"]
        else:
            backend_fallback = []

        # ##########################
        # # How this behaves today #
        # ##########################
        #
        # The prose below describes the implementation and a *possible* way to
        # generalize "networkx" as "just another backend". The code is structured
        # to perhaps someday support backend-to-backend conversions (including
        # simply passing objects from one backend directly to another backend;
        # the dispatch machinery does not necessarily need to perform conversions),
        # but since backend-to-backend matching is not yet supported, the following
        # code is merely a convenient way to implement dispatch behaviors that have
        # been carefully developed since NetworkX 3.0 and to include falling back
        # to the default NetworkX implementation.
        #
        # The current behavior for functions that don't mutate input graphs:
        #
        # 1. If backend is specified by `backend=` keyword, use it (done above).
        # 2. If input is from a backend other than "networkx", try to use it.
        #    - Note: if present, "networkx" graphs will be converted to the backend.
        # 3. If input is from "networkx" (or no backend), try to use backends from
        #    `backend_priority` before running with the default "networkx" implementation.
        # 4. If configured, "fall back" and run with the default "networkx" implementation.
        #
        # ################################################
        # # How this is implemented and may work someday #
        # ################################################
        #
        # Let's determine the order of backends we should try according
        # to `backend_priority`, `backend_fallback`, and input backends.
        # There are two† dimensions of priorities to consider:
        #   backend_priority > unspecified > backend_fallback
        # and
        #   backend of an input > not a backend of an input
        # These are combined to form five groups of priorities as such:
        #
        #                    input   ~input
        #                  +-------+-------+
        # backend_priority |   1   |   2   |
        #      unspecified |   3   |  N/A  | (if only 1)
        # backend_fallback |   4   |   5   |
        #                  +-------+-------+
        #
        # This matches the behaviors we developed in versions 3.0 to 3.2, it
        # ought to cover virtually all use cases we expect, and I (@eriknw) don't
        # think it can be done any simpler (although it can be generalized further
        # and made to be more complicated to capture 100% of *possible* use cases).
        # Some observations:
        #
        #   1. If an input is in `backend_priority`, it will be used before trying a
        #      backend that is higher priority in `backend_priority` and not an input.
        #   2. To prioritize converting from one backend to another even if both implement
        #      a function, list one in `backend_priority` and one in `backend_fallback`.
        #   3. To disable conversions, set `backend_priority` and `backend_fallback` to [].
        #
        # †: There is actually a third dimension of priorities:
        #        should_run == True > should_run == False
        #    Backends with `can_run == True` and `should_run == False` are tried last.
        #
        seen = set()
        group1 = []  # In backend_priority, and an input
        group2 = []  # In backend_priority, but not an input
        for name in backend_priority:
            if name in seen:
                continue
            seen.add(name)
            if name in graph_backend_names:
                group1.append(name)
            else:
                group2.append(name)
        group4 = []  # In backend_fallback, and an input
        group5 = []  # In backend_fallback, but not an input
        for name in backend_fallback:
            if name in seen:
                continue
            seen.add(name)
            if name in graph_backend_names:
                group4.append(name)
            else:
                group5.append(name)
        # An input, but not in backend_priority or backend_fallback.
        group3 = graph_backend_names - seen
        if len(group3) > 1:
            # `group3` backends are not configured for automatic conversion or fallback.
            # There are at least two issues if this group contains multiple backends:
            #
            #   1. How should we prioritize them? We have no good way to break ties.
            #      Although we could arbitrarily choose alphabetical or left-most,
            #      let's follow the Zen of Python and refuse the temptation to guess.
            #   2. We probably shouldn't automatically convert to these backends,
            #      because we are not configured to do so.
            #
            # (2) is important to allow disabling all conversions by setting both
            # `nx.config.backend_priority` and `nx.config.backend_fallback` to [].
            #
            # If there is a single backend in `group3`, then giving it priority over
            # the fallback backends is what is generally expected. For example, this
            # allows input graphs of `backend_fallback` backends (such as "networkx")
            # to be converted to, and run with, the unspecified backend.
            _logger.debug(
                "Call to '%s' has inputs from multiple backends, %s, that "
                "have no priority set in `nx.config.backend_priority`, "
                "so automatic conversions to "
                "these backends will not be attempted.",
                self.name,
                group3,
            )
            group3 = ()

        try_order = list(itertools.chain(group1, group2, group3, group4, group5))
        if len(try_order) > 1:
            # Should we consider adding an option for more verbose logging?
            # For example, we could explain the order of `try_order` in detail.
            _logger.debug(
                "Call to '%s' has inputs from %s backends, and will try to use "
                "backends in the following order: %s",
                self.name,
                graph_backend_names or "no",
                try_order,
            )
        backends_to_try_again = []
        for is_not_first, backend_name in enumerate(try_order):
            if is_not_first:
                _logger.debug("Trying next backend: '%s'", backend_name)
            try:
                if not graph_backend_names or graph_backend_names == {backend_name}:
                    if self._can_backend_run(backend_name, args, kwargs):
                        return self._call_with_backend(backend_name, args, kwargs)
                elif self._can_convert(
                    backend_name, graph_backend_names
                ) and self._can_backend_run(backend_name, args, kwargs):
                    if self._should_backend_run(backend_name, args, kwargs):
                        rv = self._convert_and_call(
                            backend_name, graph_backend_names, args, kwargs
                        )
                        if (
                            self._returns_graph
                            and graph_backend_names
                            and backend_name not in graph_backend_names
                        ):
                            # If the function has graph inputs and graph output, we try
                            # to make it so the backend of the return type will match the
                            # backend of the input types. In case this is not possible,
                            # let's tell the user that the backend of the return graph
                            # has changed. Perhaps we could try to convert back, but
                            # "fallback" backends for graph generators should typically
                            # be compatible with NetworkX graphs.
                            _logger.debug(
                                "Call to '%s' is returning a graph from a different "
                                "backend! It has inputs from %s backends, but ran with "
                                "'%s' backend and is returning graph from '%s' backend",
                                self.name,
                                graph_backend_names,
                                backend_name,
                                backend_name,
                            )
                        return rv
                    # `should_run` is False, but `can_run` is True, so try again later
                    backends_to_try_again.append(backend_name)
            except NotImplementedError as exc:
                _logger.debug(
                    "Backend '%s' raised when calling '%s': %s",
                    backend_name,
                    self.name,
                    exc,
                )

        # We are about to fail. Let's try backends with can_run=True and should_run=False.
        # This is unlikely to help today since we try to run with "networkx" before this.
        for backend_name in backends_to_try_again:
            _logger.debug(
                "Trying backend: '%s' (ignoring `should_run=False`)", backend_name
            )
            try:
                rv = self._convert_and_call(
                    backend_name, graph_backend_names, args, kwargs
                )
                if (
                    self._returns_graph
                    and graph_backend_names
                    and backend_name not in graph_backend_names
                ):
                    _logger.debug(
                        "Call to '%s' is returning a graph from a different "
                        "backend! It has inputs from %s backends, but ran with "
                        "'%s' backend and is returning graph from '%s' backend",
                        self.name,
                        graph_backend_names,
                        backend_name,
                        backend_name,
                    )
                return rv
            except NotImplementedError as exc:
                _logger.debug(
                    "Backend '%s' raised when calling '%s': %s",
                    backend_name,
                    self.name,
                    exc,
                )
        # As a final effort, we could try to convert and run with `group3` backends
        # that we discarded when `len(group3) > 1`, but let's not consider doing
        # so until there is a reasonable request for it.

        if len(unspecified_backends := graph_backend_names - seen) > 1:
            raise TypeError(
                f"Unable to convert inputs from {graph_backend_names} backends and "
                f"run '{self.name}'. NetworkX is configured to automatically convert "
                f"to {try_order} backends. To remedy this, you may enable automatic "
                f"conversion to {unspecified_backends} backends by adding them to "
                "`nx.config.backend_priority`, or you "
                "may specify a backend to use with the `backend=` keyword argument."
            )
        if "networkx" not in self.backends:
            extra = (
                " This function is included in NetworkX as an API to dispatch to "
                "other backends."
            )
        else:
            extra = ""
        raise NotImplementedError(
            f"'{self.name}' is not implemented by {try_order} backends. To remedy "
            "this, you may enable automatic conversion to more backends (including "
            "'networkx') by adding them to `nx.config.backend_priority`, "
            "or you may specify a backend to use with "
            f"the `backend=` keyword argument.{extra}"
        )

    # Dispatch only if there exist any installed backend(s)
    __call__: typing.Callable = (
        _call_if_any_backends_installed if backends else _call_if_no_backends_installed
    )

    def _will_call_mutate_input(self, args, kwargs):
        # Fairly few nx functions mutate the input graph. Most that do, always do.
        # So a boolean input indicates "always" or "never".
        if isinstance((mutates_input := self.mutates_input), bool):
            return mutates_input

        # The ~10 other nx functions either use "copy=True" to control mutation or
        # an arg naming an edge/node attribute to mutate (None means no mutation).
        # Now `mutates_input` is a dict keyed by arg_name to its func-sig position.
        # The `copy=` args are keyed as "not copy" to mean "negate the copy argument".
        # Keys w/o "not " mean the call mutates only when the arg value `is not None`.
        #
        # This section might need different code if new functions mutate in new ways.
        #
        # NetworkX doesn't have any `mutates_input` dicts with more than 1 item.
        # But we treat it like it might have more than 1 item for generality.
        n = len(args)
        return any(
            (args[arg_pos] if n > arg_pos else kwargs.get(arg_name)) is not None
            if not arg_name.startswith("not ")
            # This assumes that e.g. `copy=True` is the default
            else not (args[arg_pos] if n > arg_pos else kwargs.get(arg_name[4:], True))
            for arg_name, arg_pos in mutates_input.items()
        )

    def _can_convert(self, backend_name, graph_backend_names):
        # Backend-to-backend conversion not supported yet.
        # We can only convert to and from networkx.
        rv = backend_name == "networkx" or graph_backend_names.issubset(
            {"networkx", backend_name}
        )
        if not rv:
            _logger.debug(
                "Unable to convert from %s backends to '%s' backend",
                graph_backend_names,
                backend_name,
            )
        return rv

    def _does_backend_have(self, backend_name):
        """Does the specified backend have this algorithm?"""
        if backend_name == "networkx":
            return "networkx" in self.backends
        # Inspect the backend; don't trust metadata used to create `self.backends`
        backend = _load_backend(backend_name)
        return hasattr(backend, self.name)

    def _can_backend_run(self, backend_name, args, kwargs):
        """Can the specified backend run this algorithm with these arguments?"""
        if backend_name == "networkx":
            return "networkx" in self.backends
        backend = _load_backend(backend_name)
        # `backend.can_run` and `backend.should_run` may return strings that describe
        # why they can't or shouldn't be run.
        if not hasattr(backend, self.name):
            _logger.debug(
                "Backend '%s' does not implement '%s'", backend_name, self.name
            )
            return False
        can_run = backend.can_run(self.name, args, kwargs)
        if isinstance(can_run, str) or not can_run:
            reason = f", because: {can_run}" if isinstance(can_run, str) else ""
            _logger.debug(
                "Backend '%s' can't run `%s` with arguments: %s%s",
                backend_name,
                self.name,
                _LazyArgsRepr(self, args, kwargs),
                reason,
            )
            return False
        return True

    def _should_backend_run(self, backend_name, args, kwargs):
        """Should the specified backend run this algorithm with these arguments?

        Note that this does not check ``backend.can_run``.
        """
        # `backend.can_run` and `backend.should_run` may return strings that describe
        # why they can't or shouldn't be run.
        # `_should_backend_run` may assume that `_can_backend_run` returned True.
        if backend_name == "networkx":
            return True
        backend = _load_backend(backend_name)
        should_run = backend.should_run(self.name, args, kwargs)
        if isinstance(should_run, str) or not should_run:
            reason = f", because: {should_run}" if isinstance(should_run, str) else ""
            _logger.debug(
                "Backend '%s' shouldn't run `%s` with arguments: %s%s",
                backend_name,
                self.name,
                _LazyArgsRepr(self, args, kwargs),
                reason,
            )
            return False
        return True

    def _convert_arguments(self, backend_name, args, kwargs, *, use_cache, mutations):
        """Convert graph arguments to the specified backend.

        Returns
        -------
        args tuple and kwargs dict
        """
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        if not self.graphs:
            bound_kwargs = bound.kwargs
            del bound_kwargs["backend"]
            return bound.args, bound_kwargs
        if backend_name == "networkx":
            # `backend_interface.convert_from_nx` preserves everything
            preserve_edge_attrs = preserve_node_attrs = preserve_graph_attrs = True
        else:
            preserve_edge_attrs = self.preserve_edge_attrs
            preserve_node_attrs = self.preserve_node_attrs
            preserve_graph_attrs = self.preserve_graph_attrs
            edge_attrs = self.edge_attrs
            node_attrs = self.node_attrs
        # Convert graphs into backend graph-like object
        # Include the edge and/or node labels if provided to the algorithm
        if preserve_edge_attrs is False:
            # e.g. `preserve_edge_attrs=False`
            pass
        elif preserve_edge_attrs is True:
            # e.g. `preserve_edge_attrs=True`
            edge_attrs = None
        elif isinstance(preserve_edge_attrs, str):
            if bound.arguments[preserve_edge_attrs] is True or callable(
                bound.arguments[preserve_edge_attrs]
            ):
                # e.g. `preserve_edge_attrs="attr"` and `func(attr=True)`
                # e.g. `preserve_edge_attrs="attr"` and `func(attr=myfunc)`
                preserve_edge_attrs = True
                edge_attrs = None
            elif bound.arguments[preserve_edge_attrs] is False and (
                isinstance(edge_attrs, str)
                and edge_attrs == preserve_edge_attrs
                or isinstance(edge_attrs, dict)
                and preserve_edge_attrs in edge_attrs
            ):
                # e.g. `preserve_edge_attrs="attr"` and `func(attr=False)`
                # Treat `False` argument as meaning "preserve_edge_data=False"
                # and not `False` as the edge attribute to use.
                preserve_edge_attrs = False
                edge_attrs = None
            else:
                # e.g. `preserve_edge_attrs="attr"` and `func(attr="weight")`
                preserve_edge_attrs = False
        # Else: e.g. `preserve_edge_attrs={"G": {"weight": 1}}`

        if edge_attrs is None:
            # May have been set to None above b/c all attributes are preserved
            pass
        elif isinstance(edge_attrs, str):
            if edge_attrs[0] == "[":
                # e.g. `edge_attrs="[edge_attributes]"` (argument of list of attributes)
                # e.g. `func(edge_attributes=["foo", "bar"])`
                edge_attrs = {
                    edge_attr: 1 for edge_attr in bound.arguments[edge_attrs[1:-1]]
                }
            elif callable(bound.arguments[edge_attrs]):
                # e.g. `edge_attrs="weight"` and `func(weight=myfunc)`
                preserve_edge_attrs = True
                edge_attrs = None
            elif bound.arguments[edge_attrs] is not None:
                # e.g. `edge_attrs="weight"` and `func(weight="foo")` (default of 1)
                edge_attrs = {bound.arguments[edge_attrs]: 1}
            elif self.name == "to_numpy_array" and hasattr(
                bound.arguments["dtype"], "names"
            ):
                # Custom handling: attributes may be obtained from `dtype`
                edge_attrs = {
                    edge_attr: 1 for edge_attr in bound.arguments["dtype"].names
                }
            else:
                # e.g. `edge_attrs="weight"` and `func(weight=None)`
                edge_attrs = None
        else:
            # e.g. `edge_attrs={"attr": "default"}` and `func(attr="foo", default=7)`
            # e.g. `edge_attrs={"attr": 0}` and `func(attr="foo")`
            edge_attrs = {
                edge_attr: bound.arguments.get(val, 1) if isinstance(val, str) else val
                for key, val in edge_attrs.items()
                if (edge_attr := bound.arguments[key]) is not None
            }

        if preserve_node_attrs is False:
            # e.g. `preserve_node_attrs=False`
            pass
        elif preserve_node_attrs is True:
            # e.g. `preserve_node_attrs=True`
            node_attrs = None
        elif isinstance(preserve_node_attrs, str):
            if bound.arguments[preserve_node_attrs] is True or callable(
                bound.arguments[preserve_node_attrs]
            ):
                # e.g. `preserve_node_attrs="attr"` and `func(attr=True)`
                # e.g. `preserve_node_attrs="attr"` and `func(attr=myfunc)`
                preserve_node_attrs = True
                node_attrs = None
            elif bound.arguments[preserve_node_attrs] is False and (
                isinstance(node_attrs, str)
                and node_attrs == preserve_node_attrs
                or isinstance(node_attrs, dict)
                and preserve_node_attrs in node_attrs
            ):
                # e.g. `preserve_node_attrs="attr"` and `func(attr=False)`
                # Treat `False` argument as meaning "preserve_node_data=False"
                # and not `False` as the node attribute to use. Is this used?
                preserve_node_attrs = False
                node_attrs = None
            else:
                # e.g. `preserve_node_attrs="attr"` and `func(attr="weight")`
                preserve_node_attrs = False
        # Else: e.g. `preserve_node_attrs={"G": {"pos": None}}`

        if node_attrs is None:
            # May have been set to None above b/c all attributes are preserved
            pass
        elif isinstance(node_attrs, str):
            if node_attrs[0] == "[":
                # e.g. `node_attrs="[node_attributes]"` (argument of list of attributes)
                # e.g. `func(node_attributes=["foo", "bar"])`
                node_attrs = {
                    node_attr: None for node_attr in bound.arguments[node_attrs[1:-1]]
                }
            elif callable(bound.arguments[node_attrs]):
                # e.g. `node_attrs="weight"` and `func(weight=myfunc)`
                preserve_node_attrs = True
                node_attrs = None
            elif bound.arguments[node_attrs] is not None:
                # e.g. `node_attrs="weight"` and `func(weight="foo")`
                node_attrs = {bound.arguments[node_attrs]: None}
            else:
                # e.g. `node_attrs="weight"` and `func(weight=None)`
                node_attrs = None
        else:
            # e.g. `node_attrs={"attr": "default"}` and `func(attr="foo", default=7)`
            # e.g. `node_attrs={"attr": 0}` and `func(attr="foo")`
            node_attrs = {
                node_attr: bound.arguments.get(val) if isinstance(val, str) else val
                for key, val in node_attrs.items()
                if (node_attr := bound.arguments[key]) is not None
            }

        # It should be safe to assume that we either have networkx graphs or backend graphs.
        # Future work: allow conversions between backends.
        for gname in self.graphs:
            if gname in self.list_graphs:
                bound.arguments[gname] = [
                    self._convert_graph(
                        backend_name,
                        g,
                        edge_attrs=edge_attrs,
                        node_attrs=node_attrs,
                        preserve_edge_attrs=preserve_edge_attrs,
                        preserve_node_attrs=preserve_node_attrs,
                        preserve_graph_attrs=preserve_graph_attrs,
                        graph_name=gname,
                        use_cache=use_cache,
                        mutations=mutations,
                    )
                    if getattr(g, "__networkx_backend__", "networkx") != backend_name
                    else g
                    for g in bound.arguments[gname]
                ]
            else:
                graph = bound.arguments[gname]
                if graph is None:
                    if gname in self.optional_graphs:
                        continue
                    raise TypeError(
                        f"Missing required graph argument `{gname}` in {self.name} function"
                    )
                if isinstance(preserve_edge_attrs, dict):
                    preserve_edges = False
                    edges = preserve_edge_attrs.get(gname, edge_attrs)
                else:
                    preserve_edges = preserve_edge_attrs
                    edges = edge_attrs
                if isinstance(preserve_node_attrs, dict):
                    preserve_nodes = False
                    nodes = preserve_node_attrs.get(gname, node_attrs)
                else:
                    preserve_nodes = preserve_node_attrs
                    nodes = node_attrs
                if isinstance(preserve_graph_attrs, set):
                    preserve_graph = gname in preserve_graph_attrs
                else:
                    preserve_graph = preserve_graph_attrs
                if getattr(graph, "__networkx_backend__", "networkx") != backend_name:
                    bound.arguments[gname] = self._convert_graph(
                        backend_name,
                        graph,
                        edge_attrs=edges,
                        node_attrs=nodes,
                        preserve_edge_attrs=preserve_edges,
                        preserve_node_attrs=preserve_nodes,
                        preserve_graph_attrs=preserve_graph,
                        graph_name=gname,
                        use_cache=use_cache,
                        mutations=mutations,
                    )
        bound_kwargs = bound.kwargs
        del bound_kwargs["backend"]
        return bound.args, bound_kwargs

    def _convert_graph(
        self,
        backend_name,
        graph,
        *,
        edge_attrs,
        node_attrs,
        preserve_edge_attrs,
        preserve_node_attrs,
        preserve_graph_attrs,
        graph_name,
        use_cache,
        mutations,
    ):
        nx_cache = getattr(graph, "__networkx_cache__", None) if use_cache else None
        if nx_cache is not None:
            cache = nx_cache.setdefault("backends", {}).setdefault(backend_name, {})
            key = _get_cache_key(
                edge_attrs=edge_attrs,
                node_attrs=node_attrs,
                preserve_edge_attrs=preserve_edge_attrs,
                preserve_node_attrs=preserve_node_attrs,
                preserve_graph_attrs=preserve_graph_attrs,
            )
            compat_key, rv = _get_from_cache(cache, key, mutations=mutations)
            if rv is not None:
                if "cache" not in nx.config.warnings_to_ignore:
                    warnings.warn(
                        "Note: conversions to backend graphs are saved to cache "
                        "(`G.__networkx_cache__` on the original graph) by default."
                        "\n\nThis warning means the cached graph is being used "
                        f"for the {backend_name!r} backend in the "
                        f"call to {self.name}.\n\nFor the cache to be consistent "
                        "(i.e., correct), the input graph must not have been "
                        "manually mutated since the cached graph was created. "
                        "Examples of manually mutating the graph data structures "
                        "resulting in an inconsistent cache include:\n\n"
                        "    >>> G[u][v][key] = val\n\n"
                        "and\n\n"
                        "    >>> for u, v, d in G.edges(data=True):\n"
                        "    ...     d[key] = val\n\n"
                        "Using methods such as `G.add_edge(u, v, weight=val)` "
                        "will correctly clear the cache to keep it consistent. "
                        "You may also use `G.__networkx_cache__.clear()` to "
                        "manually clear the cache, or set `G.__networkx_cache__` "
                        "to None to disable caching for G. Enable or disable caching "
                        "globally via `nx.config.cache_converted_graphs` config.\n\n"
                        "To disable this warning:\n\n"
                        '    >>> nx.config.warnings_to_ignore.add("cache")\n'
                    )
                if rv == FAILED_TO_CONVERT:
                    # NotImplementedError is reasonable to use since the backend doesn't
                    # implement this conversion. However, this will be different than
                    # the original exception that the backend raised when it failed.
                    # Using NotImplementedError allows the next backend to be attempted.
                    raise NotImplementedError(
                        "Graph conversion aborted: unable to convert graph to "
                        f"'{backend_name}' backend in call to `{self.name}', "
                        "because this conversion has previously failed."
                    )
                _logger.debug(
                    "Using cached converted graph (from '%s' to '%s' backend) "
                    "in call to '%s' for '%s' argument",
                    getattr(graph, "__networkx_backend__", None),
                    backend_name,
                    self.name,
                    graph_name,
                )
                return rv

        if backend_name == "networkx":
            # Perhaps we should check that "__networkx_backend__" attribute exists
            # and return the original object if not.
            if not hasattr(graph, "__networkx_backend__"):
                _logger.debug(
                    "Unable to convert input to 'networkx' backend in call to '%s' for "
                    "'%s argument, because it is not from a backend (i.e., it does not "
                    "have `G.__networkx_backend__` attribute). Using the original "
                    "object: %s",
                    self.name,
                    graph_name,
                    graph,
                )
                # This may fail, but let it fail in the networkx function
                return graph
            backend = _load_backend(graph.__networkx_backend__)
            try:
                rv = backend.convert_to_nx(graph)
            except Exception:
                if nx_cache is not None:
                    _set_to_cache(cache, key, FAILED_TO_CONVERT)
                raise
        else:
            backend = _load_backend(backend_name)
            try:
                rv = backend.convert_from_nx(
                    graph,
                    edge_attrs=edge_attrs,
                    node_attrs=node_attrs,
                    preserve_edge_attrs=preserve_edge_attrs,
                    preserve_node_attrs=preserve_node_attrs,
                    # Always preserve graph attrs when we are caching b/c this should be
                    # cheap and may help prevent extra (unnecessary) conversions. Because
                    # we do this, we don't need `preserve_graph_attrs` in the cache key.
                    preserve_graph_attrs=preserve_graph_attrs or nx_cache is not None,
                    name=self.name,
                    graph_name=graph_name,
                )
            except Exception:
                if nx_cache is not None:
                    _set_to_cache(cache, key, FAILED_TO_CONVERT)
                raise
        if nx_cache is not None:
            _set_to_cache(cache, key, rv)
            _logger.debug(
                "Caching converted graph (from '%s' to '%s' backend) "
                "in call to '%s' for '%s' argument",
                getattr(graph, "__networkx_backend__", None),
                backend_name,
                self.name,
                graph_name,
            )

        return rv

    def _call_with_backend(self, backend_name, args, kwargs, *, extra_message=None):
        """Call this dispatchable function with a backend without converting inputs."""
        if backend_name == "networkx":
            return self.orig_func(*args, **kwargs)
        backend = _load_backend(backend_name)
        _logger.debug(
            "Using backend '%s' for call to '%s' with arguments: %s",
            backend_name,
            self.name,
            _LazyArgsRepr(self, args, kwargs),
        )
        try:
            return getattr(backend, self.name)(*args, **kwargs)
        except NotImplementedError as exc:
            if extra_message is not None:
                _logger.debug(
                    "Backend '%s' raised when calling '%s': %s",
                    backend_name,
                    self.name,
                    exc,
                )
                raise NotImplementedError(extra_message) from exc
            raise

    def _convert_and_call(
        self,
        backend_name,
        input_backend_names,
        args,
        kwargs,
        *,
        extra_message=None,
        mutations=None,
    ):
        """Call this dispatchable function with a backend after converting inputs.

        Parameters
        ----------
        backend_name : str
        input_backend_names : set[str]
        args : arguments tuple
        kwargs : keywords dict
        extra_message : str, optional
            Additional message to log if NotImplementedError is raised by backend.
        mutations : list, optional
            Used to clear objects gotten from cache if inputs will be mutated.
        """
        if backend_name == "networkx":
            func = self.orig_func
        else:
            backend = _load_backend(backend_name)
            func = getattr(backend, self.name)
        other_backend_names = input_backend_names - {backend_name}
        _logger.debug(
            "Converting input graphs from %s backend%s to '%s' backend for call to '%s'",
            other_backend_names
            if len(other_backend_names) > 1
            else f"'{next(iter(other_backend_names))}'",
            "s" if len(other_backend_names) > 1 else "",
            backend_name,
            self.name,
        )
        try:
            converted_args, converted_kwargs = self._convert_arguments(
                backend_name,
                args,
                kwargs,
                use_cache=nx.config.cache_converted_graphs,
                mutations=mutations,
            )
        except NotImplementedError as exc:
            # Only log the exception if we are adding an extra message
            # because we don't want to lose any information.
            _logger.debug(
                "Failed to convert graphs from %s to '%s' backend for call to '%s'"
                + ("" if extra_message is None else ": %s"),
                input_backend_names,
                backend_name,
                self.name,
                *(() if extra_message is None else (exc,)),
            )
            if extra_message is not None:
                raise NotImplementedError(extra_message) from exc
            raise
        if backend_name != "networkx":
            _logger.debug(
                "Using backend '%s' for call to '%s' with arguments: %s",
                backend_name,
                self.name,
                _LazyArgsRepr(self, converted_args, converted_kwargs),
            )
        try:
            return func(*converted_args, **converted_kwargs)
        except NotImplementedError as exc:
            if extra_message is not None:
                _logger.debug(
                    "Backend '%s' raised when calling '%s': %s",
                    backend_name,
                    self.name,
                    exc,
                )
                raise NotImplementedError(extra_message) from exc
            raise

    def _convert_and_call_for_tests(
        self, backend_name, args, kwargs, *, fallback_to_nx=False
    ):
        """Call this dispatchable function with a backend; for use with testing."""
        backend = _load_backend(backend_name)
        if not self._can_backend_run(backend_name, args, kwargs):
            if fallback_to_nx or not self.graphs:
                if fallback_to_nx:
                    _logger.debug(
                        "Falling back to use 'networkx' instead of '%s' backend "
                        "for call to '%s' with arguments: %s",
                        backend_name,
                        self.name,
                        _LazyArgsRepr(self, args, kwargs),
                    )
                return self.orig_func(*args, **kwargs)

            import pytest

            msg = f"'{self.name}' not implemented by {backend_name}"
            if hasattr(backend, self.name):
                msg += " with the given arguments"
            pytest.xfail(msg)

        from collections.abc import Iterable, Iterator, Mapping
        from copy import copy, deepcopy
        from io import BufferedReader, BytesIO, StringIO, TextIOWrapper
        from itertools import tee
        from random import Random

        import numpy as np
        from numpy.random import Generator, RandomState
        from scipy.sparse import sparray

        # We sometimes compare the backend result (or input graphs) to the
        # original result (or input graphs), so we need two sets of arguments.
        compare_result_to_nx = (
            self._returns_graph
            and "networkx" in self.backends
            and self.name
            not in {
                # Has graphs as node values (unable to compare)
                "quotient_graph",
                # We don't handle tempfile.NamedTemporaryFile arguments
                "read_gml",
                "read_graph6",
                "read_sparse6",
                # We don't handle io.BufferedReader or io.TextIOWrapper arguments
                "bipartite_read_edgelist",
                "read_adjlist",
                "read_edgelist",
                "read_graphml",
                "read_multiline_adjlist",
                "read_pajek",
                "from_pydot",
                "pydot_read_dot",
                "agraph_read_dot",
                # graph comparison fails b/c of nan values
                "read_gexf",
            }
        )
        compare_inputs_to_nx = (
            "networkx" in self.backends and self._will_call_mutate_input(args, kwargs)
        )

        # Tee iterators and copy random state so that they may be used twice.
        if not args or not compare_result_to_nx and not compare_inputs_to_nx:
            args_to_convert = args_nx = args
        else:
            args_to_convert, args_nx = zip(
                *(
                    (arg, deepcopy(arg))
                    if isinstance(arg, RandomState)
                    else (arg, copy(arg))
                    if isinstance(arg, BytesIO | StringIO | Random | Generator)
                    else tee(arg)
                    if isinstance(arg, Iterator)
                    and not isinstance(arg, BufferedReader | TextIOWrapper)
                    else (arg, arg)
                    for arg in args
                )
            )
        if not kwargs or not compare_result_to_nx and not compare_inputs_to_nx:
            kwargs_to_convert = kwargs_nx = kwargs
        else:
            kwargs_to_convert, kwargs_nx = zip(
                *(
                    ((k, v), (k, deepcopy(v)))
                    if isinstance(v, RandomState)
                    else ((k, v), (k, copy(v)))
                    if isinstance(v, BytesIO | StringIO | Random | Generator)
                    else ((k, (teed := tee(v))[0]), (k, teed[1]))
                    if isinstance(v, Iterator)
                    and not isinstance(v, BufferedReader | TextIOWrapper)
                    else ((k, v), (k, v))
                    for k, v in kwargs.items()
                )
            )
            kwargs_to_convert = dict(kwargs_to_convert)
            kwargs_nx = dict(kwargs_nx)

        try:
            converted_args, converted_kwargs = self._convert_arguments(
                backend_name,
                args_to_convert,
                kwargs_to_convert,
                use_cache=False,
                mutations=None,
            )
        except NotImplementedError as exc:
            if fallback_to_nx:
                _logger.debug(
                    "Graph conversion failed; falling back to use 'networkx' instead "
                    "of '%s' backend for call to '%s'",
                    backend_name,
                    self.name,
                )
                return self.orig_func(*args_nx, **kwargs_nx)
            import pytest

            pytest.xfail(
                exc.args[0] if exc.args else f"{self.name} raised {type(exc).__name__}"
            )

        if compare_inputs_to_nx:
            # Ensure input graphs are different if the function mutates an input graph.
            bound_backend = self.__signature__.bind(*converted_args, **converted_kwargs)
            bound_backend.apply_defaults()
            bound_nx = self.__signature__.bind(*args_nx, **kwargs_nx)
            bound_nx.apply_defaults()
            for gname in self.graphs:
                graph_nx = bound_nx.arguments[gname]
                if bound_backend.arguments[gname] is graph_nx is not None:
                    bound_nx.arguments[gname] = graph_nx.copy()
            args_nx = bound_nx.args
            kwargs_nx = bound_nx.kwargs
            kwargs_nx.pop("backend", None)

        _logger.debug(
            "Using backend '%s' for call to '%s' with arguments: %s",
            backend_name,
            self.name,
            _LazyArgsRepr(self, converted_args, converted_kwargs),
        )
        try:
            result = getattr(backend, self.name)(*converted_args, **converted_kwargs)
        except NotImplementedError as exc:
            if fallback_to_nx:
                _logger.debug(
                    "Backend '%s' raised when calling '%s': %s; "
                    "falling back to use 'networkx' instead.",
                    backend_name,
                    self.name,
                    exc,
                )
                return self.orig_func(*args_nx, **kwargs_nx)
            import pytest

            pytest.xfail(
                exc.args[0] if exc.args else f"{self.name} raised {type(exc).__name__}"
            )

        # Verify that `self._returns_graph` is correct. This compares the return type
        # to the type expected from `self._returns_graph`. This handles tuple and list
        # return types, but *does not* catch functions that yield graphs.
        if (
            self._returns_graph
            != (
                isinstance(result, nx.Graph)
                or hasattr(result, "__networkx_backend__")
                or isinstance(result, tuple | list)
                and any(
                    isinstance(x, nx.Graph) or hasattr(x, "__networkx_backend__")
                    for x in result
                )
            )
            and not (
                # May return Graph or None
                self.name in {"check_planarity", "check_planarity_recursive"}
                and any(x is None for x in result)
            )
            and not (
                # May return Graph or dict
                self.name in {"held_karp_ascent"}
                and any(isinstance(x, dict) for x in result)
            )
            and self.name
            not in {
                # yields graphs
                "all_triads",
                "general_k_edge_subgraphs",
                # yields graphs or arrays
                "nonisomorphic_trees",
            }
        ):
            raise RuntimeError(f"`returns_graph` is incorrect for {self.name}")

        def check_result(val, depth=0):
            if isinstance(val, np.number):
                raise RuntimeError(
                    f"{self.name} returned a numpy scalar {val} ({type(val)}, depth={depth})"
                )
            if isinstance(val, np.ndarray | sparray):
                return
            if isinstance(val, nx.Graph):
                check_result(val._node, depth=depth + 1)
                check_result(val._adj, depth=depth + 1)
                return
            if isinstance(val, Iterator):
                raise NotImplementedError
            if isinstance(val, Iterable) and not isinstance(val, str):
                for x in val:
                    check_result(x, depth=depth + 1)
            if isinstance(val, Mapping):
                for x in val.values():
                    check_result(x, depth=depth + 1)

        def check_iterator(it):
            for val in it:
                try:
                    check_result(val)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"{self.name} returned a numpy scalar {val} ({type(val)})"
                    ) from exc
                yield val

        if self.name in {"from_edgelist"}:
            # numpy scalars are explicitly given as values in some tests
            pass
        elif isinstance(result, Iterator):
            result = check_iterator(result)
        else:
            try:
                check_result(result)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{self.name} returned a numpy scalar {result} ({type(result)})"
                ) from exc
            check_result(result)

        if self.name.endswith("__new__"):
            # Graph is not yet done initializing; no sense doing more here
            return result

        def assert_graphs_equal(G1, G2, strict=True):
            assert G1.number_of_nodes() == G2.number_of_nodes()
            assert G1.number_of_edges() == G2.number_of_edges()
            assert G1.is_directed() is G2.is_directed()
            assert G1.is_multigraph() is G2.is_multigraph()
            if strict:
                assert G1.graph == G2.graph
                assert G1._node == G2._node
                assert G1._adj == G2._adj
            else:
                assert set(G1) == set(G2)
                assert set(G1.edges) == set(G2.edges)

        if compare_inputs_to_nx:
            # Special-case algorithms that mutate input graphs
            result_nx = self.orig_func(*args_nx, **kwargs_nx)
            for gname in self.graphs:
                G0 = bound_backend.arguments[gname]
                G1 = bound_nx.arguments[gname]
                if G0 is not None or G1 is not None:
                    G1 = backend.convert_to_nx(G1)
                    assert_graphs_equal(G0, G1, strict=False)

        converted_result = backend.convert_to_nx(result)
        if compare_result_to_nx and isinstance(converted_result, nx.Graph):
            # For graph return types (e.g. generators), we compare that results are
            # the same between the backend and networkx, then return the original
            # networkx result so the iteration order will be consistent in tests.
            if compare_inputs_to_nx:
                G = result_nx
            else:
                G = self.orig_func(*args_nx, **kwargs_nx)
            assert_graphs_equal(G, converted_result)
            return G

        return converted_result

    def _make_doc(self):
        """Generate the backends section at the end for functions having an alternate
        backend implementation(s) using the `backend_info` entry-point."""

        if self.backends == {"networkx"}:
            return self._orig_doc
        # Add "Backends" section to the bottom of the docstring (if there are backends)
        lines = [
            "Backends",
            "--------",
        ]
        for backend in sorted(self.backends - {"networkx"}):
            info = backend_info[backend]
            if "short_summary" in info:
                lines.append(f"{backend} : {info['short_summary']}")
            else:
                lines.append(backend)
            if "functions" not in info or self.name not in info["functions"]:
                lines.append("")
                continue

            func_info = info["functions"][self.name]

            # Renaming extra_docstring to additional_docs
            if func_docs := (
                func_info.get("additional_docs") or func_info.get("extra_docstring")
            ):
                lines.extend(
                    f"  {line}" if line else line for line in func_docs.split("\n")
                )
                add_gap = True
            else:
                add_gap = False

            # Renaming extra_parameters to additional_parameters
            if extra_parameters := (
                func_info.get("extra_parameters")
                or func_info.get("additional_parameters")
            ):
                if add_gap:
                    lines.append("")
                lines.append("  Additional parameters:")
                for param in sorted(extra_parameters):
                    lines.append(f"    {param}")
                    if desc := extra_parameters[param]:
                        lines.append(f"      {desc}")
                    lines.append("")
            else:
                lines.append("")

            if func_url := func_info.get("url"):
                lines.append(f"[`Source <{func_url}>`_]")
                lines.append("")

        # We assume the docstrings are indented by four spaces (true for now)
        new_doc = self._orig_doc or ""
        if not new_doc.rstrip():
            new_doc = f"The original docstring for {self.name} was empty."
        if self.backends:
            lines.pop()  # Remove last empty line
            to_add = "\n    ".join(lines)
            new_doc = f"{new_doc.rstrip()}\n\n    {to_add}"

        # For backend-only funcs, add "Attention" admonishment after the one line summary
        if "networkx" not in self.backends:
            lines = new_doc.split("\n")
            index = 0
            while not lines[index].strip():
                index += 1
            while index < len(lines) and lines[index].strip():
                index += 1
            backends = sorted(self.backends)
            if len(backends) == 0:
                example = ""
            elif len(backends) == 1:
                example = f' such as "{backends[0]}"'
            elif len(backends) == 2:
                example = f' such as "{backends[0]} or "{backends[1]}"'
            else:
                example = (
                    " such as "
                    + ", ".join(f'"{x}"' for x in backends[:-1])
                    + f', or "{backends[-1]}"'  # Oxford comma
                )
            to_add = (
                "\n    .. attention:: This function does not have a default NetworkX implementation.\n"
                "        It may only be run with an installable :doc:`backend </backends>` that\n"
                f"        supports it{example}.\n\n"
                "        Hint: use ``backend=...`` keyword argument to specify a backend or add\n"
                "        backends to ``nx.config.backend_priority``."
            )
            lines.insert(index, to_add)
            new_doc = "\n".join(lines)
        return new_doc

    def __reduce__(self):
        """Allow this object to be serialized with pickle.

        This uses the global registry `_registered_algorithms` to deserialize.
        """
        return _restore_dispatchable, (self.name,)


def _restore_dispatchable(name):
    return _registered_algorithms[name].__wrapped__


def _get_cache_key(
    *,
    edge_attrs,
    node_attrs,
    preserve_edge_attrs,
    preserve_node_attrs,
    preserve_graph_attrs,
):
    """Return key used by networkx caching given arguments for ``convert_from_nx``."""
    # edge_attrs: dict | None
    # node_attrs: dict | None
    # preserve_edge_attrs: bool (False if edge_attrs is not None)
    # preserve_node_attrs: bool (False if node_attrs is not None)
    return (
        frozenset(edge_attrs.items())
        if edge_attrs is not None
        else preserve_edge_attrs,
        frozenset(node_attrs.items())
        if node_attrs is not None
        else preserve_node_attrs,
    )


def _get_from_cache(cache, key, *, backend_name=None, mutations=None):
    """Search the networkx cache for a graph that is compatible with ``key``.

    Parameters
    ----------
    cache : dict
        If ``backend_name`` is given, then this is treated as ``G.__networkx_cache__``,
        but if ``backend_name`` is None, then this is treated as the resolved inner
        cache such as ``G.__networkx_cache__["backends"][backend_name]``.
    key : tuple
        Cache key from ``_get_cache_key``.
    backend_name : str, optional
        Name of the backend to control how ``cache`` is interpreted.
    mutations : list, optional
        Used internally to clear objects gotten from cache if inputs will be mutated.

    Returns
    -------
    tuple or None
        The key of the compatible graph found in the cache.
    graph or "FAILED_TO_CONVERT" or None
        A compatible graph if possible. "FAILED_TO_CONVERT" indicates that a previous
        conversion attempt failed for this cache key.
    """
    if backend_name is not None:
        cache = cache.get("backends", {}).get(backend_name, {})
    if not cache:
        return None, None

    # Do a simple search for a cached graph with compatible data.
    # For example, if we need a single attribute, then it's okay
    # to use a cached graph that preserved all attributes.
    # This looks for an exact match first.
    edge_key, node_key = key
    for compat_key in itertools.product(
        (edge_key, True) if edge_key is not True else (True,),
        (node_key, True) if node_key is not True else (True,),
    ):
        if (rv := cache.get(compat_key)) is not None and (
            rv != FAILED_TO_CONVERT or key == compat_key
        ):
            if mutations is not None:
                # Remove this item from the cache (after all conversions) if
                # the call to this dispatchable function will mutate an input.
                mutations.append((cache, compat_key))
            return compat_key, rv

    # Iterate over the items in `cache` to see if any are compatible.
    # For example, if no edge attributes are needed, then a graph
    # with any edge attribute will suffice. We use the same logic
    # below (but switched) to clear unnecessary items from the cache.
    # Use `list(cache.items())` to be thread-safe.
    for (ekey, nkey), graph in list(cache.items()):
        if graph == FAILED_TO_CONVERT:
            # Return FAILED_TO_CONVERT if any cache key that requires a subset
            # of the edge/node attributes of the given cache key has previously
            # failed to convert. This logic is similar to `_set_to_cache`.
            if ekey is False or edge_key is True:
                pass
            elif ekey is True or edge_key is False or not ekey.issubset(edge_key):
                continue
            if nkey is False or node_key is True:  # or nkey == node_key:
                pass
            elif nkey is True or node_key is False or not nkey.issubset(node_key):
                continue
            # Save to cache for faster subsequent lookups
            cache[key] = FAILED_TO_CONVERT
        elif edge_key is False or ekey is True:
            pass  # Cache works for edge data!
        elif edge_key is True or ekey is False or not edge_key.issubset(ekey):
            continue  # Cache missing required edge data; does not work
        if node_key is False or nkey is True:
            pass  # Cache works for node data!
        elif node_key is True or nkey is False or not node_key.issubset(nkey):
            continue  # Cache missing required node data; does not work
        if mutations is not None:
            # Remove this item from the cache (after all conversions) if
            # the call to this dispatchable function will mutate an input.
            mutations.append((cache, (ekey, nkey)))
        return (ekey, nkey), graph

    return None, None


def _set_to_cache(cache, key, graph, *, backend_name=None):
    """Set a backend graph to the cache, and remove unnecessary cached items.

    Parameters
    ----------
    cache : dict
        If ``backend_name`` is given, then this is treated as ``G.__networkx_cache__``,
        but if ``backend_name`` is None, then this is treated as the resolved inner
        cache such as ``G.__networkx_cache__["backends"][backend_name]``.
    key : tuple
        Cache key from ``_get_cache_key``.
    graph : graph or "FAILED_TO_CONVERT"
        Setting value to "FAILED_TO_CONVERT" prevents this conversion from being
        attempted in future calls.
    backend_name : str, optional
        Name of the backend to control how ``cache`` is interpreted.

    Returns
    -------
    dict
        The items that were removed from the cache.
    """
    if backend_name is not None:
        cache = cache.setdefault("backends", {}).setdefault(backend_name, {})
    # Remove old cached items that are no longer necessary since they
    # are dominated/subsumed/outdated by what was just calculated.
    # This uses the same logic as above, but with keys switched.
    # Also, don't update the cache here if the call will mutate an input.
    removed = {}
    edge_key, node_key = key
    cache[key] = graph  # Set at beginning to be thread-safe
    if graph == FAILED_TO_CONVERT:
        return removed
    for cur_key in list(cache):
        if cur_key == key:
            continue
        ekey, nkey = cur_key
        if ekey is False or edge_key is True:
            pass
        elif ekey is True or edge_key is False or not ekey.issubset(edge_key):
            continue
        if nkey is False or node_key is True:
            pass
        elif nkey is True or node_key is False or not nkey.issubset(node_key):
            continue
        # Use pop instead of del to try to be thread-safe
        if (graph := cache.pop(cur_key, None)) is not None:
            removed[cur_key] = graph
    return removed


class _LazyArgsRepr:
    """Simple wrapper to display arguments of dispatchable functions in logging calls."""

    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.value = None

    def __repr__(self):
        if self.value is None:
            bound = self.func.__signature__.bind_partial(*self.args, **self.kwargs)
            inner = ", ".join(f"{key}={val!r}" for key, val in bound.arguments.items())
            self.value = f"({inner})"
        return self.value


if os.environ.get("_NETWORKX_BUILDING_DOCS_"):
    # When building docs with Sphinx, use the original function with the
    # dispatched __doc__, b/c Sphinx renders normal Python functions better.
    # This doesn't show e.g. `*, backend=None, **backend_kwargs` in the
    # signatures, which is probably okay. It does allow the docstring to be
    # updated based on the installed backends.
    _orig_dispatchable = _dispatchable

    def _dispatchable(func=None, **kwargs):  # type: ignore[no-redef]
        if func is None:
            return partial(_dispatchable, **kwargs)
        dispatched_func = _orig_dispatchable(func, **kwargs)
        func.__doc__ = dispatched_func.__doc__
        return func

    _dispatchable.__doc__ = _orig_dispatchable.__new__.__doc__  # type: ignore[method-assign,assignment]
    _sig = inspect.signature(_orig_dispatchable.__new__)
    _dispatchable.__signature__ = _sig.replace(  # type: ignore[method-assign,assignment]
        parameters=[v for k, v in _sig.parameters.items() if k != "cls"]
    )
