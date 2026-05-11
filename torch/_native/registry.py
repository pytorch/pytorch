import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar

import torch.library


__all__ = [
    "UserOrderingFn",
    "register_op_override",
    "reorder_graphs_from_user_function",
    "reenable_op_overrides",
    "deregister_op_overrides",
    "get_dsl_operations",
]

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_OpCondFn = Callable[P, bool]
_OpImplFn = Callable[P, R]


@dataclass
class _OverrideNode:
    """Track function override data."""

    dsl_name: str
    op_symbol: str
    dispatch_key: str
    cond_fn: _OpCondFn
    impl_fn: _OpImplFn
    # Identifier of the opaque `_native::<node_id>` op that carries this
    # override's impl. Assigned once at registration time and never reused —
    # reorder / deregister / reenable do not change it, so the cached
    # `_defined_native_ops` entry remains valid.
    node_id: str
    unconditional_override: bool = False
    active: bool = True


UserOrderingFn = Callable[[str, str, list[_OverrideNode]], list[_OverrideNode]]


@dataclass
class _FilterState:
    """Manages filtering state for override nodes."""

    _dsl_names: set[str] = field(default_factory=set)
    _op_symbols: set[str] = field(default_factory=set)
    _dispatch_keys: set[str] = field(default_factory=set)

    def check_enabled(self, node: _OverrideNode) -> bool:
        """
        Check if a node is enabled based on current filter state.

        Args:
            node: The override node to check

        Returns:
            bool: True if the node should be enabled, False if filtered out
        """
        if node.dsl_name in self._dsl_names:
            return False

        if node.op_symbol in self._op_symbols:
            return False

        if node.dispatch_key in self._dispatch_keys:
            return False

        return True

    def update(
        self,
        dsl_names: str | Iterable[str] | None,
        op_symbols: str | Iterable[str] | None,
        dispatch_keys: str | Iterable[str] | None,
        remove_keys: bool = False,
    ) -> None:
        """
        Update filter sets as (current | new) or (current ~ new).

        Args:
            dsl_names: DSL names to add/remove from filter
            op_symbols: Operation symbols to add/remove from filter
            dispatch_keys: Dispatch keys to add/remove from filter
            remove_keys: If True, remove keys from filter; if False, add them

        Note:
            Uses set.discard as it doesn't raise an exception if the element
            wasn't in the set to begin with.
        """
        if remove_keys:
            self._dsl_names -= set(_resolve_iterable(dsl_names))
            self._op_symbols -= set(_resolve_iterable(op_symbols))
            self._dispatch_keys -= set(_resolve_iterable(dispatch_keys))
        else:
            self._dsl_names |= set(_resolve_iterable(dsl_names))
            self._op_symbols |= set(_resolve_iterable(op_symbols))
            self._dispatch_keys |= set(_resolve_iterable(dispatch_keys))

    def build_disable_key_set(self) -> set[tuple[str, str]]:
        """
        Build a set of dictionary keys based on the current filter state.

        Returns:
            set[tuple[str, str]]: Set of (op_symbol, dispatch_key) tuples
        """
        return _build_key_set(
            self._dsl_names,
            self._op_symbols,
            self._dispatch_keys,
        )

    def __str__(self) -> str:
        """Return string representation of filter state."""
        s = ""
        s += "Filter State:\n"
        s += "  === DSL: ===\n"
        for i, dsl in enumerate(self._dsl_names):
            s += f"    {i}: {dsl}\n"
        s += "  === OP SYMBOL: ===\n"
        for i, op in enumerate(self._op_symbols):
            s += f"    {i}: {op}\n"
        s += "  === DISPATCH KEYS: ===\n"
        for i, key in enumerate(self._dispatch_keys):
            s += f"    {i}: {key}\n"

        return s


# Store the global override filtering state
_filter_state: _FilterState = _FilterState()

# Store torch.library.Library instances
_libs: dict[tuple[str, str], torch.library.Library] = {}

# store graph structures
_GraphsType = dict[tuple[str, str], list[_OverrideNode]]
_graphs: _GraphsType = {}

_MappingType = dict[str, list[tuple[str, str]]]

# map a {dsl, op, dispatch_key} to keys to all graphs that contain it
_dsl_name_to_lib_graph: _MappingType = {}
_dispatch_key_to_lib_graph: _MappingType = {}
_op_symbol_to_lib_graph: _MappingType = {}

# Monotonic counter used to mint stable, unique `_native::<id>` op names.
# Must never decrease across the process, since `_defined_native_ops`
# pins a fake kernel for each minted id.
_node_id_counter: int = 0

# Dispatch keys where installing an override would break the registry's
# assumptions. The eager router is wired at a backend key so aten's
# higher-priority Autograd/Autocast kernels handle those layers, and the
# fake kernel for `_native::<id>` redispatches to the aten op's meta — if
# the override lived at Meta or CompositeImplicitAutograd, that
# redispatch would loop back into the router.
_DISALLOWED_DISPATCH_KEYS: frozenset[str] = frozenset(
    {"Meta", "CompositeImplicitAutograd", "CompositeExplicitAutograd"}
)


def _build_key_set(
    dsl_names: str | Iterable[str] | None,
    op_symbols: str | Iterable[str] | None,
    dispatch_keys: str | Iterable[str] | None,
) -> set[tuple[str, str]]:
    """
    Build a set of dictionary keys based on filter criteria.

    Args:
        dsl_names: DSL names to include in key set
        op_symbols: Operation symbols to include in key set
        dispatch_keys: Dispatch keys to include in key set

    Returns:
        set[tuple[str, str]]: Set of (op_symbol, dispatch_key) tuples
    """
    key_set: set[tuple[str, str]] = set()

    def _append_to_set(
        entries: str | Iterable[str] | None, graph_lib_dict: _MappingType
    ) -> None:
        """Helper to add matching keys from graph_lib_dict to key_set."""
        resolved_entries = _resolve_iterable(entries)

        for entry in resolved_entries:
            if entry in graph_lib_dict:
                for key in graph_lib_dict[entry]:
                    key_set.add(key)

    _append_to_set(dsl_names, _dsl_name_to_lib_graph)
    _append_to_set(op_symbols, _op_symbol_to_lib_graph)
    _append_to_set(dispatch_keys, _dispatch_key_to_lib_graph)

    return key_set


def _print_override_graphs(*, print_inactive: bool = False) -> None:
    """
    Print all override graphs for debugging purposes.

    Args:
        print_inactive: Whether to print inactive nodes
    """
    for (op, key), node_list in _graphs.items():
        print(f"{op=}, {key=}")

        for i, node in enumerate(node_list):
            if node.active or print_inactive:
                s: str = f"    {i}: {node.dsl_name=}, {node.unconditional_override=}"
                if print_inactive:
                    s += f" {node.active=}"

                print(s)


# One DEF/FRAGMENT library per namespace — only one is allowed per process.
_def_libs: dict[str, torch.library.Library] = {}
# Ops that have already been `define`d on the _native namespace.
_defined_native_ops: set[str] = set()
# IMPL libraries on the aten namespace, keyed by (op_symbol, dispatch_key).
# One library per op/key pair so we can call `_destroy()` on it to tear down
# just that one override without affecting other ops at the same dispatch
# key. Torch's Library has no per-kernel removal API — `_destroy` on the
# library is the only teardown mechanism.
_aten_override_libs: dict[tuple[str, str], torch.library.Library] = {}


def _get_def_library(namespace: str) -> torch.library.Library:
    if namespace not in _def_libs:
        _def_libs[namespace] = torch.library.Library(namespace, "FRAGMENT")
    return _def_libs[namespace]


def _get_or_create_library(dispatch_key: str) -> torch.library.Library:
    """
    Get or create the _native IMPL library for a given dispatch key.

    One library per dispatch key is shared across all overridden ops.
    """
    global _libs

    key = ("_native", dispatch_key)
    if key not in _libs:
        _libs[key] = torch.library.Library("_native", "IMPL", dispatch_key)

    return _libs[key]


def _install_aten_override(op_symbol: str, dispatch_key: str, kernel: Callable) -> None:
    """
    Install (or replace) an aten kernel at (op_symbol, dispatch_key).

    Creates a fresh Library per (op, key) so we can tear down just this
    one override via `_destroy_aten_override` without affecting any other
    override at the same dispatch key.
    """
    key = (op_symbol, dispatch_key)
    # Destroy any existing library so its kernel is fully removed before
    # we install the new one. `_destroy` calls into the dispatcher to
    # unregister all kernels on the library.
    existing = _aten_override_libs.pop(key, None)
    if existing is not None:
        existing._destroy()

    lib = torch.library.Library("aten", "IMPL", dispatch_key)
    lib.impl(op_symbol, kernel, dispatch_key, with_keyset=True)
    _aten_override_libs[key] = lib


def _destroy_aten_override(op_symbol: str, dispatch_key: str) -> None:
    """Tear down the aten override at (op_symbol, dispatch_key), if any."""
    lib = _aten_override_libs.pop((op_symbol, dispatch_key), None)
    if lib is not None:
        lib._destroy()


def _resolve_aten_overload(op_symbol: str) -> "torch._ops.OpOverload | None":
    """
    Resolve `op_symbol` to a concrete OpOverload on `torch.ops.aten`.

    Accepts bare names ("bmm" → aten.bmm.default) and overload-qualified
    names ("add_.Tensor" → aten.add_.Tensor). Returns None if the op is not
    registered (e.g. a test-only op_symbol that never hit the C++ dispatcher).
    """
    name, _, overload_name = op_symbol.partition(".")
    overload_name = overload_name or "default"
    try:
        packet = getattr(torch.ops.aten, name)
        return getattr(packet, overload_name)
    except AttributeError:
        return None


def _aten_schema_tail(op_symbol: str) -> str:
    """Return the schema of at::<op_symbol> with the `aten::<name>` prefix stripped.

    Accepts bare names ("bmm" → aten.bmm.default) and overload-qualified
    names ("add_.Tensor" → aten.add_.Tensor).
    """
    overload = _resolve_aten_overload(op_symbol)
    if overload is None:
        raise AttributeError(f"aten op not found for op_symbol={op_symbol!r}")
    s = str(overload._schema)
    # "aten::bmm(Tensor self, Tensor mat2) -> Tensor" -> "(Tensor self, Tensor mat2) -> Tensor"
    _, rest = s.split("::", 1)
    _, args = rest.split("(", 1)
    return f"({args}"


def _define_native_op_once(name: str, op_symbol: str) -> None:
    # Invariant: callers must only install the eager router at a real backend
    # dispatch key (CPU, CUDA, XPU, ...). The fake kernel below redispatches
    # to the aten op, and if the router were installed at Meta /
    # CompositeImplicitAutograd that redispatch would re-enter the router.
    # `register_op_override` enforces this via `_DISALLOWED_DISPATCH_KEYS`.
    if name in _defined_native_ops:
        return
    _get_def_library("_native").define(f"{name}{_aten_schema_tail(op_symbol)}")
    # Fake/meta kernel: required so export / dynamo / AOTAutograd can shape-infer
    # through the opaque _native op. Reusing the aten op's meta is safe because
    # the schema (and therefore shape rules) is cloned from it.
    aten_overload = _resolve_aten_overload(op_symbol)
    torch.library.register_fake(f"_native::{name}")(
        lambda *args, _aten_overload=aten_overload, **kwargs: _aten_overload(
            *args, **kwargs
        )
    )
    # No autograd or autocast kernels are registered on _native::<id>. Because
    # the router is registered at the backend dispatch key (e.g. CUDA), aten's
    # own higher-priority kernels on aten::<op> handle both:
    #   - Autograd: AutogradCUDA sees aten::<op> in the autograd graph and
    #     uses aten's built-in derivative formula.
    #   - Autocast: AutocastCUDA casts inputs to the autocast dtype before
    #     redispatching down to our CUDA-level router.
    _defined_native_ops.add(name)


def _register_node_impl(
    lib: torch.library.Library, node: _OverrideNode, dispatch_key: str
) -> None:
    """
    Register a single node implementation with the library.

    Args:
        lib: The torch.library.Library instance
        node: The override node to register
        dispatch_key: The dispatch key for registration
    """
    if not node.node_id:
        raise ValueError(
            f"_OverrideNode must have a non-empty node_id before registration "
            f"(dsl_name={node.dsl_name!r}, op_symbol={node.op_symbol!r})"
        )
    _define_native_op_once(node.node_id, node.op_symbol)
    lib.impl(
        node.node_id,
        node.impl_fn,
        dispatch_key,
        with_keyset=False,
        allow_override=True,
    )


def _resolve_iterable(iterable: str | Iterable[str] | None) -> Iterable[str]:
    """
    Resolve various input types to a consistent iterable of strings.

    Args:
        iterable: String, iterable of strings, or None

    Returns:
        Iterable[str]: Consistent iterable output
    """
    if iterable is None:
        return []

    if not isinstance(iterable, Iterable) or isinstance(iterable, str):
        return (iterable,)

    return iterable


def reenable_op_overrides(
    *,
    enable_dsl_names: str | list[str] | None = None,
    enable_op_symbols: str | list[str] | None = None,
    enable_dispatch_keys: str | list[str] | None = None,
) -> None:
    """
    Re-enable overrides by removing them from filter state and reregistering.

    Args:
        enable_dsl_names: DSL names to re-enable
        enable_op_symbols: Operation symbols to re-enable
        enable_dispatch_keys: Dispatch keys to re-enable

    Note:
        This function uses reverse filter state management (removing from
        filters to enable).
    """
    log.info(
        "Re-registering ops by dsl: %s, op_symbol: %s, dispatch_key: %s",
        enable_dsl_names,
        enable_op_symbols,
        enable_dispatch_keys,
    )

    # Update the filters - note `remove_keys=True` because
    # we are removing keys from the filters (vs. adding them)
    _filter_state.update(
        enable_dsl_names,
        enable_op_symbols,
        enable_dispatch_keys,
        remove_keys=True,
    )

    # Get the set of keys that need to be reprocessed
    key_set: set[tuple[str, str]] = _build_key_set(
        enable_dsl_names,
        enable_op_symbols,
        enable_dispatch_keys,
    )

    # Process each affected graph with updated filter state
    for key in key_set:
        op_symbol, dispatch_key = key

        if key in _graphs:
            # Note: We don't need to cleanup and recreate the library here
            # since we're just updating the registration with new filter state
            _register_overrides_from_graph(
                op_symbol, dispatch_key, _graphs[key], filter_state=_filter_state
            )


def deregister_op_overrides(
    *,
    disable_dsl_names: str | list[str] | None = None,
    disable_op_symbols: str | list[str] | None = None,
    disable_dispatch_keys: str | list[str] | None = None,
) -> None:
    """
    De-register overrides by updating filter state and reregistering graphs.

    Args:
        disable_dsl_names: DSL names to disable
        disable_op_symbols: Operation symbols to disable
        disable_dispatch_keys: Dispatch keys to disable

    Note:
        This function uses filter state management to selectively disable
        operations.
    """
    log.info(
        "De-registering ops by dsl: %s, op_symbol: %s, dispatch_key: %s",
        disable_dsl_names,
        disable_op_symbols,
        disable_dispatch_keys,
    )

    # Update filter state to disable specified entries
    _filter_state.update(disable_dsl_names, disable_op_symbols, disable_dispatch_keys)

    # Get the set of keys that need to be reprocessed
    key_set: set[tuple[str, str]] = _filter_state.build_disable_key_set()

    # Process each affected graph with filter state
    for key in key_set:
        op_symbol, dispatch_key = key

        if key in _graphs:
            _cleanup_and_reregister_graph(
                op_symbol,
                dispatch_key,
                _graphs[key],
                filter_state=_filter_state,
            )


def get_dsl_operations(dsl_name: str) -> list[str]:
    """Get list of operations registered by a specific DSL.

    Args:
        dsl_name: Name of the DSL to query.

    Returns:
        Sorted list of operation names registered by the DSL.
    """
    operations = set()
    for (op_symbol, _), nodes in _graphs.items():
        for node in nodes:
            if node.dsl_name == dsl_name:
                operations.add(op_symbol)
                break
    return sorted(operations)


def _update_registration_maps(
    dsl_name: str,
    op_symbol: str,
    dispatch_key: str,
    key: tuple[str, str],
) -> None:
    """
    Update the registration mapping dictionaries.

    Args:
        dsl_name: The DSL name
        op_symbol: The operation symbol
        dispatch_key: The dispatch key
        key: The dictionary key tuple
    """
    global _dsl_name_to_lib_graph
    global _op_symbol_to_lib_graph
    global _dispatch_key_to_lib_graph

    def _get_new_entry_or_append(
        registration: dict[str, list[tuple[str, str]]],
        symbol: str,
        key: tuple[str, str],
    ) -> None:
        """Helper to add key to registration list or create new entry."""
        entry_list = registration.get(symbol)

        if entry_list is None:
            entry_list = [key]
            registration[symbol] = entry_list
        else:
            entry_list.append(key)

    _get_new_entry_or_append(_dsl_name_to_lib_graph, dsl_name, key)
    _get_new_entry_or_append(_op_symbol_to_lib_graph, op_symbol, key)
    _get_new_entry_or_append(_dispatch_key_to_lib_graph, dispatch_key, key)


def _always_true(*args: object, **kwargs: object) -> bool:
    return True


def register_op_override(
    backend: str,
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    cond: _OpCondFn | None,
    impl: _OpImplFn,
    *,
    allow_multiple_override: bool = False,
    unconditional_override: bool = False,
) -> None:
    """
    Register a passed override function to the dispatcher.

    Actually a graph-building operation; real registration happens later.

    Args:
        backend: The backend name (DSL name)
        lib_symbol: Library you're overriding symbols in (must be "aten")
        op_symbol: Name of the operation you're overriding
        dispatch_key: Dispatch key to override
        cond: Predicate choosing whether `impl` applies to a given call. May
            be None if `unconditional_override=True`.
        impl: Implementation function for the override
        allow_multiple_override: Allow overriding an existing override
        unconditional_override: Implementation doesn't have a fallback and
            doesn't require torch.DispatchKeySet as the first argument. When
            True, a trivially-True predicate is supplied for the router if
            `cond` is None.

    Raises:
        ValueError: If lib_symbol is not "aten", if dispatch_key is in
            _DISALLOWED_DISPATCH_KEYS (Meta / CompositeImplicitAutograd /
            CompositeExplicitAutograd), or if cond is None without
            unconditional_override=True.
    """
    if lib_symbol != "aten":
        raise ValueError(f'Unsupported lib_symbol (must be "aten", got: "{lib_symbol}"')

    if dispatch_key in _DISALLOWED_DISPATCH_KEYS:
        raise ValueError(
            f"dispatch_key={dispatch_key!r} is not supported. Overrides must be "
            f"installed at a backend key (e.g. CPU, CUDA, XPU); the router's fake "
            f"kernel redispatches to aten and would recurse otherwise."
        )

    if cond is None:
        if not unconditional_override:
            raise ValueError("cond must be provided unless unconditional_override=True")
        cond = _always_true

    key = (op_symbol, dispatch_key)

    global _graphs, _node_id_counter
    op_graph = _graphs.get(key, [])

    # Mint a stable id for the opaque `_native::<node_id>` op. `_sanitized`
    # strips dots from overload-qualified names so the result is a valid op
    # name. The monotonic counter guarantees uniqueness even if the same
    # (op_symbol, dsl_name) is registered, deregistered, and re-registered.
    _sanitized = op_symbol.replace(".", "_")
    node_id = f"{_sanitized}_{backend}_{_node_id_counter}"
    _node_id_counter += 1

    op_graph.append(
        _OverrideNode(
            dsl_name=backend,
            op_symbol=op_symbol,
            dispatch_key=dispatch_key,
            cond_fn=cond,
            impl_fn=impl,
            unconditional_override=unconditional_override,
            node_id=node_id,
        )
    )
    _graphs[key] = op_graph
    # Build additional maps helpful for de-registration
    _update_registration_maps(backend, op_symbol, dispatch_key, key=key)


def _should_reregister_graph(
    original_graph: list[_OverrideNode],
    new_graph: list[_OverrideNode],
    *,
    force_reregister: bool = False,
) -> bool:
    """
    Determine if a graph needs reregistration based on changes.

    Args:
        original_graph: The original graph before modification
        new_graph: The graph after modification
        force_reregister: If True, always reregister regardless of changes

    Returns:
        bool: True if reregistration is needed
    """
    if force_reregister:
        return True

    # Check if the graph structure has changed
    return original_graph != new_graph


def _cleanup_and_reregister_graph(
    op_symbol: str,
    dispatch_key: str,
    graph: list[_OverrideNode],
    *,
    filter_state: _FilterState | None = None,
) -> None:
    """
    Reregister a graph's routes from scratch.

    Used by reorder / deregister / reenable. Libraries are intentionally
    long-lived singletons; we rebuild the per-op router closure here.

    Args:
        op_symbol: The operation symbol
        dispatch_key: The dispatch key
        graph: The graph to register
        filter_state: Optional filter state for conditional registration
    """
    _register_overrides_from_graph(
        op_symbol,
        dispatch_key,
        graph,
        filter_state=filter_state,
    )


def _apply_graph_transformation(
    transformation_fn: UserOrderingFn,
    *,
    keys_to_process: set[tuple[str, str]] | None = None,
    reregister_overrides: bool = False,
    filter_state: _FilterState | None = None,
) -> None:
    """
    Apply a transformation function to graphs and optionally reregister.

    This is the core pattern used by reorder_graphs_from_user_function and
    can be reused for other graph transformation operations.

    Args:
        transformation_fn: Function to transform each graph
        keys_to_process: Keys to process, or None for all graphs
        reregister_overrides: Whether to reregister changed graphs
        filter_state: Optional filter state for conditional registration

    Note:
        If transformation_fn raises an exception for a specific graph, that graph
        will be skipped and processing will continue with remaining graphs.
    """
    global _graphs

    # Determine which graphs to process
    target_keys = (
        keys_to_process if keys_to_process is not None else set(_graphs.keys())
    )

    # Process each graph
    for op_symbol, dispatch_key in list(target_keys):
        if (op_symbol, dispatch_key) not in _graphs:
            continue  # Skip if graph doesn't exist

        original_graph = list(_graphs[(op_symbol, dispatch_key)])

        # Apply the transformation with error handling
        try:
            new_graph = transformation_fn(op_symbol, dispatch_key, original_graph)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            log.warning(
                "Graph transformation failed for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
                exc_info=True,
            )
            continue
        except Exception:
            log.exception(
                "Unexpected error in graph transformation for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
            )
            continue

        # Validate that the transformation returned a valid result
        if not isinstance(new_graph, list):
            log.warning(
                "Graph transformation returned invalid type %s for %s/%s. Expected list. Preserving original graph.",
                type(new_graph).__name__,
                op_symbol,
                dispatch_key,
            )
            continue

        # Update the graph
        _graphs[(op_symbol, dispatch_key)] = new_graph

        # Reregister if needed
        if reregister_overrides and _should_reregister_graph(
            original_graph, new_graph, force_reregister=False
        ):
            _cleanup_and_reregister_graph(
                op_symbol,
                dispatch_key,
                new_graph,
                filter_state=filter_state,
            )


def _register_overrides_from_graph(
    op_symbol: str,
    dispatch_key: str,
    graph: list[_OverrideNode],
    *,
    filter_state: _FilterState | None = None,
) -> None:
    """
    Register all overrides in a single graph.

    Args:
        op_symbol: The operation symbol
        dispatch_key: The dispatch key
        graph: List of override nodes to register
        filter_state: Optional filter state for conditional registration
    """
    lib = _get_or_create_library(dispatch_key)

    cond_impl: list[tuple[_OpCondFn, str]] = []

    # node.node_id is minted once at `register_op_override` time and is
    # stable across reorder / deregister / reenable — never regenerate it
    # here, since `_defined_native_ops` pins a fake kernel against each id.
    for node in graph:
        enable = True
        if filter_state:
            enable = filter_state.check_enabled(node)

        if enable:
            _register_node_impl(lib, node, dispatch_key)
            cond_impl.append((node.cond_fn, node.node_id))
            node.active = True
        else:
            node.active = False

    # Tear down any existing aten override so either (a) the op cleanly
    # reverts to native aten (empty cond_impl), or (b) the subsequent
    # `get_kernel` call returns the native kernel rather than a stale
    # previously-installed router.
    _destroy_aten_override(op_symbol, dispatch_key)

    # If no active conds remain for this (op, key), leave the native op
    # behavior intact.
    if not cond_impl:
        return

    # Capture the prior kernel at this (op, dispatch_key) *before* we install
    # our override. The fallback path calls it via `call_boxed`, which
    # re-enters the dispatcher with the original kernel handle — bypassing
    # our just-registered router and avoiding the recursion that would
    # otherwise appear in aten's backward formulas (e.g. bmm's backward
    # calls bmm, which would route back to us).
    fallback_kernel = torch.library.get_kernel(f"aten::{op_symbol}", dispatch_key)

    # First-match-wins dispatch over `cond_impl`. Factored into a helper so
    # the router closure stays small and so other routers (e.g. for the
    # compile/export path) can share the same matching logic.
    _NO_MATCH = object()  # sentinel; impl return values of None would be valid outputs

    def _dispatch(args, kwargs, swallow_cond_exceptions: bool):
        for cond, impl_name in cond_impl:
            try:
                matched = cond(*args, **kwargs)
            except Exception:
                if not swallow_cond_exceptions:
                    raise
                continue
            if matched:
                return getattr(torch.ops._native, impl_name)(*args, **kwargs)
        return _NO_MATCH

    # Eager router: cond exceptions indicate a genuine bug (we're on real
    # tensors); missing a match falls back to the captured native kernel.
    def eager_router(keyset, *args, _fallback=fallback_kernel, **kwargs):
        result = _dispatch(args, kwargs, swallow_cond_exceptions=False)
        if result is _NO_MATCH:
            return _fallback.call_boxed(keyset, *args, **kwargs)
        return result

    # Install a fresh aten override for this (op, key).
    _install_aten_override(op_symbol, dispatch_key, eager_router)


def _register_all_overrides() -> None:
    """
    Perform all registration calls from previously-built override graphs.
    """
    for key, graph in _graphs.items():
        op_symbol, dispatch_key = key

        _register_overrides_from_graph(
            op_symbol,
            dispatch_key,
            graph,
        )


def reorder_graphs_from_user_function(
    fn: UserOrderingFn,
    *,
    reregister_overrides: bool = False,
) -> None:
    """
    Reorder override graphs using a user-provided ordering function.

    Args:
        fn: User-provided function that takes (op_symbol, dispatch_key, graph)
            and returns a reordered graph
        reregister_overrides: Whether to reregister graphs that have changed

    Note:
        This function uses the common graph transformation pattern and can serve
        as an example for other graph manipulation operations.
    """
    _apply_graph_transformation(
        transformation_fn=fn,
        reregister_overrides=reregister_overrides,
    )


def _apply_graph_filter(
    filter_fn: Callable[[str, str, _OverrideNode], bool],
    *,
    reregister_overrides: bool = False,
) -> None:
    """
    Apply a filter function to remove nodes from graphs.

    This is a convenience function that uses the graph transformation pattern
    to filter out unwanted nodes.

    Args:
        filter_fn: Function that takes (op_symbol, dispatch_key, node) and
            returns True to keep the node, False to remove it
        reregister_overrides: Whether to reregister modified graphs

    Example:
        # Remove all nodes with "deprecated" in the DSL name
        _apply_graph_filter(
            lambda op, dk, node: "deprecated" not in node.dsl_name,
            reregister_overrides=True
        )

    Note:
        If filter_fn raises an exception for a specific graph, the original
        graph will be preserved and processing will continue.
    """

    def filtering_transformation(
        op_symbol: str, dispatch_key: str, graph: list[_OverrideNode]
    ) -> list[_OverrideNode]:
        """Apply filter_fn to graph with error handling."""
        try:
            return [node for node in graph if filter_fn(op_symbol, dispatch_key, node)]
        except (TypeError, ValueError, AttributeError, RuntimeError):
            log.warning(
                "Graph transformation failed for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
                exc_info=True,
            )
            return graph
        except Exception:
            log.exception(
                "Unexpected error in graph transformation for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
            )
            return graph

    _apply_graph_transformation(
        transformation_fn=filtering_transformation,
        reregister_overrides=reregister_overrides,
    )


def _apply_selective_reordering(
    condition_fn: Callable[[str, str], bool],
    ordering_fn: UserOrderingFn,
    *,
    reregister_overrides: bool = False,
) -> None:
    """
    Apply reordering only to graphs that match a condition.

    This allows for more targeted reordering operations.

    Args:
        condition_fn: Function that takes (op_symbol, dispatch_key) and
            returns True if the graph should be reordered
        ordering_fn: Ordering function to apply to matching graphs
        reregister_overrides: Whether to reregister modified graphs

    Example:
        # Only reorder CUDA operations
        _apply_selective_reordering(
            condition_fn=lambda op, dk: dk == "CUDA",
            ordering_fn=lambda op, dk, g: sorted(g, key=lambda n: n.dsl_name),
            reregister_overrides=True
        )

    Note:
        If condition_fn or ordering_fn raises an exception for a specific graph,
        the original graph will be preserved and processing will continue.
    """

    def conditional_transformation(
        op_symbol: str, dispatch_key: str, graph: list[_OverrideNode]
    ) -> list[_OverrideNode]:
        """Apply ordering_fn conditionally based on condition_fn result."""
        try:
            should_reorder = condition_fn(op_symbol, dispatch_key)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            log.warning(
                "Graph transformation failed for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
                exc_info=True,
            )
            return graph
        except Exception:
            log.exception(
                "Unexpected error in graph transformation for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
            )
            return graph

        if should_reorder:
            try:
                return ordering_fn(op_symbol, dispatch_key, graph)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                log.warning(
                    "Graph transformation failed for %s/%s. Preserving original graph.",
                    op_symbol,
                    dispatch_key,
                    exc_info=True,
                )
                return graph
            except Exception:
                log.exception(
                    "Unexpected error in graph transformation for %s/%s. Preserving original graph.",
                    op_symbol,
                    dispatch_key,
                )
                return graph

        return graph  # Return unchanged if condition doesn't match

    _apply_graph_transformation(
        transformation_fn=conditional_transformation,
        reregister_overrides=reregister_overrides,
    )
