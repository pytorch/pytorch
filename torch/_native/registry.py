import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Concatenate, ParamSpec, TypeVar

import torch.library


__all__ = [
    "UserOrderingFn",
    "register_op_override",
    "reorder_graphs_from_user_function",
    "reenable_op_overrides",
    "deregister_op_overrides",
]

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_OpOverrideFn = Callable[Concatenate[torch.DispatchKeySet, P], R]
_OpReplaceFn = Callable[P, R]

_OpFn = _OpOverrideFn | _OpReplaceFn


@dataclass
class _OverrideNode:
    """Track function override data."""

    dsl_name: str
    op_symbol: str
    dispatch_key: str
    override_fn: _OpFn
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


def _get_or_create_library(op_symbol: str, dispatch_key: str) -> torch.library.Library:
    """
    Get or create a torch.library.Library instance for the given key.

    Args:
        op_symbol: The operation symbol
        dispatch_key: The dispatch key

    Returns:
        torch.library.Library: The library instance
    """
    global _libs

    key = (op_symbol, dispatch_key)
    if key not in _libs:
        _libs[key] = torch.library.Library("aten", "IMPL", dispatch_key)

    return _libs[key]


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
    lib.impl(
        node.op_symbol,
        node.override_fn,
        dispatch_key,
        with_keyset=not node.unconditional_override,
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


def register_op_override(
    backend: str,
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    impl: _OpOverrideFn | _OpReplaceFn,
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
        impl: Implementation function for the override
        allow_multiple_override: Allow overriding an existing override
        unconditional_override: Implementation doesn't have a fallback and
            doesn't require torch.DispatchKeySet as the first argument

    Raises:
        ValueError: If lib_symbol is not "aten"
    """
    if lib_symbol != "aten":
        raise ValueError(f'Unsupported lib_symbol (must be "aten", got: "{lib_symbol}"')

    key = (op_symbol, dispatch_key)

    global _graphs
    op_graph = _graphs.get(key, [])

    op_graph.append(
        _OverrideNode(
            dsl_name=backend,
            op_symbol=op_symbol,
            dispatch_key=dispatch_key,
            override_fn=impl,
            unconditional_override=unconditional_override,
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
    Clean up existing library and reregister a graph.

    This is the common pattern used across reorder, deregister, and reenable operations.

    Args:
        op_symbol: The operation symbol
        dispatch_key: The dispatch key
        graph: The graph to register
        filter_state: Optional filter state for conditional registration
    """
    key = (op_symbol, dispatch_key)

    # Remove existing library if it exists
    if key in _libs:
        del _libs[key]

    # Only create a library if the graph has nodes
    # Empty graphs (disabled operations) shouldn't get libraries
    if graph:
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
    key = (op_symbol, dispatch_key)
    lib = _get_or_create_library(*key)

    for node in graph:
        enable = True
        if filter_state:
            enable = filter_state.check_enabled(node)

        if enable:
            _register_node_impl(lib, node, dispatch_key)
            node.active = True
        else:
            node.active = False


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
