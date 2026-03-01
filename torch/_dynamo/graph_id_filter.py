# mypy: disallow-untyped-defs

from __future__ import annotations

import functools
import logging
import re
from typing import Any, Generic, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._guards import CompileId


log = logging.getLogger(__name__)

# Valid modes for inductor backend
_INDUCTOR_MODES = frozenset(
    {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
)


class GraphIdFilter:
    """
    A filter for matching graph IDs based on various conditions.
    Supports individual IDs, ranges, and comparison operators.
    """

    def __init__(self, filter_str: str) -> None:
        self._explicit_ids: frozenset[int] = frozenset()
        self._conditions: list[tuple[str, int]] = []
        self._parse(filter_str)

    def _parse(self, filter_str: str) -> None:
        if not filter_str or not filter_str.strip():
            return

        explicit_ids: set[int] = set()
        conditions: list[tuple[str, int]] = []

        # Pattern for comparison operators (>=, >, <=, <) followed by a number
        cmp_pattern = re.compile(r"^(>=|>|<=|<)(\d+)$")
        # Pattern for ranges like "10-20"
        range_pattern = re.compile(r"^(\d+)-(\d+)$")

        for part in filter_str.split(","):
            part = part.strip()
            if not part:
                continue

            if match := cmp_pattern.match(part):
                conditions.append((match.group(1), int(match.group(2))))
            elif match := range_pattern.match(part):
                start, end = int(match.group(1)), int(match.group(2))
                explicit_ids.update(range(start, end + 1))
            else:
                try:
                    explicit_ids.add(int(part))
                except ValueError:
                    log.warning("Invalid graph ID filter: %s", part)

        self._explicit_ids = frozenset(explicit_ids)
        self._conditions = conditions

    def __contains__(self, graph_id: int) -> bool:
        """Check if the given graph ID matches this filter."""
        if graph_id in self._explicit_ids:
            return True

        for op, val in self._conditions:
            if op == ">" and graph_id > val:
                return True
            elif op == ">=" and graph_id >= val:
                return True
            elif op == "<" and graph_id < val:
                return True
            elif op == "<=" and graph_id <= val:
                return True

        return False

    def __repr__(self) -> str:
        # pyrefly: ignore [implicit-any]
        parts = []
        if self._explicit_ids:
            parts.append(f"ids={sorted(self._explicit_ids)}")
        if self._conditions:
            parts.append(f"conditions={self._conditions}")
        return f"GraphIdFilter({', '.join(parts) if parts else 'empty'})"


T = TypeVar("T")


class _GraphRouterBase(Generic[T]):
    """
    Base class for routing graphs to different values based on their IDs.

    The router parses a configuration string with rules in the format:
        "filter1:value1;filter2:value2;..."

    Rules are evaluated in order, and the first matching rule wins.
    """

    def __init__(self, config_str: str, rule_type: str) -> None:
        self._rules: list[tuple[GraphIdFilter, T]] = []
        self._values: list[T | None] = []
        self._overflow_value: T | None = None
        self._rule_type = rule_type
        self._parse(config_str)
        self._precompute()

    def _parse_value_str(self, value_str: str) -> T | None:
        """Parse a value string into the appropriate type. Returns None to skip."""
        raise NotImplementedError

    def _parse(self, config_str: str) -> None:
        if not config_str or not config_str.strip():
            return

        rule_strs = config_str.split(";")
        for rule_str in rule_strs:
            rule_str = rule_str.strip()
            if not rule_str:
                continue

            colon_idx = rule_str.find(":")
            if colon_idx == -1:
                log.warning(
                    "Invalid %s override rule (missing ':'): %s",
                    self._rule_type,
                    rule_str,
                )
                continue

            filter_str = rule_str[:colon_idx].strip()
            value_str = rule_str[colon_idx + 1 :].strip()

            if not filter_str or not value_str:
                log.warning("Invalid %s override rule: %s", self._rule_type, rule_str)
                continue

            value = self._parse_value_str(value_str)
            if value is not None:
                self._rules.append((GraphIdFilter(filter_str), value))

    def _precompute(self) -> None:
        if not self._rules:
            return

        # Find max ID from explicit IDs and comparison thresholds
        max_id = 0
        for f, _ in self._rules:
            if f._explicit_ids:
                max_id = max(max_id, *f._explicit_ids)
            for _, val in f._conditions:
                max_id = max(max_id, val)

        # Pre-compute values for IDs 0 to max_id
        for i in range(max_id + 1):
            self._values.append(self._match_rules(i))

        # For IDs > max_id, the result is constant (only unbounded conditions apply)
        self._overflow_value = self._match_rules(max_id + 1)

    def _match_rules(self, graph_id: int) -> T | None:
        for f, value in self._rules:
            if graph_id in f:
                return value
        return None

    def get_value_for_graph(self, graph_id: int) -> T | None:
        """Get the value for a given graph ID. Returns None if no rule matches."""
        if graph_id < len(self._values):
            return self._values[graph_id]
        return self._overflow_value

    def is_empty(self) -> bool:
        """Check if no rules are configured."""
        return len(self._rules) == 0


class GraphBackendRouter(_GraphRouterBase[Any]):
    """
    Routes graphs to different backends based on their IDs.

    The router parses a configuration string with rules in the format:
        "filter1:backend1;filter2:backend2;..."

    Rules are evaluated in order, and the first matching rule wins.

    Examples:
        "0-5:eager;>5:inductor"     - IDs 0-5 use eager, rest use inductor
        ">10:aot_eager"             - IDs > 10 use aot_eager
        "<=3:eager;4-10:aot_eager"  - IDs 0-3 use eager, 4-10 use aot_eager

    Special backend values:
        "eager"                     - Run in eager mode (no compilation)
        "aot_eager"                 - AOT Autograd with eager execution
        "aot_eager_decomp_partition" - AOT Autograd with partitioner and decomps
        "inductor"                  - Default inductor backend
        "inductor:reduce-overhead"  - Inductor with reduce-overhead mode (cudagraphs)
        "inductor:max-autotune"     - Inductor with max-autotune mode
    """

    def __init__(self, config_str: str) -> None:
        super().__init__(config_str, "backend")

    def _parse_value_str(self, value_str: str) -> Any | None:
        """Look up a backend, supporting 'backend:mode' format for inductor."""
        import torch

        from .backends.registry import lookup_backend
        from .eval_frame import cached_backends

        backend: Any = None
        if ":" in value_str:
            parts = value_str.split(":", 1)
            backend_name, mode = parts[0], parts[1]

            if backend_name == "inductor" and mode in _INDUCTOR_MODES:
                backend = torch._TorchCompileInductorWrapper(
                    mode=mode, options=None, dynamic=None
                )

        if backend is None:
            backend = lookup_backend(value_str)

        # Register the backend so its reset() is called during torch._dynamo.reset()
        assert backend is not None, "Invalid override backend: " + value_str
        cached_backends.setdefault(id(backend), backend)
        return backend

    def __repr__(self) -> str:
        if not self._rules:
            return "GraphBackendRouter(empty)"
        return f"GraphBackendRouter({self._rules})"


class GraphConfigRouter(_GraphRouterBase[dict[str, Any]]):
    """
    Routes graphs to different inductor configs based on their IDs.

    The router parses a configuration string with rules in the format:
        "filter1:config1;filter2:config2;..."

    Rules are evaluated in order, and the first matching rule wins.
    Config format is "key=value" or "key1=value1,key2=value2" for multiple settings.

    Examples:
        "0-5:triton.cudagraph_skip_dynamic_graphs=False"
        ">10:triton.cudagraphs=False,triton.cudagraph_trees=False"
    """

    def __init__(self, config_str: str) -> None:
        super().__init__(config_str, "config")

    @staticmethod
    def _parse_scalar_value(value_str: str) -> Any:
        """Parse a string value into the appropriate Python type."""
        value_str = value_str.strip()
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False
        if value_str.lower() == "none":
            return None
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            return value_str

    def _parse_value_str(self, value_str: str) -> dict[str, Any] | None:
        """Parse a config string like 'key1=val1,key2=val2' into a dict."""
        result: dict[str, Any] = {}
        for item in value_str.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                log.warning("Invalid config item (missing '='): %s", item)
                continue
            key, value = item.split("=", 1)
            result[key.strip()] = self._parse_scalar_value(value)
        return result if result else None

    def __repr__(self) -> str:
        if not self._rules:
            return "GraphConfigRouter(empty)"
        return f"GraphConfigRouter({self._rules})"


def _get_override_for_compile_id(
    compile_id: CompileId | None,
    config_str: str,
    create_router: Callable[[str], _GraphRouterBase[T]],
    log_msg: str,
) -> T | None:
    """
    Get the override value for a given CompileId.

    Returns the value from the router, or None if no override applies.
    """
    if compile_id is None or not config_str:
        return None

    graph_id = compile_id.frame_id
    if graph_id is None:
        return None

    router = create_router(config_str)
    value = router.get_value_for_graph(graph_id)
    if value is not None:
        log.info(log_msg, compile_id, graph_id, value)
    return value


@functools.lru_cache
def _create_backend_router(config_str: str) -> GraphBackendRouter:
    """Create and cache GraphBackendRouter instances based on config string."""
    return GraphBackendRouter(config_str)


@functools.lru_cache
def _create_config_router(config_str: str) -> GraphConfigRouter:
    """Create and cache GraphConfigRouter instances based on config string."""
    return GraphConfigRouter(config_str)


def get_backend_override_for_compile_id(
    compile_id: CompileId | None,
    config_str: str,
) -> Any:
    """
    Get the backend override for a given CompileId.

    Returns the backend function to use, or None if no override applies.
    """
    return _get_override_for_compile_id(
        compile_id,
        config_str,
        _create_backend_router,
        "Graph %s (frame_id=%d) overridden to use backend: %s",
    )


def get_inductor_config_override_for_compile_id(
    compile_id: CompileId | None,
    config_str: str,
) -> dict[str, Any] | None:
    """
    Get the inductor config override for a given CompileId.

    Returns a dict of config patches to apply, or None if no override applies.
    """
    return _get_override_for_compile_id(
        compile_id,
        config_str,
        _create_config_router,  # type: ignore[arg-type]
        "Graph %s (frame_id=%d) overridden with inductor config: %s",
    )


# Keep old name for backwards compatibility
_create_router = _create_backend_router
