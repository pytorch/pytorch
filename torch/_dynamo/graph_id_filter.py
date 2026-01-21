# mypy: disallow-untyped-defs

from __future__ import annotations

import functools
import logging
import re
from typing import Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from torch._guards import CompileId


log = logging.getLogger(__name__)

# Valid modes for inductor backend
_INDUCTOR_MODES = frozenset(
    {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
)


def lookup_backend_with_mode(backend_str: str) -> Any:
    """
    Look up a backend, supporting 'backend:mode' format for inductor.
    """
    import torch

    from .backends.registry import lookup_backend

    if ":" in backend_str:
        parts = backend_str.split(":", 1)
        backend_name, mode = parts[0], parts[1]

        if backend_name == "inductor" and mode in _INDUCTOR_MODES:
            return torch._TorchCompileInductorWrapper(
                mode=mode, options=None, dynamic=None
            )

    return lookup_backend(backend_str)


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
        parts = []
        if self._explicit_ids:
            parts.append(f"ids={sorted(self._explicit_ids)}")
        if self._conditions:
            parts.append(f"conditions={self._conditions}")
        return f"GraphIdFilter({', '.join(parts) if parts else 'empty'})"


class GraphBackendRouter:
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
        self._rules: list[tuple[GraphIdFilter, str]] = []
        self._backends: list[Optional[str]] = []
        self._overflow_backend: Optional[str] = None
        self._parse(config_str)
        self._precompute()

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
                log.warning("Invalid backend override rule (missing ':'): %s", rule_str)
                continue

            filter_str = rule_str[:colon_idx].strip()
            backend = rule_str[colon_idx + 1 :].strip()

            if not filter_str or not backend:
                log.warning("Invalid backend override rule: %s", rule_str)
                continue

            self._rules.append((GraphIdFilter(filter_str), backend))

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

        # Pre-compute backends for IDs 0 to max_id
        for i in range(max_id + 1):
            self._backends.append(self._match_rules(i))

        # For IDs > max_id, the result is constant (only unbounded conditions apply)
        self._overflow_backend = self._match_rules(max_id + 1)

    def _match_rules(self, graph_id: int) -> Optional[str]:
        for f, backend in self._rules:
            if graph_id in f:
                return backend
        return None

    def get_backend_for_graph(self, graph_id: int) -> Optional[str]:
        """
        Get the backend override for a given graph ID.
        Returns None if no override matches.
        """
        if graph_id < len(self._backends):
            return self._backends[graph_id]
        return self._overflow_backend

    def is_empty(self) -> bool:
        """Check if no rules are configured."""
        return len(self._rules) == 0

    def __repr__(self) -> str:
        if not self._rules:
            return "GraphBackendRouter(empty)"
        return f"GraphBackendRouter({self._rules})"


@functools.lru_cache
def _create_router(config_str: str) -> GraphBackendRouter:
    """
    Create and cache GraphBackendRouter instances based on config string.
    """
    return GraphBackendRouter(config_str)


def get_backend_override_for_compile_id(
    compile_id: Optional[CompileId],
    config_str: str,
) -> Any:
    """
    Get the backend override for a given CompileId.

    Returns the backend function to use, or None if no override applies.
    """
    if compile_id is None or not config_str:
        return None

    graph_id = compile_id.frame_id
    if graph_id is None:
        return None

    router = _create_router(config_str)
    backend_str = router.get_backend_for_graph(graph_id)
    if backend_str:
        log.debug(
            "Graph %s (frame_id=%d) overridden to use backend: %s",
            compile_id,
            graph_id,
            backend_str,
        )
        return lookup_backend_with_mode(backend_str)
    return None
