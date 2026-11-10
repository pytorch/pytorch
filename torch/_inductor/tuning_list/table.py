from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import warning_once
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable

    from triton import Config


log = logging.getLogger(__name__)

# Positional order of block sizes in the config constructors.
AXIS_ORDER = ("x", "y", "z", "r0_")

# Everything else a supplied entry may carry. These reach the constructor as keywords, so
# the set is deliberately narrow: an unknown key is a caller mistake, not a stale config.
KNOB_ARGS: dict[str, frozenset[str]] = {
    "pointwise": frozenset(
        ["num_warps", "num_stages", "num_elements_per_warp", "waves_per_eu", "kpack"]
    ),
    "reduction": frozenset(["num_warps", "num_stages", "waves_per_eu"]),
    "persistent_reduction": frozenset(["num_warps", "num_stages", "waves_per_eu"]),
}

MODES = frozenset(["append", "replace"])


class InvalidTuningList(ValueError):
    """A supplied tuning_list table is malformed."""


def lookup(
    category: str, size_hints: dict[str, int], device: Any = None
) -> dict[str, Any] | None:
    """Find the entry for this category and arity, device-specific key first."""
    table = torch._inductor.config.tuning_list.table
    if not table or category not in KNOB_ARGS:
        return None

    suffix = f"{category}/{len(size_hints)}"
    # DeviceProperties.cc is the gfx arch name on ROCm and an int elsewhere.
    cc = getattr(device, "cc", None)
    keys = (f"{cc}+{suffix}", suffix) if isinstance(cc, str) and cc else (suffix,)
    for key in keys:
        if key in table:
            entry = table[key]
            if not isinstance(entry, dict) or "configs" not in entry:
                raise InvalidTuningList(
                    f"tuning_list entry {key!r} must be a dict carrying 'configs', got "
                    f"{type(entry).__name__}"
                )
            mode = entry.get("mode", "append")
            if mode not in MODES:
                raise InvalidTuningList(
                    f"tuning_list entry {key!r} has mode {mode!r}, expected one of "
                    f"{sorted(MODES)}"
                )
            return {"key": key, "mode": mode, "configs": entry["configs"]}
    return None


def _block_names(category: str, size_hints: dict[str, int]) -> list[str]:
    """Block sizes an entry must name, in constructor order."""
    axes = [a for a in AXIS_ORDER if a in size_hints]
    if category == "persistent_reduction":
        axes = [a for a in axes if a != "r0_"]
    return [f"{a.upper()}BLOCK" for a in axes]


def _knob_names(category: str, size_hints: dict[str, int]) -> frozenset[str]:
    knobs = KNOB_ARGS[category]
    if category != "pointwise" and "y" in size_hints:
        # triton_config_tiled_reduction takes no num_warps
        knobs -= frozenset(["num_warps"])
    return knobs


def _split(
    category: str, names: list[str], knobs: frozenset[str], supplied: Any, key: str
) -> tuple[list[int], dict[str, Any]]:
    """Split an entry into positional block sizes and keyword knobs."""
    if not isinstance(supplied, dict):
        raise InvalidTuningList(
            f"tuning_list entry {key!r} holds {type(supplied).__name__}, expected a dict "
            "of block sizes and knobs"
        )

    unknown = OrderedSet(supplied) - OrderedSet(names) - knobs
    if unknown:
        reason = ""
        if category == "persistent_reduction" and "R0_BLOCK" in unknown:
            reason = "; the compiler spans the whole reduction in one block"
        elif "num_warps" in unknown:
            reason = "; a tiled reduction derives it from the tile"
        raise InvalidTuningList(
            f"tuning_list entry {key!r} carries {sorted(unknown)}, which {category} does "
            f"not accept{reason}. Block sizes are {names}; knobs are {sorted(knobs)}"
        )

    blocks = []
    for name in names:
        if name not in supplied:
            raise InvalidTuningList(
                f"tuning_list entry {key!r} does not name {name}; kernels under this key "
                f"take {names}"
            )
        value = supplied[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InvalidTuningList(
                f"tuning_list entry {key!r} has {name}={value!r}, expected a positive int"
            )
        blocks.append(value)

    return blocks, {k: v for k, v in supplied.items() if k not in names}


def apply(
    category: str,
    size_hints: dict[str, int],
    configs: list[Config],
    make_config: Callable[..., Config],
    entry: dict[str, Any],
) -> list[Config]:
    """Fold a looked-up entry into the candidates the heuristic generated."""
    names = _block_names(category, size_hints)
    accepted = _knob_names(category, size_hints)
    supplied = []
    for one in entry["configs"]:
        blocks, knobs = _split(category, names, accepted, one, entry["key"])
        try:
            supplied.append(make_config(size_hints, *blocks, **knobs))
        except Exception as exc:
            # A list fitted on one corpus meets kernels it was not fitted on, so a config
            # that will not build for this shape is expected rather than exceptional.
            warning_once(
                log,
                "tuning_list: dropping %s from %r: %s",
                repr(one),
                entry["key"],
                str(exc),
            )

    if not supplied:
        return configs
    return supplied if entry["mode"] == "replace" else [*configs, *supplied]
