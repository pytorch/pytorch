from __future__ import annotations

import logging
import operator
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import warning_once
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable

    from triton import Config

    from torch._inductor.runtime.hints import TritonMeta


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
    category: str,
    size_hints: dict[str, int],
    triton_meta: TritonMeta,
    inductor_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Find and validate this kernel's entry, device-specific key first."""
    table = torch._inductor.config.tuning_list.table
    if not table:
        return None
    for key in table:
        named, _, arity = str(key).rpartition("+")[2].partition("/")
        if named not in KNOB_ARGS or not arity.isdigit():
            raise InvalidTuningList(
                f"tuning_list key {key!r} is not '[<device>+]<category>/<arity>' with a "
                f"category in {sorted(KNOB_ARGS)}"
            )
    if (
        category not in KNOB_ARGS
        # A native matmul builds its configs its own way, a planned reduction block must
        # match every candidate, and r1_ kernels share their key with kernels tiled on y.
        or triton_meta.get("native_matmul")
        or inductor_meta.get("strict_reduction_rblock") is not None
        or "r1_" in size_hints
    ):
        return None

    suffix = f"{category}/{len(size_hints)}"
    # DeviceProperties.cc is the gfx arch name on ROCm and an int elsewhere.
    cc = getattr(triton_meta.get("device"), "cc", None)
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
            names = _block_names(category, size_hints)
            knobs = _knob_names(category, size_hints)
            configs = [
                _validate_config(category, names, knobs, one, key)
                for one in entry["configs"]
            ]
            return {"key": key, "mode": mode, "configs": configs}
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


def _validate_config(
    category: str, names: list[str], knobs: frozenset[str], supplied: Any, key: str
) -> dict[str, int]:
    """Check one supplied config and return it with plain int values."""
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

    for name in names:
        if name not in supplied:
            raise InvalidTuningList(
                f"tuning_list entry {key!r} does not name {name}; kernels under this key "
                f"take {names}"
            )

    # Stored as plain ints: the kernel source carries the config's repr, and a NumPy
    # int's repr (np.int64(8)) does not evaluate there.
    checked: dict[str, int] = {}
    for name, value in supplied.items():
        problem = f"tuning_list entry {key!r} has {name}={value!r}, expected"
        try:
            number = None if isinstance(value, bool) else operator.index(value)
        except TypeError:
            number = None
        if number is None:
            raise InvalidTuningList(f"{problem} an int")
        if name in names or name == "num_warps":
            # Triton fails the whole kernel on any other block size or warp count.
            if number <= 0 or number & (number - 1):
                raise InvalidTuningList(f"{problem} a power of 2")
        elif number < 0:
            raise InvalidTuningList(f"{problem} a non-negative int")
        checked[name] = number
    return checked


def apply(
    category: str,
    size_hints: dict[str, int],
    configs: list[Config],
    make_config: Callable[..., Config],
    entry: dict[str, Any],
) -> list[Config]:
    """Fold a looked-up entry into the candidates the heuristic generated."""
    names = _block_names(category, size_hints)
    supplied = []
    for one in entry["configs"]:
        blocks = [one[name] for name in names]
        knobs = {k: v for k, v in one.items() if k not in names}
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
