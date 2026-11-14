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

# The axes a supplied config can name a block size for.
BLOCK_AXES = ("x", "y", "z", "r0_")

# The config constructors' argument for each block size.
BLOCK_ARGS = {"XBLOCK": "x", "YBLOCK": "y", "ZBLOCK": "z", "R0_BLOCK": "r"}

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
    """A supplied tuning_list entry is malformed."""


def table_entry(
    heuristic: str,
    size_hints: dict[str, int],
    triton_meta: TritonMeta,
    inductor_meta: dict[str, Any],
) -> Any:
    """This kernel's entry in config.tuning_list.table, device-specific key first."""
    table = torch._inductor.config.tuning_list.table
    if not table:
        return None
    for key in table:
        category, _, num_axes = str(key).rpartition("+")[2].partition("/")
        if category not in KNOB_ARGS or not num_axes.isdigit():
            raise InvalidTuningList(
                f"tuning_list key {key!r} is not '[<device>+]<category>/<num_axes>' "
                f"with a category in {sorted(KNOB_ARGS)}"
            )

    if heuristic == "cooperative_reduction":
        # Looked up as the reduction it splits, so that check() logs the skip.
        persistent = inductor_meta.get("persistent_reduction")
        heuristic = "persistent_reduction" if persistent else "reduction"
    suffix = f"{heuristic}/{len(size_hints)}"
    # DeviceProperties.cc is the gfx arch name on ROCm and the compute capability as an
    # int on CUDA, such as 90; XPU's is a dict.
    cc = getattr(triton_meta.get("device"), "cc", None)
    keys = (suffix,)
    if isinstance(cc, (str, int)) and cc:
        keys = (f"{cc}+{suffix}", suffix)
    for key in keys:
        if key in table:
            return table[key]
    return None


def check(
    entry: Any,
    heuristic: str,
    size_hints: dict[str, int],
    triton_meta: TritonMeta,
    inductor_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate the entry supplied for this kernel and return it with plain int values,
    or None if supplied configs do not apply to it."""
    if entry is None:
        return None
    if heuristic not in KNOB_ARGS:
        skipped = f"only {', '.join(KNOB_ARGS)} kernels are covered"
    elif triton_meta.get("native_matmul"):
        skipped = "a native matmul builds its configs its own way"
    elif inductor_meta.get("strict_reduction_rblock") is not None:
        skipped = "its R0_BLOCK is planned, and every candidate must match it"
    elif "r1_" in size_hints:
        skipped = "it is tiled over two reduction axes"
    else:
        skipped = ""
    if skipped:
        log.debug(
            "tuning_list: entry not applied to %s kernel %s: %s",
            heuristic,
            size_hints,
            skipped,
        )
        return None

    label = f"{heuristic}/{len(size_hints)}"
    if not isinstance(entry, dict) or "configs" not in entry:
        raise InvalidTuningList(
            f"tuning_list entry for {label} must be a dict carrying 'configs', got "
            f"{type(entry).__name__}"
        )
    mode = entry.get("mode", "append")
    if mode not in MODES:
        raise InvalidTuningList(
            f"tuning_list entry for {label} has mode {mode!r}, expected one of "
            f"{sorted(MODES)}"
        )
    names = _block_names(heuristic, size_hints)
    knobs = _knob_names(heuristic, size_hints)
    configs = [
        _validate_config(one, heuristic, names, knobs, label)
        for one in entry["configs"]
    ]
    return {"mode": mode, "configs": configs}


def supplied_configs(
    inductor_meta: dict[str, Any], build: Callable[..., Config]
) -> tuple[list[Config], bool]:
    """Build the configs codegen carried for this kernel, and say whether they replace
    the heuristic's candidates. build is the heuristic's own constructor; it gets the
    block sizes as x, y, z and r, and knobs by name."""
    entry = inductor_meta.get("tuning_list")
    if entry is None:
        return [], False
    supplied = []
    for one in entry["configs"]:
        kwargs = {BLOCK_ARGS.get(name, name): value for name, value in one.items()}
        try:
            supplied.append(build(**kwargs))
        except Exception as exc:
            # A list fitted on one corpus meets kernels it was not fitted on, so a config
            # that will not build for this shape is expected rather than exceptional.
            warning_once(log, "tuning_list: dropping %s: %s", repr(one), str(exc))
    return supplied, entry["mode"] == "replace"


def _block_names(category: str, size_hints: dict[str, int]) -> list[str]:
    """Block sizes an entry must name."""
    axes = [a for a in BLOCK_AXES if a in size_hints]
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
    config: Any, category: str, names: list[str], knobs: frozenset[str], label: str
) -> dict[str, int]:
    """Check one supplied config and return it with plain int values."""
    if not isinstance(config, dict):
        raise InvalidTuningList(
            f"tuning_list entry for {label} holds {type(config).__name__}, expected a "
            "dict of block sizes and knobs"
        )

    unknown = OrderedSet(config) - OrderedSet(names) - knobs
    if unknown:
        reason = ""
        if category == "persistent_reduction" and "R0_BLOCK" in unknown:
            reason = "; the compiler spans the whole reduction in one block"
        elif "num_warps" in unknown:
            reason = "; a tiled reduction derives it from the tile"
        raise InvalidTuningList(
            f"tuning_list entry for {label} carries {sorted(unknown)}, which {category} "
            f"does not accept{reason}. Block sizes are {names}; knobs are {sorted(knobs)}"
        )

    for name in names:
        if name not in config:
            raise InvalidTuningList(
                f"tuning_list entry for {label} does not name {name}; these kernels take "
                f"{names}"
            )

    # Stored as plain ints: the kernel source carries the config's repr, and a NumPy
    # int's repr (np.int64(8)) does not evaluate there.
    checked: dict[str, int] = {}
    for name, value in config.items():
        problem = f"tuning_list entry for {label} has {name}={value!r}, expected"
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
