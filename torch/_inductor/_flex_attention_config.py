# mypy: allow-untyped-defs
from __future__ import annotations

from typing import Any

from torch.utils._ordered_set import OrderedSet


FLEX_ATTENTION_MARKERS = OrderedSet(["SPARSE_Q_BLOCK_SIZE", "SPARSE_KV_BLOCK_SIZE"])
OPTIONS_EXAMPLES = {
    "backward": (
        "kernel_options={'bwd_BLOCK_M1': 32, 'bwd_BLOCK_N1': 32, "
        "'bwd_BLOCK_M2': 32, 'bwd_BLOCK_N2': 32, "
        "'bwd_num_stages': 1, 'bwd_num_warps': 4}"
    ),
    "forward": (
        "kernel_options={'fwd_BLOCK_M': 32, 'fwd_BLOCK_N': 64, "
        "'fwd_num_stages': 1, 'fwd_num_warps': 4}"
    ),
}
TUNING_OPTIONS = {
    "backward": (
        "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_warps, and "
        "num_stages; use the bwd_ prefix to set backward-only options"
    ),
    "decode": (
        "BLOCK_M, BLOCK_N, num_warps, and num_stages; use the fwd_ "
        "prefix to set decode-only options"
    ),
    "forward": (
        "BLOCK_M, BLOCK_N, num_warps, and num_stages; use the fwd_ "
        "prefix to set forward-only options"
    ),
}


def flex_attention_kind(config_args: dict[str, Any]) -> str | None:
    if not FLEX_ATTENTION_MARKERS <= config_args.keys():
        return None
    if "BLOCK_M1" in config_args:
        return "backward"
    if "SPLIT_KV" in config_args:
        return "decode"
    if "BLOCK_M" in config_args and "BLOCK_N" in config_args:
        return "forward"
    return None


def flex_kernel_options_example(kind: str) -> str:
    return OPTIONS_EXAMPLES["backward" if kind == "backward" else "forward"]


def flex_kernel_tuning_options(kind: str) -> str:
    return TUNING_OPTIONS[kind if kind in ("backward", "decode") else "forward"]


def flex_kernel_selected_options(
    kind: str,
    config_args: dict[str, Any],
    num_stages: int | None = None,
    num_warps: int | None = None,
) -> str:
    option_names = (
        ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")
        if kind == "backward"
        else ("BLOCK_M", "BLOCK_N")
    )
    options = [
        f"{name}={config_args[name]}" for name in option_names if name in config_args
    ]
    if num_stages is not None:
        options.append(f"num_stages={num_stages}")
    if num_warps is not None:
        options.append(f"num_warps={num_warps}")
    return ", ".join(options)
