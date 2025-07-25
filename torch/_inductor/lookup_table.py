"""
Template Lookup Table System

This module provides a lookup table system for template configurations used in autotuning.
The typical usage pattern is:

1. Get the template configs directly with a single call:
   template_configs = lookup_template_configs(input_nodes, op_name, template_id)

2. Use those configs to append choices directly:
   for kwargs in template_configs:
       template.maybe_append_choice(choices, input_nodes=input_nodes, **kwargs)

3. Extract final choices (here, we pass add_aten as a function to add aten if choices is empty):
   choices = lookup_table_extract_choices(choices, add_aten)

Note: None is used to indicate that the table is not in use, whereas empty lists and
dicts are fine to use (they indicate no matching configs were found).

See torch/_inductor/kernel/mm.py (tuned_mm function) for a complete usage example.
"""

import copy
import logging
from functools import lru_cache
from typing import Any, Callable, Optional

import torch
from torch._inductor.virtualized import V

from . import config as inductor_config


log = logging.getLogger(__name__)


def _in_use() -> bool:
    """
    Determine if the template lookup table should be used.
    This system is only used when cuda is available and the table has data.
    """
    active = (
        torch.cuda.is_available() and inductor_config.template_lookup_table is not None
    )
    # the lookup table requires max-autotune and it is an error to use it without max-autotune
    if active and not (
        inductor_config.max_autotune or inductor_config.max_autotune_gemm
    ):
        raise RuntimeError(
            "The template lookup table requires max-autotune to be enabled. "
            "Please set inductor_config.max_autotune=True, or remove the lookup table. "
        )
    return active


def _get_lookup_table() -> Optional[dict[str, list[dict[str, Any]]]]:
    """
    Get the template lookup table from config.
    """
    if not _in_use():
        return None
    return inductor_config.template_lookup_table


@lru_cache
def _dev_key(device: torch.device) -> Optional[str]:
    """
    Generate a device key for lookup table indexing.
    For CPU devices, raises an error.
    For CUDA devices, returns the props.gcnArchName string.
    """
    if device.type != "cuda":
        # only cuda devices are supported, this indicates that the system is not in use
        # for this device
        return None

    # Get CUDA device properties
    props = torch.cuda.get_device_properties(device.index)

    return props.gcnArchName


def get_size_hint(mat: Any) -> list[int]:
    """Get size hints for a tensor, handling symbolic sizes."""
    size = mat.get_size()
    if not all(isinstance(dim, int) for dim in size):
        size = V.graph.sizevars.size_hints(
            size,
            fallback=inductor_config.unbacked_symint_fallback,
        )
    return list(size)


def get_stride_hint(mat: Any) -> list[int]:
    """Get stride hints for a tensor, handling symbolic strides."""
    stride = mat.get_stride()
    if not all(isinstance(dim, int) for dim in stride):
        stride = V.graph.sizevars.size_hints(
            stride,
            fallback=inductor_config.unbacked_symint_fallback,
        )
    return list(stride)


def _inputs_lookup_key(input_nodes: list[Any]) -> str:
    """
    Generate a lookup key based on input node properties.
    The key includes dtype, size, and stride information for each input node.
    """
    return str(
        tuple(
            # List here, because we want a standard encoding e.g. [] instead of ()
            (node.get_dtype(), list(get_size_hint(node)), list(get_stride_hint(node)))
            for node in input_nodes
        )
    )


def lookup_key_suffix() -> str:
    """
    This is suffix we append to every lookup key to help us
    control for the Inductor environment as a whole
    """
    return f"tf32={bool(torch.backends.cuda.matmul.allow_tf32)}"


def make_lookup_key(input_nodes: list[Any], op_name: str) -> Optional[str]:
    """
    Create a flattened lookup key from input nodes and operation name.

    If template_hash is provided, we will look for 'template_hash' in the config
    and compare it
    If there is no match, or no template_hash in the config, we will filter
    out the config

    Args:
        input_nodes: List of input nodes for the operation
        op_name: Operation name (e.g., "mm", "addmm")

    Returns:
        A string key combining device, operation, and input information
    """
    # Get device key
    device = input_nodes[0].get_device()
    dev_key = _dev_key(device)
    if dev_key is None:
        return None

    # Generate input key
    input_key = _inputs_lookup_key(input_nodes)

    # Create the flattened lookup key
    return f"{dev_key}+{op_name}+{input_key}+{lookup_key_suffix()}"


def lookup_template_configs(
    input_nodes: list[Any],
    op_name: str,
    template_id: str,
    template_hash: Optional[str] = None,
) -> Optional[list[dict[str, Any]]]:
    """
    Unified function to look up template configurations for a specific operation and template.

    Args:
        input_nodes: List of input nodes for the operation
        op_name: Operation name (e.g., "mm", "addmm")
        template_id: Template identifier (e.g., "mm", "tma", "decompose_k")
        template_hash: Unique hash for the template, or None

    Returns:
        None: No lookup table is in use
        []: Lookup table exists but no match found or all configs filtered out
        [kwargs1, kwargs2, ...]: Match found, filtered configurations
    """
    lookup_table = _get_lookup_table()
    if lookup_table is None:
        return None

    # Create the flattened lookup key
    flat_key = make_lookup_key(input_nodes, op_name)
    if flat_key is None:
        # Could be a non-cuda tensor, even though cuda is available
        return None

    # Get configs for this key
    config_list = lookup_table.get(flat_key, [])

    # Filter configs by template_id
    matching_configs = []
    for config in config_list:
        if not isinstance(config, dict):
            raise ValueError(
                f"Config for {op_name} operation is not a dictionary: {config}"
            )
        if "template_id" not in config:
            raise ValueError(
                f"Config for {op_name} operation missing required 'template_id' field: {config}"
            )
        if config["template_id"] == template_id:
            matching_configs.append(config)

    log.debug("configs for %s (%s): %r", template_id, template_hash, config_list)
    # Filter out configs with ALLOW_TF32=True when torch.backends.cuda.matmul.allow_tf32 is False
    filtered_configs = []
    for config in matching_configs:
        if (
            "ALLOW_TF32" in config
            and config["ALLOW_TF32"] is True
            and not torch.backends.cuda.matmul.allow_tf32
        ):
            log.warning(
                "Filtering out config with ALLOW_TF32=True because "
                "torch.backends.cuda.matmul.allow_tf32 is False. Config: %s",
                config,
            )
            continue
        if template_hash is not None:
            chash = config.get("template_hash")
            if chash is None:
                log.debug(
                    "Config %r for template_id %s has no template_hash, so we will "
                    "keep it and not compare it against provided hash %s",
                    config,
                    template_id,
                    template_hash,
                )
            elif chash != template_hash:
                log.warning(
                    "Filtering out config %r for template_id %s because template_hash %s does not match %s",
                    config,
                    template_id,
                    chash,
                    template_hash,
                )
                continue
        # Return a copy of the config, as we don't want to modify the original
        cconfig = copy.deepcopy(config)
        # Lastly, we have to throw out the template_id, as it's not a valid kwarg
        # and just used to identify which template the entry belongs to
        del cconfig["template_id"]
        # Similarly, the template_hash is not a valid kwarg
        cconfig.pop("template_hash", None)
        filtered_configs.append(cconfig)

    return filtered_configs


def lookup_table_extract_choices(
    choices: list[Any], fallback_fn: Callable[[], list[Any]]
) -> list[Any]:
    """
    If there are multiple choices, this means that the lookup table was used.
    The initial choice is always ATEN, so we want to skip it and return the rest,
    if there are other choices
    """
    if not _in_use():
        return choices
    if len(choices) > 0:
        return choices
    return fallback_fn()
