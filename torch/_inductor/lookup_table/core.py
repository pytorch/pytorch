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

Note: Empty dicts are used to indicate that the table is not in use or no matching
configs were found.

See torch/_inductor/kernel/mm.py (tuned_mm function) for a complete usage example.
"""

from __future__ import annotations

import copy
import logging
from functools import lru_cache
from typing import Any, Optional, TYPE_CHECKING

import torch

from .. import config as inductor_config


if TYPE_CHECKING:
    from ..kernel_inputs import KernelInputs


log = logging.getLogger(__name__)


def _get_lookup_table() -> dict[str, list[dict[str, Any]]]:
    """
    Get the template lookup table from config.
    """
    if (
        not torch.cuda.is_available()
        or inductor_config.template_config_lookup_table.table is None
    ):
        return {}
    return inductor_config.template_config_lookup_table.table


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


def lookup_key_suffix() -> str:
    """
    This is suffix we append to every lookup key to help us
    control for the Inductor environment as a whole
    """
    return f"tf32={bool(torch.backends.cuda.matmul.allow_tf32)}"


def make_lookup_key(kernel_inputs: KernelInputs, op_name: str) -> Optional[str]:
    """
    Create a flattened lookup key from kernel inputs and operation name.

    Args:
        kernel_inputs: KernelInputs object containing input nodes and scalars
        op_name: Operation name (e.g., "mm", "addmm")

    Returns:
        A string key combining device, operation, and input information
    """
    # Get device key
    device = kernel_inputs.device()
    dev_key = _dev_key(device)
    if dev_key is None:
        return None

    # Generate input key using the key property
    input_key = kernel_inputs.key

    # Create the flattened lookup key
    return f"{input_key}+{op_name}+{lookup_key_suffix()}"


def lookup_template_configs(
    kernel_inputs: KernelInputs,
    op_name: str,
    template_uids: list[str],
    template_hash_map: Optional[dict[str, Optional[str]]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Unified function to look up template configurations for multiple templates.

    Args:
        kernel_inputs: KernelInputs object containing input nodes and scalars
        op_name: Operation name (e.g., "mm", "addmm")
        template_uids: List of template identifiers (e.g., ["mm", "tma", "decompose_k"])
        template_hash_map: Optional mapping from template_uid to src_hash for validation

    Returns:
        {}: No lookup table in use, or no matches found for any template
        {"template_uid1": [config1, config2], ...}: Matches found, filtered configurations
    """
    lookup_table = _get_lookup_table()
    if not lookup_table:  # Empty dict means no table in use
        return {}

    # Create the flattened lookup key
    flat_key = make_lookup_key(kernel_inputs, op_name)
    if flat_key is None:
        # Could be a non-cuda tensor, even though cuda is available
        return {}
    # Get configs for this key
    config_list = lookup_table.get(flat_key, [])
    # Group configs by template_id
    configs_by_template: dict[str, list[dict[str, Any]]] = {}
    for config in config_list:
        if not isinstance(config, dict):
            raise ValueError(
                f"Config for {op_name} operation is not a dictionary: {config}"
            )
        if "template_id" not in config:
            raise ValueError(
                f"Config for {op_name} operation missing required 'template_id' field: {config}"
            )

        template_id = config["template_id"]
        if template_id in template_uids:
            if template_id not in configs_by_template:
                configs_by_template[template_id] = []
            configs_by_template[template_id].append(config)

    # Filter out TF32 configs, check template hashes, and clean up template_id field
    result = {}
    for template_id, matching_configs in configs_by_template.items():
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

            # Check template hash if enabled
            if (
                inductor_config.template_config_lookup_table.check_src_hash
                and template_hash_map
            ):
                template_hash = template_hash_map.get(template_id)
                config_hash = config.get("template_hash")

                if template_hash is not None and config_hash is not None:
                    if config_hash != template_hash:
                        log.warning(
                            "Filtering out config for template_id %s because template_hash %s does not match %s. Config: %s",
                            template_id,
                            config_hash,
                            template_hash,
                            config,
                        )
                        continue
                    else:
                        log.debug(
                            "Config for template_id %s has matching template_hash %s. Config: %s",
                            template_id,
                            template_hash,
                            config,
                        )
                elif config_hash is None:
                    log.debug(
                        "Config for template_id %s has no template_hash, keeping it. Template hash: %s. Config: %s",
                        template_id,
                        template_hash,
                        config,
                    )
                else:
                    log.debug(
                        "Template %s has no src_hash, keeping config with template_hash %s. Config: %s",
                        template_id,
                        config_hash,
                        config,
                    )

            # Return a copy of the config, as we don't want to modify the original
            cconfig = copy.deepcopy(config)
            # Lastly, we have to throw out the template_id, as it's not a valid kwarg
            # and just used to identify which template the entry belongs to
            del cconfig["template_id"]
            # Similarly, the template_hash is not a valid kwarg
            cconfig.pop("template_hash", None)
            filtered_configs.append(cconfig)

        if filtered_configs:
            result[template_id] = filtered_configs

    return result
