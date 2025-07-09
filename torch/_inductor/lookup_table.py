import logging
from typing import Any, Optional

import torch
from torch._inductor.virtualized import V

from . import config as inductor_config


log = logging.getLogger(__name__)


def _in_use() -> bool:
    """
    Determine if the template lookup table should be used.
    This system is only used when cuda is available and the table has data.
    """
    return torch.cuda.is_available() and bool(
        inductor_config.template_lookup_table.table
    )


def _get_lookup_table() -> Optional[
    dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]
]:
    """
    Get the template lookup table from config.
    """
    if not _in_use():
        return None
    return inductor_config.template_lookup_table.table


def _dev_key(device: torch.device) -> str:
    """
    Generate a device key for lookup table indexing.
    For CPU devices, raises an error.
    For CUDA devices, returns a string combining device name and capability.
    """
    if device.type != "cuda":
        raise ValueError("CPU devices are not supported for lookup table")

    # Get CUDA device properties and capability
    props = torch.cuda.get_device_properties(device.index)
    capability = torch.cuda.get_device_capability(device.index)

    # Return string combining gcnArchName and capability
    return f"{props.gcnArchName}{capability}"


def _template_lookup_key(input_nodes: list[Any]) -> str:
    return str(
        tuple(
            # List here, because we want a standard encoding e.g. [] instead of ()
            (node.get_dtype(), list(get_size_hint(node)), list(get_stride_hint(node)))
            for node in input_nodes
        )
    )


def get_template_lookup_table(
    input_nodes: list[Any], method: str
) -> Optional[dict[str, dict[str, Any]]]:
    """
    Get the lookup table for a specific operation and input configuration.

    Args:
        input_nodes: List of input nodes for the operation
        method: Operation name (e.g., "mm", "addmm")

    Returns:
        Dictionary mapping template names to complete template_options, or None if not found
    """
    lookup_dict = None
    lookup_table = _get_lookup_table()
    if _in_use() and lookup_table is not None:
        # Assume the first input parameter is used to determine the device
        device = input_nodes[0].get_device()
        dev_key = _dev_key(device)
        log.debug("device_name: %s", dev_key)
        # Generate lookup table key
        lookup_key = _template_lookup_key(input_nodes)
        log.debug("lookup_key: %s", lookup_key)
        # Retrieve the lookup dictionary
        lookup_dict = lookup_table.get(dev_key, {}).get(method, {}).get(lookup_key, {})
    log.debug("lookup_dict for %s: %s", method, lookup_dict)
    return lookup_dict


def lookup_template_dict(
    lookup_dict: Optional[dict[str, dict[str, Any]]], template_key: str
) -> Optional[dict[str, Any]]:
    """
    Look up template configuration from the lookup dictionary.

    Args:
        lookup_dict: Dictionary containing template configurations
        key: Template key (e.g., "triton", "tma")

    Returns:
        Complete template_options dictionary, None if not found, or empty dict if no match
    """
    if not _in_use():
        return None

    # If lookup_dict is None or key not found, return None
    if lookup_dict is None or template_key not in lookup_dict:
        return None

    # Return the complete template_options dictionary directly
    return lookup_dict[template_key]


def lookup_table_extract_choice(choices: list[Any]) -> tuple[list[Any], bool]:
    """
    If there are multiple choices, this means we want the last choice as
    the lookup table produces at most one choice. The first choice is ATEN
    and the choice we fall back to when the lookup table has no match
    """
    if not _in_use():
        return choices, False
    if len(choices) > 1:
        return [choices[-1]], False

    from torch._inductor.select_algorithm import ExternKernelCaller

    # indicate to the caller that we're using the ATEN choice
    # as they might use this information to modify the layout
    return choices, isinstance(choices[0], ExternKernelCaller)


def get_size_hint(mat: Any) -> list[int]:
    size = mat.get_size()
    if not all(isinstance(dim, int) for dim in size):
        size = V.graph.sizevars.size_hints(
            size,
            fallback=inductor_config.unbacked_symint_fallback,
        )
    return list(size)


def get_stride_hint(mat: Any) -> list[int]:
    stride = mat.get_stride()
    if not all(isinstance(dim, int) for dim in stride):
        stride = V.graph.sizevars.size_hints(
            stride,
            fallback=inductor_config.unbacked_symint_fallback,
        )
    return list(stride)
