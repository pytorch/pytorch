import json
import logging
import re
from ast import literal_eval
from functools import lru_cache
from typing import Any, Optional

import torch
from torch._inductor import config as inductor_config
from torch._inductor.virtualized import V

from . import config as inductor_config
from .ir import ChoiceCaller


log = logging.getLogger(__name__)


def _in_use() -> bool:
    """
    Determine if the gemm config lookup table should be used.
    This system is only used when cuda is available
    Returns True if either:
    1. The kernel_config_lookup_table_path is set, OR
    2. The kernel_config_lookup_table global has been set
    """
    return torch.cuda.is_available() and (
        inductor_config.triton.kernel_config_lookup_table_path is not None
        or kernel_config_lookup_table is not None
    )


@lru_cache(0)
def _read_lookup_table(
    path: str,
) -> Optional[dict[str, dict[str, dict[str, dict[str, str]]]]]:
    """
    Handle actual file reading, so this can be cached depending on the input path
    """
    if not _in_use():
        return None
    # Assert supported file extensions
    if path.endswith(".json"):
        with open(path) as f:
            table = json.load(f)
    elif path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML files. "
                "Install with: pip install PyYAML, or convert your file to JSON format instead."
            ) from e
        with open(path) as f:
            table = yaml.safe_load(f)
    else:
        raise AssertionError(
            f"Unsupported file format. Only .json and .yaml/.yml files are supported. Got: {path}"
        )
    return table


def _get_lookup_table() -> Optional[dict[str, dict[str, dict[str, dict[str, str]]]]]:
    """
    Load and return the gemm config lookup table from file if configured.

    If the table is already defined, this takes precedence over the file path
    """
    if not _in_use():
        return None

    # If table is directly defined, use that
    if kernel_config_lookup_table is not None:
        # Log warning if both table and path are defined
        if inductor_config.triton.kernel_config_lookup_table_path is not None:
            log.warning(
                "Both kernel_config_lookup_table and kernel_config_lookup_table_path are defined. "
                "Ignoring the path because a table is already defined."
            )
        return kernel_config_lookup_table

    # Otherwise use the path if it's defined
    if inductor_config.triton.kernel_config_lookup_table_path is not None:
        return _read_lookup_table(
            inductor_config.triton.kernel_config_lookup_table_path
        )

    return None


# A static lookup table for (triton) configurations for GEMMs.
# This is a lookup table in the form of
# [device name][op][input_nodes_key][backend_key] = JSON_string
# where the JSON string format is operation-dependent:
#
# For MM family operations (mm, bmm, addmm, mm_plus_mm):

# for triton and tma backend_key:
# - "config": a list of GemmConfig parameters [BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps]
# - "kwargs": an optional dictionary of additional keyword arguments (defaults to empty dict if not present)
# Example: '{"config": [128, 128, 64, 2, 2], "kwargs": {"allow_tf32": "True"}}'
#
# for decompose_k backend_key:
# - "config": a list with a single value representing the k decomposition factor
# - "kwargs": an optional dictionary of additional keyword arguments (defaults to empty dict if not present)
# Example: '{"config": [4], "kwargs": {}}'
#
# Other operations may have different config structures as needed.
kernel_config_lookup_table: Optional[
    dict[str, dict[str, dict[str, dict[str, str]]]]
] = None


def _dev_key(device: torch.device) -> str:
    """
    Generate a device key for lookup table indexing.
    For CPU devices, returns "cpu".
    For CUDA devices, returns a string combining device name and capability.
    """
    if device.type != "cuda":
        return "cpu"

    # Get CUDA device properties and capability
    props = torch.cuda.get_device_properties(device.index)
    capability = torch.cuda.get_device_capability(device.index)

    # Return string combining gcnArchName and capability
    return f"{props.gcnArchName}{capability}"


def _gemm_lookup_key(input_nodes: list[Any]) -> str:
    return str(
        tuple(
            # List here, because we want a standard encoding e.g. [] instead of ()
            (node.get_dtype(), list(get_size_hint(node)), list(get_stride_hint(node)))
            for node in input_nodes
        )
    )


def get_gemm_lookup_table(
    input_nodes: list[Any], method: str
) -> Optional[dict[str, str]]:
    lookup_dict = None
    lookup_table = _get_lookup_table()
    if _in_use() and lookup_table is not None:
        # Assume the first input parameter is used to determine the device
        device = input_nodes[0].get_device()
        dev_key = _dev_key(device)
        log.debug("device_name: %s", dev_key)
        # Generate lookup table key
        lookup_key = _gemm_lookup_key(input_nodes)
        log.debug("lookup_key: %s", lookup_key)
        # Retrieve the lookup dictionary
        lookup_dict = lookup_table.get(dev_key, {}).get(method, {}).get(lookup_key, {})
    log.debug("lookup_dict for %s: %s", method, lookup_dict)
    return lookup_dict


def lookup_template_dict(
    lookup_dict: Optional[dict[str, str]], key: str
) -> Optional[dict[str, Any]]:
    if not _in_use():
        return None

    # Here, we return an empty config to indicate that
    # the lookup table exists but there is no match
    if lookup_dict is None or key not in lookup_dict:
        return {"config": [], "kwargs": {}}

    # Parse the JSON content from lookup_dict.get(key)
    try:
        parsed_data = json.loads(lookup_dict[key])

        # Ensure the parsed data is a dictionary
        if not isinstance(parsed_data, dict):
            raise ValueError(
                f"Invalid JSON structure for lookup table {lookup_dict[key]}, expected a dictionary"
            )

        # Check that 'config' key exists and is a list (required)
        if "config" not in parsed_data:
            raise ValueError(
                f"Invalid JSON structure for lookup table {lookup_dict[key]}, missing required 'config' key"
            )

        if not isinstance(parsed_data["config"], list):
            raise ValueError(
                f"Invalid JSON structure for lookup table {lookup_dict[key]}, 'config' must be a list"
            )

        # Check that 'kwargs' is a dictionary if present (optional)
        if "kwargs" in parsed_data and not isinstance(parsed_data["kwargs"], dict):
            raise ValueError(
                f"Invalid JSON structure for lookup table {lookup_dict[key]}, 'kwargs' must be a dictionary if present"
            )

        # If kwargs is not present, add an empty dict
        if "kwargs" not in parsed_data:
            parsed_data["kwargs"] = {}

        return parsed_data

    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(
            f"Invalid JSON structure for lookup table {lookup_dict[key]}, expected valid JSON"
        ) from e


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


def _extract_backend_key(caller: ChoiceCaller) -> Optional[str]:
    """
    Helper function to extract backend key from different caller types.
    Returns None if the caller should be skipped.
    """
    from torch._inductor.codegen.subgraph import SubgraphChoiceCaller
    from torch._inductor.select_algorithm import (
        ExternKernelCaller,
        TritonTemplateCaller,
    )

    if isinstance(caller, TritonTemplateCaller):
        # Check if "tma" is in the caller's name
        if "tma" in caller.name.lower():
            return "tma"
        else:
            return "triton"
    elif isinstance(caller, ExternKernelCaller):
        # Skip unless bias_addmm is in the name
        if "bias_addmm" in caller.name.lower():
            return "bias_addmm"
        else:
            return None  # Skip this caller
    elif isinstance(caller, SubgraphChoiceCaller):
        # Look for decompose_k_mm pattern
        if hasattr(caller, "name") and "decompose_k_mm" in caller.name:
            return "decompose_k"
        else:
            return None  # Skip this caller
    return None  # Skip unknown caller types


def _extract_config_dict(caller: ChoiceCaller) -> dict[str, Any]:
    """
    Helper function to extract config dictionary from different caller types.
    """
    from torch._inductor.codegen.subgraph import SubgraphChoiceCaller
    from torch._inductor.select_algorithm import (
        ExternKernelCaller,
        TritonTemplateCaller,
    )

    if isinstance(caller, TritonTemplateCaller):
        # Get config from info_dict similar to select_algorithm.py:1681-1708
        info = caller.info_dict()
        config_list = []
        kwargs_dict = {}

        # Extract tile shape - assume there's always a 3-member tuple
        tile_shape_str = info["tile_shape"]
        tile_vals = literal_eval(tile_shape_str)  # type: ignore[arg-type]
        # Assume there's always a 3-member tuple
        config_list.extend(
            # Note that we're using the order from the GemmConfig object
            # here which is [BLOCK_M, BLOCK_N, BLOCK_K] whereas the tile
            # is [BLOCK_M, BLOCK_K, BLOCK_N]
            [tile_vals[0], tile_vals[2], tile_vals[1]]
        )

        # Add num_stages and num_warps
        config_list.append(info["num_stages"])
        config_list.append(info["num_warps"])

        # Only care about allow_tf32 and make it uppercase
        if "allow_tf32" in info:
            # we know that allow_tf32 is a string, but might be "None" or "True"/"False"
            kwargs_dict["ALLOW_TF32"] = literal_eval(info["allow_tf32"])  # type: ignore[arg-type]

        return {"config": config_list, "kwargs": kwargs_dict}

    elif isinstance(caller, ExternKernelCaller):
        # For bias_addmm, return empty config
        return {"config": [], "kwargs": {}}

    elif isinstance(caller, SubgraphChoiceCaller):
        # Extract k_split from name pattern "decompose_k_mm_{k_split}_split"
        if hasattr(caller, "name"):
            match = re.search(r"decompose_k_mm_(\d+)_split", caller.name)
            if match:
                k_split = int(match.group(1))
                return {"config": [k_split], "kwargs": {}}

        return {"config": [], "kwargs": {}}

    return {"config": [], "kwargs": {}}


def _lookup_table_gemm_choice_entry(
    input_nodes: list[Any], method: str, caller: ChoiceCaller
) -> Optional[str]:
    """
    generate the log string for a gemm choice entry
    """
    # broken out to facilitate testing
    # Extract backend key - skip if None
    backend_key = _extract_backend_key(caller)
    if backend_key is None:
        # No logging to avoid polluting the log file, even in debug here
        return None

    # Get device key
    device = input_nodes[0].get_device()
    dev_key = _dev_key(device)

    # Get input key
    input_key = _gemm_lookup_key(input_nodes)

    # Extract config
    config_dict = _extract_config_dict(caller)

    # Create a valid lookup table entry that can be used with json.loads()
    lookup_entry = {
        dev_key: {
            method: {input_key: {backend_key: json.dumps(config_dict, sort_keys=True)}}
        }
    }

    return json.dumps(lookup_entry, sort_keys=True)


def log_gemm_choice(input_nodes: list[Any], method: str, caller: ChoiceCaller) -> None:
    """
    Log what a lookup table entry would look like for the given choice.
    The logged string is useable in json/python/yaml right away as a single entry.

    Format: A valid JSON string that can be used with json.loads()

    Args:
        input_nodes: List of input nodes
        method: Operation method name (e.g., "mm", "bmm", etc.)
        caller: ChoiceCaller instance (TritonTemplateCaller, ExternKernelCaller, or SubgraphChoiceCaller)
    """
    line = _lookup_table_gemm_choice_entry(input_nodes, method, caller)
    if line is not None:
        log.debug("Lookup table entry for gemm choice")
        # %r so that we get a properly escaped representation fo the dual encoding
        log.debug("%r", line)


def _reduction_lookup_key(
    size_hints: dict[str, int], inductor_meta: dict[str, Any]
) -> Optional[str]:
    """
    Generate a lookup key for reduction configs based on size_hints and inductor_meta.
    Similar to make_config_table_key but adapted for the lookup table structure.
    """
    size_hint = [f"{k}:{v}" for k, v in size_hints.items()]
    inductor_meta_key = []
    if "reduction_hint" not in inductor_meta:
        return None
    for k in ["grid_type", "num_load", "num_reduction", "reduction_hint"]:
        if k in inductor_meta:
            inductor_meta_key.append(f"{k}:{str(inductor_meta[k])}")

    return ",".join(size_hint + inductor_meta_key)


def get_reduction_lookup_table(
    size_hints: dict[str, int], inductor_meta: dict[str, Any], method: str
) -> Optional[dict[str, str]]:
    """
    Get reduction config lookup table for the given size_hints and method.

    Args:
        size_hints: Dictionary of size hints for the reduction operation
        inductor_meta: Metadata dictionary containing reduction information
        method: Method name, must be either 'reduction' or 'persistent_reduction'

    Returns:
        Dictionary containing lookup configurations or None if not available
    """
    assert method in [
        "reduction",
        "persistent_reduction",
    ], f"Method must be 'reduction' or 'persistent_reduction', got {method}"

    lookup_dict = None
    lookup_table = _get_lookup_table()
    if _in_use() and lookup_table is not None:
        # For reductions, we don't have input_nodes with devices, so we'll use the current device
        # or default to cuda device 0 if not available
        try:
            device_name = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            )
        except (RuntimeError, AssertionError):
            device_name = "cpu"

        # Generate lookup table key
        lookup_key = _reduction_lookup_key(size_hints, inductor_meta)
        if lookup_key is not None:
            log.debug(f"reduction lookup_key: {lookup_key}")
            # Retrieve the lookup dictionary
            lookup_dict = (
                lookup_table.get(method, {}).get(device_name, {}).get(lookup_key, {})
            )
    log.debug(f"lookup_dict for {method}: {lookup_dict}")
    return lookup_dict
