import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import torch._export
from torch._inductor.utils import is_cpu_device

from .runtime.runtime_utils import cache_dir


log = logging.getLogger(__name__)


@dataclass
class AOTICompileBackend:
    compile_fn: Callable[..., str]
    load_fn: Callable[[str, str, str], list[dict[str, Any] | None]]


_aoti_compile_backends: dict[str, AOTICompileBackend] = {}


def register_aoti_compile_backend(
    device_type: str,
    compile_fn: Callable[..., str],
    load_fn: Callable[[str, str, str], list[dict[str, Any] | None]],
) -> None:
    _aoti_compile_backends[device_type] = AOTICompileBackend(
        compile_fn=compile_fn,
        load_fn=load_fn,
    )


def aoti_eager_cache_dir(namespace: str, device: str) -> Path:
    return Path(cache_dir()) / "aoti_eager" / namespace / device


def aoti_eager_op_conf_lock(op_func_name_with_overload: str) -> Any:
    # Avoid circular import
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT
    from torch.utils._filelock import FileLock

    op_conf_lock_file = f"{op_func_name_with_overload}.lock"
    lock_dir = get_lock_dir()
    return FileLock(os.path.join(lock_dir, op_conf_lock_file), timeout=LOCK_TIMEOUT)


def load_aoti_eager_cache(
    ns: str, op_func_name_with_overload: str, device_type: str
) -> list[dict[str, Any] | None]:
    backend = _aoti_compile_backends.get(device_type)
    if backend:
        return backend.load_fn(ns, op_func_name_with_overload, device_type)

    device_kernel_cache = aoti_eager_cache_dir(ns, device_type)
    op_conf = device_kernel_cache / f"{op_func_name_with_overload}.json"
    if not op_conf.exists():
        return []

    try:
        with aoti_eager_op_conf_lock(op_func_name_with_overload):
            with open(op_conf) as f:
                json_data = json.load(f)
                for item in json_data:
                    # Get absolute path for kernel library
                    kernel_lib_abs_path = device_kernel_cache / item["kernel_path"]
                    item["kernel_path"] = kernel_lib_abs_path.as_posix()

                    # Check if the kernel library exists
                    if not kernel_lib_abs_path.exists():
                        return []

                    for metadata in item["meta_info"]:
                        if metadata.get("is_dynamic"):
                            raise NotImplementedError(
                                "Only support static shape for now"
                            )
                        if (
                            "device_type" in metadata
                            and metadata["device_type"] == "cpu"
                        ):
                            metadata["device_index"] = -1
                        for dtype_key in ["dtype", "dtype_value"]:
                            if dtype_key in metadata:
                                metadata[dtype_key] = getattr(
                                    torch, metadata[dtype_key].split(".")[-1]
                                )
                        if "layout_value" in metadata:
                            metadata["layout_value"] = getattr(
                                torch, metadata["layout_value"].split(".")[-1]
                            )
                        if "memory_format_value" in metadata:
                            metadata["memory_format_value"] = getattr(
                                torch, metadata["memory_format_value"].split(".")[-1]
                            )

                return json_data
    except Exception as e:
        err_msg = f"Failed to load aoti eager cache: {e}"
        log.exception(err_msg)
        return []


def supported_builtin_dtype_torch_dtype() -> dict[type, torch.dtype]:
    return {int: torch.int32, float: torch.float, bool: torch.bool}


def supported_scalar_types() -> tuple[type, ...]:
    type_to_torch_dtype = supported_builtin_dtype_torch_dtype()
    return tuple(type_to_torch_dtype.keys())


def extract_tensor_metadata(dynamic: bool, input: torch.Tensor) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    metadata["is_dynamic"] = dynamic

    if not isinstance(input, torch.Tensor):
        raise AssertionError(f"expected torch.Tensor, got {type(input)}")
    metadata["device_type"] = f"{input.device.type}"
    if is_cpu_device([input]):
        metadata["device_index"] = -1
    else:
        metadata["device_index"] = input.device.index
    metadata["dtype"] = f"{input.dtype}"
    metadata["sizes"] = list(input.size())
    metadata["strides"] = list(input.stride())
    metadata["requires_grad"] = input.requires_grad
    metadata["dispatch_key_set"] = torch._C._dispatch_keys(input).raw_repr()
    return metadata


def extract_tensor_list_metadata(
    dynamic: bool,
    input: list[torch.Tensor],
) -> dict[str, Any]:
    metadata_list = []
    for item in input:
        if not isinstance(item, torch.Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(item)}")
        metadata_list.append(extract_tensor_metadata(dynamic, item))

    metadata: dict[str, Any] = {}
    metadata["tensor_list"] = metadata_list
    return metadata


def extract_scalar_metadata(device_type: str, input: Any) -> dict[str, Any]:
    if not isinstance(input, supported_scalar_types()):
        raise AssertionError(f"expected a supported scalar type, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["is_dynamic"] = False
    # Scalar tensor
    metadata["device_type"] = device_type
    metadata["device_index"] = -1 if device_type == "cpu" else 0
    type_to_torch_dtype = supported_builtin_dtype_torch_dtype()
    metadata["dtype"] = f"{type_to_torch_dtype[type(input)]}"
    metadata["scalar_value"] = input
    return metadata


def extract_string_metadata(input: str) -> dict[str, Any]:
    if not isinstance(input, str):
        raise AssertionError(f"expected str, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["string_value"] = input
    return metadata


def extract_dtype_metadata(input: torch.dtype) -> dict[str, Any]:
    if not isinstance(input, torch.dtype):
        raise AssertionError(f"expected torch.dtype, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["dtype_value"] = f"{input}"
    return metadata


def extract_device_metadata(input: torch.device) -> dict[str, Any]:
    if not isinstance(input, torch.device):
        raise AssertionError(f"expected torch.device, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["device_type_value"] = f"{input.type}"
    metadata["device_index_value"] = input.index
    return metadata


def extract_layout_metadata(input: torch.layout) -> dict[str, Any]:
    if not isinstance(input, torch.layout):
        raise AssertionError(f"expected torch.layout, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["layout_value"] = f"{input}"
    return metadata


def extract_int_list_metadata(input: list[int]) -> dict[str, Any]:
    if not (
        isinstance(input, (list, tuple))
        and all(isinstance(item, int) and not isinstance(item, bool) for item in input)
    ):
        raise AssertionError(f"expected a list/tuple of int, got {input!r}")
    metadata: dict[str, Any] = {}
    metadata["int_list_value"] = list(input)
    return metadata


def aoti_compile_with_persistent_cache(
    ns: str,
    op_func_name_with_overload: str,
    device_type: str,
    dynamic: bool,
    f: Callable[..., Any],
    args: tuple[Any],
    kwargs: dict[str, Any],
    *,
    dynamic_shapes: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
) -> str:
    """
    Compile the given function with persistent cache for AOTI eager mode.
    """
    backend = _aoti_compile_backends.get(device_type)
    if backend:
        return backend.compile_fn(
            ns,
            op_func_name_with_overload,
            device_type,
            dynamic,
            f,
            args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            options=options,
            remove_runtime_assertions=remove_runtime_assertions,
            disable_constraint_solver=disable_constraint_solver,
        )

    if dynamic:
        raise AssertionError("Only support static shape for now")
    flattened_inputs = list(args) + list(kwargs.values())
    if not all(
        isinstance(
            input,
            (
                supported_scalar_types(),
                torch.Tensor,
                list,
                str,
                torch.dtype,
                torch.device,
                torch.layout,
            ),
        )
        for input in flattened_inputs
    ):
        err_msg = f"Unsupported input types: {flattened_inputs}"
        log.exception(err_msg)
        raise NotImplementedError(err_msg)

    for input in flattened_inputs:
        if isinstance(input, list) and not all(
            isinstance(item, torch.Tensor) for item in input
        ):
            err_msg = f"_impl_with_aoti_compile encounters unsupported input types: {flattened_inputs}"
            log.exception(err_msg)
            raise NotImplementedError(err_msg)

    persistent_cache = aoti_eager_cache_dir(ns, device_type)
    if not persistent_cache.exists():
        persistent_cache.mkdir(parents=True)

    persistent_cache_lib = persistent_cache / "lib"
    if not persistent_cache_lib.exists():
        persistent_cache_lib.mkdir()

    with mock.patch.dict(
        os.environ,
        {"TORCHINDUCTOR_CACHE_DIR": persistent_cache_lib.absolute().as_posix()},
    ):
        try:
            kernel_lib_path = torch._export.aot_compile(
                f,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                remove_runtime_assertions=remove_runtime_assertions,
                disable_constraint_solver=disable_constraint_solver,
                # Some operations may have non-Tensor parameters like int, float, bool. These
                # non-Tensor parameters will not be the input of the graph. Therefore, we do
                # need to keep the same signature.
                same_signature=False,
            )
            if not isinstance(kernel_lib_path, str):
                raise AssertionError(
                    f"expected kernel_lib_path to be str, got {type(kernel_lib_path)}"
                )

            kernel_metadata_items = []

            for idx, input in enumerate(flattened_inputs):
                # Skip explicit-None args (e.g. an optional `int[1]? dim`) so the
                # written metadata vector matches the cache reader, which also
                # drops None entries.
                if input is None:
                    continue
                if isinstance(input, torch.Tensor):
                    metadata = extract_tensor_metadata(dynamic, input)
                elif isinstance(input, (list, tuple)) and all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in input
                ):
                    # `int[]` / `SymInt[]` params. Checked before the Tensor-list
                    # branch so an empty list/tuple is tagged INT_LIST, matching
                    # the cache reader and C++ unpack_input_parameters for an
                    # empty `int[]`. An empty `Tensor[]` is the converse case:
                    # C++ tags it TENSOR_LIST (isTensorList() is checked before
                    # isIntList()) while this tags it INT_LIST -- only a cache
                    # miss/recompile (per-op cache, fixed schema type per slot ->
                    # no aliasing), and it does not arise for real ops
                    # (`cat([])`/`stack([])` raise in eager).
                    metadata = extract_int_list_metadata(list(input))
                elif isinstance(input, (list, tuple)) and all(
                    isinstance(item, torch.Tensor) for item in input
                ):
                    metadata = extract_tensor_list_metadata(dynamic, list(input))
                elif isinstance(input, supported_scalar_types()):
                    metadata = extract_scalar_metadata(device_type, input)
                elif isinstance(input, str):
                    metadata = extract_string_metadata(input)
                elif isinstance(input, torch.dtype):
                    metadata = extract_dtype_metadata(input)
                elif isinstance(input, torch.device):
                    metadata = extract_device_metadata(input)
                elif isinstance(input, torch.layout):
                    metadata = extract_layout_metadata(input)
                else:
                    raise NotImplementedError(f"Unsupported input type: {type(input)}")

                metadata["arg_order"] = idx
                kernel_metadata_items.append(metadata)

            kernel_meta_info: dict[str, Any] = {}
            kernel_meta_info["meta_info"] = kernel_metadata_items
            kernel_meta_info["kernel_path"] = (
                Path(kernel_lib_path).relative_to(persistent_cache).as_posix()
            )

            json_data = []
            update_json = True
            op_conf = persistent_cache / f"{op_func_name_with_overload}.json"
            mode = "r" if op_conf.exists() else "w"
            with aoti_eager_op_conf_lock(op_func_name_with_overload):
                with open(op_conf, mode) as op_conf_file:
                    try:
                        json_data = json.load(op_conf_file)
                    except Exception:
                        json_data = []

                    if not isinstance(json_data, list):
                        raise AssertionError(
                            f"expected json_data to be list, got {type(json_data)}"
                        )
                    for item in json_data:
                        if not isinstance(item, dict):
                            raise AssertionError(
                                f"expected item to be dict, got {type(item)}"
                            )
                        # Same kernel meta info already exists in the json file
                        if item["meta_info"] == kernel_metadata_items:
                            update_json = False
                            break

                if update_json:
                    json_data.append(kernel_meta_info)
                    with open(op_conf, "w") as op_conf_file:
                        json.dump(json_data, op_conf_file, indent=4)

            return kernel_lib_path
        except Exception as e:
            err_msg = f"Failed to compile {op_func_name_with_overload}: {e}"
            log.exception(err_msg)
            return ""
