import json
import logging
import operator
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest import mock

import sympy

import torch
import torch._dynamo.config
import torch._export
import torch.export._trace
from torch._dynamo.source import ConstantSource
from torch._inductor.utils import is_cpu_device

from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.sym_node import SymNode, sympy_is_contiguous_generic
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from .runtime.runtime_utils import cache_dir


log = logging.getLogger(__name__)


def aoti_eager_cache_dir(namespace: str, device: str) -> Path:
    return Path(cache_dir()) / "aoti_eager" / namespace / device


def aoti_eager_op_conf_lock(op_func_name_with_overload: str) -> Any:
    from filelock import FileLock

    # Avoid circular import
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT

    op_conf_lock_file = f"{op_func_name_with_overload}.lock"
    lock_dir = get_lock_dir()
    return FileLock(os.path.join(lock_dir, op_conf_lock_file), timeout=LOCK_TIMEOUT)


def create_symtype(
    cls: Any, pytype: type, shape_env: ShapeEnv, val: Any, duck: bool = True
) -> Any:
    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}"),
        dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        constraint_dim=None,
    )
    return cls(
        SymNode(
            symbol,
            shape_env,
            pytype,
            hint=val,
        )
    )


def create_symint(shape_env: ShapeEnv, i: int, duck: bool = True) -> Any:
    return create_symtype(torch.SymInt, int, shape_env, i, duck=duck)


def load_aoti_eager_cache(
    ns: str, op_func_name_with_overload: str, device_type: str
) -> List[Optional[Dict[str, Any]]]:
    device_kernel_cache = aoti_eager_cache_dir(ns, device_type)
    op_conf = device_kernel_cache / f"{op_func_name_with_overload}.json"
    if not op_conf.exists():
        return []

    try:
        with aoti_eager_op_conf_lock(op_func_name_with_overload):
            with open(op_conf) as f:
                json_data = json.load(f)
                for item in json_data:
                    # Get absolution path for kernel library
                    kernel_lib_abs_path = device_kernel_cache / item["kernel_path"]
                    item["kernel_path"] = kernel_lib_abs_path.as_posix()

                    # Check if the kernel library exists
                    if not kernel_lib_abs_path.exists():
                        return []

                    # Create shape environment per kernel
                    shape_env = ShapeEnv()

                    for metadata in item["meta_info"]:
                        if metadata.get("is_dynamic"):
                            assert isinstance(metadata["sizes_hint"], Dict)
                            assert isinstance(metadata["strides_hint"], Dict)
                            sizes_hint = list(metadata["sizes_hint"].values())
                            strides_hint = list(metadata["strides_hint"].values())
                            sympy_sizes_expr = [
                                sympy.simplify(str_sym_size)
                                for str_sym_size in metadata["sizes"].values()
                            ]
                            sympy_strides_expr = [
                                sympy.simplify(str_sym_stride)
                                for str_sym_stride in metadata["strides"].values()
                            ]
                            metadata["sizes"] = [
                                shape_env.create_symintnode(
                                    sympy_sizes_expr[idx], hint=sizes_hint[idx]
                                )
                                for idx in range(len(sizes_hint))
                            ]
                            metadata["strides"] = [
                                shape_env.create_symintnode(
                                    sympy_strides_expr[idx], hint=strides_hint[idx]
                                )
                                for idx in range(len(sizes_hint))
                            ]

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


def supported_builtin_dtype_torch_dtype() -> Dict[type, torch.dtype]:
    return {int: torch.int32, float: torch.float, bool: torch.bool}


def supported_scalar_types() -> Tuple[type, ...]:
    type_to_torch_dtype = supported_builtin_dtype_torch_dtype()
    return tuple(type_to_torch_dtype.keys())


def extract_tensor_metadata(
    dynamic: bool, input: torch.Tensor, fake_input: Union[FakeTensor, None]
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    metadata["is_dynamic"] = dynamic

    assert isinstance(input, torch.Tensor)
    metadata["device_type"] = f"{input.device.type}"
    if is_cpu_device([input]):
        metadata["device_index"] = -1
    else:
        metadata["device_index"] = input.device.index
    metadata["dtype"] = f"{input.dtype}"

    if dynamic:
        assert fake_input is not None
        # If dynamic is specified, we expect all the size and strides are symbolic
        sym_size = fake_input.size()
        sym_strides = fake_input.stride()
        metadata["sizes"] = {}
        metadata["sizes_hint"] = {}
        for idx, sym_item in enumerate(sym_size):
            metadata["sizes"][idx] = str(sym_item)
            metadata["sizes_hint"][idx] = input.size(idx)

        metadata["strides"] = {}
        metadata["strides_hint"] = {}
        for idx, sym_item in enumerate(sym_strides):
            metadata["strides"][idx] = str(sym_item)
            metadata["strides_hint"][idx] = input.stride(idx)

        sorted_strides_hint = dict(
            sorted(metadata["strides_hint"].items(), key=operator.itemgetter(1))
        )
        tensor_sizes = list(input.size())
        tensor_strides = list(input.stride())
        tensor_layout = list(sorted_strides_hint.keys())
        assert sympy_is_contiguous_generic(tensor_sizes, tensor_strides, tensor_layout)
        metadata["dim_order"] = tensor_layout
    else:
        metadata["sizes"] = list(input.size())
        metadata["strides"] = list(input.stride())
    metadata["requires_grad"] = input.requires_grad
    metadata["dispatch_key_set"] = torch._C._dispatch_keys(input).raw_repr()
    return metadata


def extract_tensor_list_metadata(
    dynamic: bool, input: List[torch.Tensor], fake_inputs: List[Union[FakeTensor, None]]
) -> Dict[str, Any]:
    metadata_list = []
    for idx, item in enumerate(input):
        assert isinstance(item, torch.Tensor)
        metadata_list.append(extract_tensor_metadata(dynamic, item, fake_inputs[idx]))

    metadata: Dict[str, Any] = {}
    metadata["tensor_list"] = metadata_list
    return metadata


def extract_scalar_metadata(device_type: str, input: Any) -> Dict[str, Any]:
    assert isinstance(input, supported_scalar_types())
    metadata: Dict[str, Any] = {}
    metadata["is_dynamic"] = False
    # Scalar tensor
    metadata["device_type"] = device_type
    metadata["device_index"] = -1 if device_type == "cpu" else 0
    type_to_torch_dtype = supported_builtin_dtype_torch_dtype()
    metadata["dtype"] = f"{type_to_torch_dtype[type(input)]}"
    metadata["scalar_value"] = input
    return metadata


def extract_string_metadata(input: str) -> Dict[str, Any]:
    assert isinstance(input, str)
    metadata: Dict[str, Any] = {}
    metadata["string_value"] = input
    return metadata


def extract_dtype_metadata(input: torch.dtype) -> Dict[str, Any]:
    assert isinstance(input, torch.dtype)
    metadata: Dict[str, Any] = {}
    metadata["dtype_value"] = f"{input}"
    return metadata


def extract_device_metadata(input: torch.device) -> Dict[str, Any]:
    assert isinstance(input, torch.device)
    metadata: Dict[str, Any] = {}
    metadata["device_type_value"] = f"{input.type}"
    metadata["device_index_value"] = input.index
    return metadata


def extract_layout_metadata(input: torch.layout) -> Dict[str, Any]:
    assert isinstance(input, torch.layout)
    metadata: Dict[str, Any] = {}
    metadata["layout_value"] = f"{input}"
    return metadata


def mark_tensor_dim_as_dynamic(inputs: Any) -> None:
    def _mark_tensor_dim_as_dynamic(input_item: Any) -> None:
        torch._dynamo.mark_dynamic(input_item, list(range(input_item.ndim)))

    for input_item in inputs:
        if isinstance(input_item, torch.Tensor):
            _mark_tensor_dim_as_dynamic(input_item)
        elif isinstance(input_item, list):
            for item in input_item:
                if isinstance(item, torch.Tensor):
                    _mark_tensor_dim_as_dynamic(input_item)


def aoti_compile_with_persistent_cache(
    ns: str,
    op_func_name_with_overload: str,
    device_type: str,
    dynamic: bool,
    f: Callable[..., Any],
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    *,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    disable_constraint_solver: bool = False,
) -> str:
    """
    Compile the given function with persistent cache for AOTI eager mode.
    """
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

    dynamic = dynamic and not torch._dynamo.config.assume_static_by_default

    with mock.patch.dict(
        os.environ,
        {"TORCHINDUCTOR_CACHE_DIR": persistent_cache_lib.absolute().as_posix()},
    ), torch._dynamo.config.patch(
        automatic_dynamic_shapes=dynamic,
        dynamic_shapes=dynamic,
    ):
        try:
            if dynamic:
                mark_tensor_dim_as_dynamic(flattened_inputs)

            gm = torch.export._trace._export_to_torch_ir(
                f,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                disable_constraint_solver=disable_constraint_solver,
                # Disabling this flag, because instead we can rely on the mapping
                # dynamo_flat_name_to_original_fqn which is coming from Dynamo.
                restore_fqn=False,
                assume_static_by_default=torch._dynamo.config.assume_static_by_default,
            )

            # Remove unused nodes. Should the signature of the graph be updated?
            gm.graph.lint()
            for node in reversed(gm.graph.nodes):
                if len(node.users) == 0 and node.op != "output":
                    gm.graph.erase_node(node)
            gm.recompile()

            # Compile the graph to produce kernel library
            with torch.no_grad():
                kernel_lib_path = torch._inductor.aot_compile(gm, args, kwargs, options=options)  # type: ignore[arg-type]

            # Get fake inputs to get symbolic shape information. The fake inputs will be mapped
            # to the actual input tensors in the kernel metadata.
            input_nodes = gm.graph.find_nodes(op="placeholder")
            if dynamic:
                assert all(
                    hasattr(node.meta, "val") is not None for node in input_nodes
                )
            fake_inputs = [
                node.meta.get("val") if dynamic else None for node in input_nodes
            ]

            kernel_metadata_items = []

            tensor_arg_offset = 0
            for idx, input in enumerate(flattened_inputs):
                if isinstance(input, torch.Tensor):
                    metadata = extract_tensor_metadata(
                        dynamic, input, fake_inputs[tensor_arg_offset]
                    )
                    tensor_arg_offset = tensor_arg_offset + 1
                elif isinstance(input, list):
                    assert all(isinstance(item, torch.Tensor) for item in input)
                    metadata = extract_tensor_list_metadata(
                        dynamic, input, fake_inputs[tensor_arg_offset:]
                    )
                    tensor_arg_offset = tensor_arg_offset + len(input)
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

            kernel_meta_info: Dict[str, Any] = {}
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
                    except Exception as e:
                        json_data = []

                    assert isinstance(json_data, list)
                    for item in json_data:
                        assert isinstance(item, dict)
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
