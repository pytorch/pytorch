from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import sys
import tempfile
from typing import Any

import torch
from torch import Tensor


TENSOR_META_TAG = "__quack_tensor_meta__"
EPILOGUE_SOURCE_TAG = "__quack_epilogue_from_source__"

_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e5m2": torch.float8_e5m2,
    "torch.float8_e8m0fnu": torch.float8_e8m0fnu,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def serialize_worker_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {
            TENSOR_META_TAG: True,
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
        }
    if isinstance(value, tuple):
        return tuple(serialize_worker_value(v) for v in value)
    if isinstance(value, list):
        return [serialize_worker_value(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_worker_value(v) for k, v in value.items()}
    return value


def deserialize_worker_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get(TENSOR_META_TAG):
        return torch.empty_strided(
            value["shape"], value["stride"], dtype=_DTYPE_MAP[value["dtype"]], device="cuda"
        )
    if isinstance(value, tuple):
        return tuple(deserialize_worker_value(v) for v in value)
    if isinstance(value, list):
        return [deserialize_worker_value(v) for v in value]
    if isinstance(value, dict):
        return {k: deserialize_worker_value(v) for k, v in value.items()}
    return value


def epilogue_source_digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def make_epilogue_cache_key(symbol_name: str, source: str) -> str:
    return f"epilogue:{symbol_name}:sha256:{epilogue_source_digest(source)}"


def set_epilogue_source_cache_key(epilogue_fn: Any, source: str) -> str:
    cache_key = make_epilogue_cache_key(epilogue_fn.__name__, source)
    setattr(epilogue_fn, "__quack_cache_key__", cache_key)
    return cache_key


def make_epilogue_source_marker(
    symbol_name: str,
    epilogue_key: str | None,
    source: str,
) -> dict[str, Any]:
    return {
        EPILOGUE_SOURCE_TAG: True,
        "symbol_name": symbol_name,
        "epilogue_key": epilogue_key,
        "source": source,
        "source_digest": epilogue_source_digest(source),
        "cache_key": make_epilogue_cache_key(symbol_name, source),
    }


def is_epilogue_source_marker(value: Any) -> bool:
    return isinstance(value, dict) and value.get(EPILOGUE_SOURCE_TAG)


def _write_epilogue_module_if_needed(module_path: str, source: str) -> None:
    source_bytes = source.encode()
    lock_fd = os.open(f"{module_path}.lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if os.path.exists(module_path):
            with open(module_path, "rb") as f:
                if f.read() == source_bytes:
                    return
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(module_path)}.",
            suffix=".tmp",
            dir=os.path.dirname(module_path),
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(source_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, module_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def load_epilogue_from_source(marker: dict[str, Any]) -> Any:
    source = marker["source"]
    symbol_name = marker["symbol_name"]
    digest = epilogue_source_digest(source)
    module_name = f"quack_generated_epilogue_{digest[:16]}"
    module_dir = os.path.join(tempfile.gettempdir(), "quack_generated_epilogues")
    os.makedirs(module_dir, exist_ok=True)
    module_path = os.path.join(module_dir, f"{module_name}.py")
    _write_epilogue_module_if_needed(module_path, source)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load generated epilogue module {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    epilogue_fn = getattr(mod, symbol_name)
    setattr(epilogue_fn, "__quack_cache_key__", make_epilogue_cache_key(symbol_name, source))
    return epilogue_fn
