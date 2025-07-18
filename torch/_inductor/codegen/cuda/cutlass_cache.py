# mypy: allow-untyped-defs
import functools
import hashlib
import inspect
import json
import logging
import os
import time
from typing import Any, Optional

import torch._inductor.config as config
from torch._inductor.codecache import cutlass_key
from torch._inductor.codegen.cuda import cutlass_utils, serialization
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch, get_cuda_version
from torch._inductor.codegen.cuda.serialization import get_cutlass_operation_serializer
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.utils import clear_on_fresh_cache


log = logging.getLogger(__name__)


CONFIG_PREFIX: str = "configs"


def get_config_request_key(
    arch: str,
    cuda_version: str,
    instantiation_level: str,
) -> str:
    """
    Return a key for the full ops, based on cutlass key, arch, cuda version, instantiation level, and serialization.py file hash.
    """

    # Get hash of serialization.py and cutlass_utils.py files using their module file paths
    def get_file_hash(file_module):
        file_path = inspect.getfile(file_module)
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    serialization_hash = get_file_hash(serialization)
    cutlass_utils_hash = get_file_hash(cutlass_utils)

    hash_target = "-".join(
        [
            cutlass_key().hex(),
            arch,
            cuda_version,
            instantiation_level,
            serialization_hash,
            cutlass_utils_hash,
        ]
    )
    return hashlib.sha256(hash_target.encode("utf-8")).hexdigest()[0:8]


def _generate_config_filename(request_key: str) -> str:
    """
    Generate a filename for the full ops.
    """
    return f"{CONFIG_PREFIX}_{request_key}.json"


@clear_on_fresh_cache
@functools.cache
def maybe_fetch_ops() -> Optional[list[Any]]:
    """
    Fetch ops from databases.
    """
    if config.force_disable_caches:
        return None

    # setup
    arch: str = get_cuda_arch()
    # get_cuda_version might return "12.4.0" or "12.4"
    # but we want to use "12.4"
    version: str = ".".join(get_cuda_version().split(".")[:2])
    instantiation_level: str = config.cuda.cutlass_instantiation_level

    # filename and filepath
    request_key: str = get_config_request_key(arch, version, instantiation_level)
    filename: str = _generate_config_filename(request_key)
    filepath: str = os.path.join(cache_dir(), filename)

    # try fetch
    serialized_ops: Optional[list[str]] = None
    start_time = time.time()
    if os.path.isfile(filepath):
        # locally
        try:
            with open(filepath) as f:
                serialized_ops = json.load(f)

            assert isinstance(serialized_ops, list), (
                f"Expected serialized ops is a list, got {type(serialized_ops)}"
            )
        except Exception as e:
            log.warning(
                "Failed to load CUTLASS config %s from local cache: %s",
                filename,
                e,
            )
            serialized_ops = None
    elif config.is_fbcode():
        from torch._inductor.fb.cutlass_remote_cache import (
            maybe_fetch_cutlass_configs_from_remote,
        )

        # from remote
        serialized_ops = maybe_fetch_cutlass_configs_from_remote(filepath)

    if serialized_ops is None:
        return None

    # deserialize
    serializer = get_cutlass_operation_serializer()
    full_ops = [serializer.deserialize(x) for x in serialized_ops]  # type: ignore[union-attr]
    log.info("Loaded ops from %s cache in %.3fs", filename, time.time() - start_time)
    return full_ops
