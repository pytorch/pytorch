from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
from time import time
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import torch

from torch._inductor import exc
from torch._inductor.codecache import DLLWrapper, get_lock_dir, write
from torch._inductor.codegen.rocm.compile_command import rocm_compile_command, rocm_compiler_version
from torch._inductor.utils import clear_on_fresh_inductor_cache

output_code_log = torch._logging.getArtifactLogger(__name__, "output_code")

LOCK_TIMEOUT = 600

log = logging.getLogger(__name__)

@clear_on_fresh_inductor_cache
class ROCmCodeCache:
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str

    cache: Dict[str, CacheEntry] = dict()
    cache_clear = staticmethod(cache.clear)
    _SOURCE_CODE_SUFFIX = "cpp"

    log.debug(f"HIP compiler version:\n{rocm_compiler_version()}")

    @classmethod
    def write(cls, source_code, dst_file_ext) -> Tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """

        cuda_command = repr(
            rocm_compile_command(["dummy_input"], "dummy_output", dst_file_ext)
        )
        key, input_path = write(
            source_code, cls._SOURCE_CODE_SUFFIX, extra=cuda_command
        )
        return key, input_path

    @classmethod
    def compile(
        cls, source_code, dst_file_ext, extra_args: Optional[List[str]] = None
    ) -> Tuple[str, str, str]:
        """
        Compiles CUDA source_code into a file with dst_file_ext extension.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
        key, input_path = cls.write(source_code, dst_file_ext)
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[: -len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                if not os.path.exists(output_path):
                    cmd = rocm_compile_command(
                        [input_path], output_path, dst_file_ext, extra_args
                    )
                    start_time = time()
                    cmd_parts = cmd.split(" ")
                    try:
                        output = subprocess.check_output(
                            cmd_parts, stderr=subprocess.STDOUT, text=True, env=os.environ
                        )
                        log.debug("Compilation output: %s", output)
                    except subprocess.CalledProcessError as error:
                        raise exc.CUDACompileError(cmd_parts, error.output) from error
                    end_time = time()
                    log_duration_msg = f"Compilation took {end_time-start_time} seconds. Compile command: {cmd}"
                    log.info(log_duration_msg)
                else:
                    log.debug(
                        "Compilation skipped: %s since output already exists",
                        input_path,
                    )
                cls.cache[key] = ROCmCodeCache.CacheEntry(input_path, output_path)

        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code, dst_file_ext) -> Tuple[DLLWrapper, str, str]:
        """
        Compiles source code and loads the generated .so file.
        Returns a tuple of DLLWrapper, hash_key, source_code_path
        """

        if dst_file_ext != "so":
            raise RuntimeError(
                f"Only support loading a .so file for now. "
                f"Requested file extension: {dst_file_ext}. Source code: {source_code}"
            )
        dst_file_path, hash_key, source_code_path = cls.compile(
            source_code, dst_file_ext
        )
        return (DLLWrapper(dst_file_path), hash_key, source_code_path)
