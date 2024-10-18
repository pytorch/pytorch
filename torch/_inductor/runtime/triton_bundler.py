import dataclasses
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch._dynamo.utils import counters

from .runtime_utils import cache_dir


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TritonBundleEntry:
    kernel_hash: str
    device: int
    directory: str


@dataclasses.dataclass
class TritonKernelArtifact:
    filename: str
    payload: bytes = dataclasses.field(repr=False)  # Do not display binary


@dataclasses.dataclass
class TritonKernelArtifacts:
    kernel_hash: str
    device: int
    artifacts: List[TritonKernelArtifact]


class TritonBundler:
    # It is safe for multiple threads to insert entries as insert operation
    # of dict is atomic
    _entries: Optional[Dict[TritonBundleEntry, None]] = None

    @staticmethod
    def is_enabled() -> bool:
        from torch._inductor import config

        if (b := config.bundle_triton_into_fx_graph_cache) is not None:
            return b

        if not config.is_fbcode():
            return False

        return torch._utils_internal.justknobs_check(
            "pytorch/remote_cache:bundle_triton_into_fx_graph_cache"
        )

    @classmethod
    def begin_compile(cls) -> None:
        if not TritonBundler.is_enabled():
            return
        assert cls._entries is None
        cls._entries = {}

    @classmethod
    def put(cls, kernel_hash: str, device: int) -> None:
        if (entries := cls._entries) is not None:
            # CachingAutotuner.__init__ unconditionally sets TRITON_CACHE_DIR
            triton_cache_dir = os.getenv("TRITON_CACHE_DIR")
            assert triton_cache_dir is not None
            entries[TritonBundleEntry(kernel_hash, device, triton_cache_dir)] = None

    @classmethod
    def collect(cls) -> List[TritonKernelArtifacts]:
        if not TritonBundler.is_enabled():
            cls._entries = None
            return []

        if (entries := cls._entries) is not None:
            result: List[TritonKernelArtifacts] = []
            for entry in entries.keys():
                artifacts: List[TritonKernelArtifact] = []
                path = os.path.join(entry.directory, entry.kernel_hash)
                if not os.path.exists(path):
                    continue
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    try:
                        assert os.path.isfile(filepath)
                        with open(filepath, "rb") as file:
                            artifacts.append(
                                TritonKernelArtifact(filename, file.read())
                            )
                        counters["inductor"]["triton_bundler_save_kernel"] += 1
                    except Exception:
                        log.debug("failed to collect triton kernel", exc_info=True)
                if artifacts:
                    result.append(
                        TritonKernelArtifacts(
                            entry.kernel_hash, entry.device, artifacts
                        )
                    )
            cls._entries = None
            return result
        return []

    @staticmethod
    def write_bundle_to_file_system(bundle: List[TritonKernelArtifacts]) -> None:
        if not TritonBundler.is_enabled():
            return

        basedir = os.getenv("TRITON_CACHE_DIR")
        custom_dir = True
        if basedir is None:
            custom_dir = False
            basedir = os.path.join(cache_dir(), "triton")

        for artifacts in bundle:
            directory = os.path.join(basedir, artifacts.kernel_hash)
            if not custom_dir:
                # When we do not use a custom triton cache dir, we separate kernels
                # by their device
                directory = os.path.join(directory, str(artifacts.device))
            Path(directory).mkdir(parents=True, exist_ok=True)

            for artifact in artifacts.artifacts:
                filepath = os.path.join(directory, artifact.filename)
                with open(filepath, "wb") as file:
                    file.write(artifact.payload)
                counters["inductor"]["triton_bundler_write_kernel"] += 1
