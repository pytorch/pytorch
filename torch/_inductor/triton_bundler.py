import dataclasses
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from torch._dynamo.utils import counters, dynamo_timed
from torch._utils_internal import justknobs_check
from torch.utils._ordered_set import OrderedSet

from .runtime.runtime_utils import cache_dir


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TritonBundleEntry:
    """
    When we have compiled a triton kernel, we take note of that kernel by
    its triton generated hash, its device, and where this kernel is located.
    This is the minimum information we can use to later retreive this kernel
    from file system.
    """

    kernel_hash: str
    compile_time_ns: int
    device: int
    directory: str


@dataclasses.dataclass(frozen=True)
class TritonKernelArtifact:
    """
    Artifact for an individual kernel converted to bytes.
    Bytes could be a cubin, json, ttir, or ttgir.
    """

    filename: str
    payload: bytes = dataclasses.field(repr=False)  # Do not display binary


@dataclasses.dataclass(frozen=True)
class TritonKernelArtifacts:
    """
    Collection of artifacts for a particular kernel.
    """

    kernel_hash: str
    compile_time_ns: int
    device: int
    artifacts: List[TritonKernelArtifact]


@dataclasses.dataclass(frozen=True)
class TritonBundlerMetadata:
    """
    Metadata used for instrumentation
    """

    num_kernels: int
    total_compile_time_ns: int


class TritonBundler:
    """
    Lightweight Triton Kernel bundler that notes each time we compile a triton
    kernel. When collect is called, converts all the previously noted kernels and
    their artifacts into a structured bytes blob, and later when write is called
    it writes this structured blob back to file system.
    """

    # It is safe for multiple threads to insert entries as insert operation
    # of OrderedSet which uses dict underhood is atomic
    _entries: Optional[OrderedSet[TritonBundleEntry]] = None

    @staticmethod
    def is_enabled() -> bool:
        from torch._inductor import config

        if (b := config.bundle_triton_into_fx_graph_cache) is not None:
            return b

        if not config.is_fbcode():
            return False

        return justknobs_check("pytorch/remote_cache:bundle_triton_into_fx_graph_cache")

    @classmethod
    def begin_compile(cls) -> None:
        if not TritonBundler.is_enabled():
            return
        assert cls._entries is None
        cls._entries = OrderedSet()

    @classmethod
    def put(cls, kernel_hash: str, compile_time_ns: int, device: int) -> None:
        if (entries := cls._entries) is not None:
            # CachingAutotuner.__init__ unconditionally sets TRITON_CACHE_DIR
            triton_cache_dir = os.getenv("TRITON_CACHE_DIR")
            assert triton_cache_dir is not None
            entries.add(
                TritonBundleEntry(
                    kernel_hash, compile_time_ns, device, triton_cache_dir
                )
            )

    @classmethod
    def collect(
        cls,
    ) -> Tuple[List[TritonKernelArtifacts], Optional[TritonBundlerMetadata]]:
        if not TritonBundler.is_enabled():
            cls._entries = None
            return [], None

        with dynamo_timed(
            key="TritonBundler::collect", phase_name="TritonBundler", fwd_only=False
        ):
            if (entries := cls._entries) is not None:
                result: List[TritonKernelArtifacts] = []
                total_compile_time_ns = 0
                for entry in entries:
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
                                entry.kernel_hash,
                                entry.compile_time_ns,
                                entry.device,
                                artifacts,
                            )
                        )
                        total_compile_time_ns += entry.compile_time_ns
                cls._entries = None
                num_kernels = len(result)
                return result, TritonBundlerMetadata(num_kernels, total_compile_time_ns)
            return [], None

    @staticmethod
    def write(bundle: List[TritonKernelArtifacts]) -> Optional[TritonBundlerMetadata]:
        if not TritonBundler.is_enabled():
            return None

        with dynamo_timed(
            key="TritonBundler::write", phase_name="TritonBundler", fwd_only=False
        ):
            num_kernels = 0
            total_compile_time_ns = 0
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

                with tempfile.TemporaryDirectory() as tmp_dir:
                    for artifact in artifacts.artifacts:
                        filepath = os.path.join(tmp_dir, artifact.filename)
                        with open(filepath, "wb") as file:
                            file.write(artifact.payload)
                        counters["inductor"]["triton_bundler_write_kernel"] += 1
                    shutil.copytree(tmp_dir, directory, dirs_exist_ok=True)
                num_kernels += 1
                total_compile_time_ns += artifacts.compile_time_ns
            return TritonBundlerMetadata(num_kernels, total_compile_time_ns)
