import dataclasses
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from torch._dynamo.utils import counters, dynamo_timed, set_feature_use
from torch._utils_internal import justknobs_check
from torch.utils._filelock import FileLock

from .runtime.runtime_utils import triton_cache_dir
from .utils import _IS_WINDOWS, GPU_KERNEL_BIN_EXTS


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TritonBundleEntry:
    """
    When we have compiled a triton kernel, we take note of that kernel by
    its triton generated hash, its device, and where this kernel is located.
    This is the minimum information we can use to later retrieve this kernel
    from file system.
    """

    kernel_hash: str
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
    device: int
    artifacts: list[TritonKernelArtifact]


@dataclasses.dataclass(frozen=True)
class TritonBundlerMetadata:
    """
    Metadata used for instrumentation
    """

    cached_kernel_names: list[str]


class TritonBundler:
    """
    Lightweight Triton Kernel bundler that notes each time we compile a triton
    kernel. When collect is called, converts all the previously noted kernels and
    their artifacts into a structured bytes blob, and later when write is called
    it writes this structured blob back to file system.

    Intended Life cycle:
    - TritonBundler.begin_compile is called when we start compiling in Inductor
    - TritonBundler.put is called each time a Triton Kernel is compiled
    - TritonBundler.collect is called when a cache entry is being generated
    - TritonBundler.end_compile is called to indicate bundling is completed,
      collect will execute this function as well.
    - TritonBundler.read_and_emit is called when a cache entry is read
    """

    _entries: Optional[list[TritonBundleEntry]] = None

    # __grp__kernel_name.json contains metadata with source code paths
    # we use this as sentinal value for search and replace
    _REPLACE_BYTES: bytes = b"[REPLACE]"

    @staticmethod
    def is_enabled() -> bool:
        from torch._inductor import config

        if config.force_disable_caches:
            return False

        if (b := config.bundle_triton_into_fx_graph_cache) is not None:
            return b

        if not config.is_fbcode():
            return False

        return justknobs_check(
            "pytorch/remote_cache:bundle_triton_into_fx_graph_cache_v2"
        )

    @classmethod
    def begin_compile(cls) -> None:
        """
        Initializes the TritonBundler.
        The current TritonBundler bundle is finalized by TritonBundler.collect.
        """
        if not TritonBundler.is_enabled():
            return
        log.debug("TritonBundler.begin_compile is called")
        assert cls._entries is None
        cls._entries = []

    @classmethod
    def end_compile(cls) -> None:
        """
        Finalizes the TritonBundler. If collect is not yet called, it
        discards the current bundle.
        """
        log.debug("TritonBundler.end_compile is called")
        cls._entries = None

    @classmethod
    def put(cls, kernel_hash: str, device: int) -> None:
        """
        Lazily observes that we have seen a Triton kernel compilation. Remembers
        it for when collect is later called.
        """
        if (entries := cls._entries) is not None:
            entries.append(
                TritonBundleEntry(kernel_hash, device, triton_cache_dir(device))
            )

    @classmethod
    def collect(
        cls,
    ) -> tuple[list[TritonKernelArtifacts], Optional[TritonBundlerMetadata]]:
        """
        This is the main function called when a cache write happens. This function
        converts all the previously remembered kernels into bundled format so that
        it can be written into a cache entry.
        This function also finalizes the current bundle.
        """
        if not TritonBundler.is_enabled():
            cls.end_compile()
            set_feature_use("triton_bundling", False)
            return [], None
        set_feature_use("triton_bundling", True)

        with dynamo_timed(key="TritonBundler.collect", log_pt2_compile_event=True):
            entries = cls._entries
            if entries is not None:
                result: list[TritonKernelArtifacts] = []
                kernel_names: list[str] = []
                for entry in entries:
                    artifacts: list[TritonKernelArtifact] = []
                    path = os.path.join(entry.directory, entry.kernel_hash)
                    if not os.path.exists(path):
                        continue
                    for filename in os.listdir(path):
                        filepath = os.path.join(path, filename)
                        try:
                            assert os.path.isfile(filepath)
                            with open(filepath, "rb") as file:
                                payload = file.read()
                                if filepath.endswith(".json"):
                                    # Make sure there's no sentinel value
                                    if TritonBundler._REPLACE_BYTES in payload:
                                        log.warning(
                                            "Bundle contains illegal %s, payload: %s",
                                            TritonBundler._REPLACE_BYTES,
                                            payload,
                                        )
                                        raise AssertionError(
                                            "Bundle contains illegal bytes"
                                        )
                                    # Remove the path from payload
                                    payload = payload.replace(
                                        str.encode(path), TritonBundler._REPLACE_BYTES
                                    )
                                artifacts.append(
                                    TritonKernelArtifact(filename, payload)
                                )
                            counters["inductor"]["triton_bundler_save_kernel"] += 1
                        except Exception:
                            log.debug("failed to collect triton kernel", exc_info=True)
                        extension = os.path.splitext(filename)[1]
                        if extension in GPU_KERNEL_BIN_EXTS.values():
                            # Each kernel has bunch of files like .cubin(for cuda), .spv(for xpu), .json, .ttir
                            # Just append one of them without the extension
                            kernel_names.append(Path(filename).stem)
                    if artifacts:
                        result.append(
                            TritonKernelArtifacts(
                                entry.kernel_hash,
                                entry.device,
                                artifacts,
                            )
                        )
                cls.end_compile()
                return result, TritonBundlerMetadata(kernel_names)
            return [], None

    @staticmethod
    def read_and_emit(
        bundle: list[TritonKernelArtifacts],
    ) -> Optional[TritonBundlerMetadata]:
        """
        This is the main function called when a cache read happens. This function
        converts the bundled format back into individual files and writes them
        to the filesystem.

        NOTE: When we are writing to the filesystem, we assume exclusive access
        to the target directory.
        This means that if the target folder already exists and is non-empty,
        we bail out.
        Exclusive access means that no other process should be writing to
        or reading from the target directory.
        """
        if not TritonBundler.is_enabled():
            return None

        with dynamo_timed(
            key="TritonBundler.read_and_emit", log_pt2_compile_event=True
        ):
            kernel_names: list[str] = []

            for artifacts in bundle:
                basedir = triton_cache_dir(artifacts.device)
                directory = os.path.join(basedir, artifacts.kernel_hash)

                if os.path.exists(directory) and len(os.listdir(directory)) != 0:
                    # If directory already exists, we bail out and leave
                    # local disk to take care of caching
                    log.debug(
                        "Bailing out TritonBundler.read_and_emit, %s is non empty",
                        directory,
                    )
                    continue

                Path(basedir).mkdir(parents=True, exist_ok=True)

                # Random ID to avoid any collisions
                rnd_id = str(uuid.uuid4())
                tmp_dir = os.path.join(basedir, f"tmp.{rnd_id}")
                os.makedirs(tmp_dir)

                for artifact in artifacts.artifacts:
                    filepath = os.path.join(tmp_dir, artifact.filename)
                    with open(filepath, "wb") as file:
                        payload = artifact.payload
                        if artifact.filename.endswith(".json"):
                            payload = payload.replace(
                                TritonBundler._REPLACE_BYTES, str.encode(directory)
                            )
                        file.write(payload)
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"] += 1
                    extension = os.path.splitext(artifact.filename)[1]
                    if extension in GPU_KERNEL_BIN_EXTS.values():
                        # Each kernel has bunch of files like .cubin(for cuda), spv(for xpu), .json, .ttir
                        # Just append one of them without the extension
                        kernel_names.append(Path(artifact.filename).stem)

                if _IS_WINDOWS:
                    with FileLock(directory + ".lock"):
                        if os.path.exists(directory):
                            shutil.rmtree(directory)
                        os.replace(tmp_dir, directory)
                else:
                    # Atomic on POSIX systems
                    os.replace(tmp_dir, directory)

            return TritonBundlerMetadata(kernel_names)
