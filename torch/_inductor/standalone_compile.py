from __future__ import annotations

import logging
import os
import pickle
import shutil
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING

import torch.fx
from torch._dynamo.utils import dynamo_timed
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch._inductor.utils import BoxedBool, InputType
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from . import config
from .utils import shape_env_from_inputs


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.compiler._cache import CacheInfo
    from torch.fx import GraphModule


log = logging.getLogger(__name__)


class CompiledArtifact:
    """
    CompiledArtifact class represents the precompiled inductor artifact that
    can be invoked in order to avoid repeated compilation.

    CompiledArtifact can be obtained by calling standalone_compile(gm, example_inputs)
    to create a fresh CompiledArtifact from a GraphModule and example inputs.

    Later this CompiledArtifact can be saved to disk, either as a binary or unpacked
    into the provided folder via the CompiledArtifact.save function.

    CompiledArtifact.load provides a way to create a CompiledArtifact from the
    binary or unpacked data.

    Finally, the CompiledArtifact can be invoked via the __call__ method
    to execute the precompiled artifact.
    """

    _compiled_fn: Callable[..., Any]
    _artifacts: Optional[tuple[bytes, CacheInfo]]

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: Optional[tuple[bytes, CacheInfo]],
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    def __call__(self, *args: Any) -> Any:
        return self._compiled_fn(*args)[0]

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        with dynamo_timed("CompiledArtifact.save"):
            if self._artifacts is None:
                raise RuntimeError(
                    "CompiledArtifact.save failed to save since there's no artifact to save"
                )
            artifact_bytes, cache_info = self._artifacts
            assert len(cache_info.aot_autograd_artifacts) == 1, cache_info
            key = cache_info.aot_autograd_artifacts[0]

            if format == "binary":
                # cant assert that it is a file since it might not exist yet
                assert not os.path.isdir(path)

                from torch.utils._appending_byte_serializer import BytesWriter

                from .codecache import torch_key

                writer = BytesWriter(0)
                writer.write_bytes(torch_key())
                writer.write_str(key)
                writer.write_bytes(artifact_bytes)
                with open(path, "wb") as file:
                    file.write(writer.to_bytes())
            else:
                assert format == "unpacked"
                assert os.path.isdir(path)
                shutil.rmtree(path, ignore_errors=True)

                from .codecache import FxGraphCache

                with temporary_cache_dir(path):
                    # This function unpacks the cache artifacts to disk
                    loaded_cache_info = torch.compiler.load_cache_artifacts(
                        artifact_bytes
                    )
                    assert loaded_cache_info is not None
                    # Now write all the output_code artifacts to disk so that
                    # they can be inspected and modified
                    for key in loaded_cache_info.inductor_artifacts:
                        subdir = FxGraphCache._get_tmp_dir_for_key(key)
                        if os.path.exists(subdir):
                            for path in sorted(os.listdir(subdir)):
                                with open(os.path.join(subdir, path), "rb") as f:
                                    graph = pickle.load(f)
                                output_file = graph.write_to_disk()
                                log.info("Output code written to: %s", output_file)

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        with dynamo_timed("CompiledArtifact.load"):
            if format == "binary":
                # cant assert that it is a file since it might not exist yet
                assert not os.path.isdir(path)
                with open(path, "rb") as file:
                    artifacts = file.read()
                from torch.utils._appending_byte_serializer import BytesReader

                from .codecache import torch_key

                reader = BytesReader(artifacts)
                assert reader.read_bytes() == torch_key()
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                assert reader.is_finished()

                torch.compiler.load_cache_artifacts(artifact_bytes)

                cache_dir_ctx: AbstractContextManager[None] = nullcontext()
            else:
                assert format == "unpacked"
                assert os.path.isdir(path)
                autograd_cache_dir = os.path.join(path, "aotautograd")
                assert os.path.isdir(autograd_cache_dir)
                files = list(os.listdir(autograd_cache_dir))
                assert len(files) == 1
                key = files[0]
                cache_dir_ctx = temporary_cache_dir(path)

            with cache_dir_ctx:
                with torch._functorch.config.patch(strict_autograd_cache=True):
                    from torch._functorch._aot_autograd.autograd_cache import (
                        AOTAutogradCache,
                    )

                    entry = AOTAutogradCache._lookup(key, local=True, remote=False)

                assert entry is not None

                from .compile_fx import _CompileFxKwargs

                fx_config = _CompileFxKwargs(
                    cudagraphs=BoxedBool(False),
                    boxed_forward_device_index=BoxedDeviceIndex(0),
                )

                context = torch._guards.TracingContext(
                    FakeTensorMode(shape_env=ShapeEnv())
                )
                with (
                    torch._guards.tracing(context),
                    config.patch(unsafe_skip_cache_dynamic_shape_guards=True),
                ):
                    compiled_fn = entry.wrap_post_compile(
                        [], entry.sanitized_aot_config, fx_config
                    )
            return CompiledArtifact(lambda *args: compiled_fn(list(args)), None)


def standalone_compile(
    gm: GraphModule, example_inputs: Sequence[InputType], **kwargs: Any
) -> CompiledArtifact:
    from torch.compiler._cache import CacheArtifactManager

    from .compile_fx import compile_fx

    shape_env = shape_env_from_inputs(example_inputs, default=True)
    assert shape_env is not None

    context = torch._guards.TracingContext(FakeTensorMode(shape_env=shape_env))
    with torch._guards.tracing(context):
        with CacheArtifactManager.with_fresh_cache():
            compiled_fn = compile_fx(gm, example_inputs, **kwargs)
            assert callable(compiled_fn)

            artifacts = torch.compiler.save_cache_artifacts()
            if artifacts is None:
                log.warning(
                    "standalone_compile artifact generation failed, cannot save. "
                    "Run with TORCH_LOGS=+torch._inductor.codecache to identify the problem"
                )

    return CompiledArtifact(compiled_fn, artifacts)
