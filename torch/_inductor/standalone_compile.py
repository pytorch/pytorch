from __future__ import annotations

import copy
import logging
import os
import pickle
import shutil
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING

import torch.fx
from torch._dynamo.utils import dynamo_timed
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch._inductor.utils import BoxedBool, InputType
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from . import config


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
        return self._compiled_fn(*args)

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
                # can't assert that it is a file since it might not exist yet
                assert not os.path.isdir(path)

                from torch.utils._appending_byte_serializer import BytesWriter

                from .codecache import torch_key

                writer = BytesWriter()
                writer.write_bytes(torch_key())
                writer.write_str(key)
                writer.write_bytes(artifact_bytes)

                from torch._inductor.codecache import write_atomic

                write_atomic(path, writer.to_bytes())
            else:
                assert format == "unpacked"
                if os.path.exists(path):
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
                        assert os.path.exists(subdir)
                        for path in sorted(os.listdir(subdir)):
                            with open(os.path.join(subdir, path), "rb") as f:
                                graph = pickle.load(f)
                            output_file = graph.write_to_disk()
                            log.info("Output code written to: %s", output_file)

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        path = normalize_path_separator(path)
        with dynamo_timed("CompiledArtifact.load"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
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

            with (
                cache_dir_ctx,
                config.patch(unsafe_skip_cache_dynamic_shape_guards=True),
            ):
                with torch._functorch.config.patch(strict_autograd_cache=True):
                    from torch._functorch._aot_autograd.autograd_cache import (
                        AOTAutogradCache,
                    )

                    entry = AOTAutogradCache._lookup(
                        key,
                        local=True,
                        remote=False,
                        args=[],
                        cache_info={},
                        aot_config=None,
                    )

                assert entry is not None

                from .compile_fx import _CompileFxKwargs

                fx_config = _CompileFxKwargs(
                    cudagraphs=BoxedBool(False),
                    boxed_forward_device_index=BoxedDeviceIndex(0),
                )

                context = torch._guards.TracingContext(
                    FakeTensorMode(shape_env=ShapeEnv())
                )
                with torch._guards.tracing(context):
                    compiled_fn = entry.wrap_post_compile(
                        [], entry.sanitized_aot_config, fx_config
                    )
            return CompiledArtifact(lambda *args: compiled_fn(list(args)), None)


def standalone_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: Any,
    options: Any,
) -> CompiledArtifact:
    from torch.compiler._cache import CacheArtifactManager

    from .compile_fx import compile_fx

    ignore_shape_env = False
    if dynamic_shapes == "from_example_inputs":
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        # tells compile_fx to ignore the shape_envs on the ambient context
        # and the graph_module.
        ignore_shape_env = True
    elif dynamic_shapes == "from_tracing_context":
        # Reuse fake_mode from the TracingContext.
        # NB: The TracingContext only exists if we're currently in a torch.compile backend.
        context = torch._guards.TracingContext.get()
        assert context.fake_mode is not None
        fake_mode = context.fake_mode
    elif dynamic_shapes == "from_graph":
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        # Strategy: find a FakeTensor in the graph output, grab its FakeTensorMode.
        # The graph passed to standalone_compile must be an Inductor-approved graph,
        # which means that there is at least one Tensor output and the output node
        # contains a flat list of Tensors.
        last_node = next(iter(reversed(gm.graph.nodes)))
        assert last_node.op == "output"
        assert len(last_node.args) == 1

        def handle_node(node: torch.fx.Node) -> None:
            nonlocal fake_mode
            if "example_value" in node.meta:
                maybe_tensor = node.meta["example_value"]
                if isinstance(maybe_tensor, torch._subclasses.fake_tensor.FakeTensor):
                    fake_mode = maybe_tensor.fake_mode

        # If gm came from Dynamo, then last_node.args[0] is always a list,
        # even in single-Tensor returns.
        #
        # It's possible to get into a situation where last_node.args[0]
        # is a Node (and not a list!). This happens if you call split_module
        # on the graph. We allow for this case since it is common.
        if isinstance(last_node.args[0], torch.fx.Node):
            handle_node(last_node.args[0])
        else:
            for node in last_node.args[0]:
                handle_node(node)

    else:
        raise ValueError(
            f"standalone_compile got unsupported `dynamic_shapes` value: dynamic_shapes={dynamic_shapes}."
        )

    context = torch._guards.TracingContext(fake_mode)
    with (
        torch._guards.tracing(context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
    ):
        # compile_fx can mutate gm
        gm = copy.deepcopy(gm)
        compiled_fn = compile_fx(
            gm, example_inputs, ignore_shape_env=ignore_shape_env, **options
        )
        assert callable(compiled_fn)

        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            log.warning(
                "standalone_compile artifact generation failed, cannot save. "
                "Run with TORCH_LOGS=+torch._inductor.codecache to identify the problem"
            )

    return CompiledArtifact(compiled_fn, artifacts)
