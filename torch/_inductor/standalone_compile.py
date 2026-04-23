from __future__ import annotations

import contextlib
import copy
import logging
import operator
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, TYPE_CHECKING


DynamicShapesType = Literal["from_example_inputs", "from_tracing_context", "from_graph"]

import torch.fx
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.utils import dynamo_timed
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch._inductor.utils import BoxedBool, InputType
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from . import config


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.compiler._cache import CacheInfo
    from torch.fx import GraphModule


log = logging.getLogger(__name__)


def _rewrite_legacy_collectives(gm: GraphModule) -> GraphModule:
    if not torch.distributed.is_available():
        return gm

    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    reduce_op_to_str = {
        0: "sum",
    }

    def _erase_get_attr_if_dead(node: torch.fx.Node) -> None:
        if node.op == "get_attr" and not node.users:
            target = node.target
            gm.graph.erase_node(node)
            if isinstance(target, str) and hasattr(gm, target):
                delattr(gm, target)

    def _resolve_group_name(pg_node: torch.fx.Node) -> str:
        if pg_node.op != "get_attr":
            raise RuntimeError(
                "Expected c10d.allreduce_ process group argument to come from get_attr"
            )
        pg_names = list(c10d._world.pg_names.values())
        if len(pg_names) != 1:
            raise RuntimeError(
                "standalone_compile only supports rewriting c10d.allreduce_ when exactly one "
                "process group is registered. Use functional collectives in the graph instead."
            )
        return pg_names[0]

    def _resolve_reduce_op_str(reduce_op_node: torch.fx.Node) -> str:
        if reduce_op_node.op != "get_attr":
            raise RuntimeError(
                "Expected c10d.allreduce_ reduce-op argument to come from get_attr"
            )
        reduce_op = getattr(gm, reduce_op_node.target)
        try:
            reduce_op_int = reduce_op.op
            if callable(reduce_op_int):
                reduce_op_int = reduce_op_int()
        except AttributeError as exc:
            raise RuntimeError(
                "Unable to inspect reduce op for c10d.allreduce_ rewrite"
            ) from exc
        if reduce_op_int not in reduce_op_to_str:
            raise RuntimeError(
                f"Unsupported reduce op for c10d.allreduce_ rewrite: op={reduce_op_int!r}"
            )
        return reduce_op_to_str[reduce_op_int]

    changed = False

    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target is dist.all_reduce:
            raise RuntimeError(
                "standalone_compile expected make_fx to lower dist.all_reduce before compile time. "
                "Please use the traced c10d form or functional collectives."
            )

        if node.op != "call_function" or node.target is not torch.ops.c10d.allreduce_.default:
            continue

        if len(node.args) < 5:
            raise RuntimeError(
                "Unexpected c10d.allreduce_ signature in standalone_compile rewrite"
            )
        tensors, pg_node, reduce_op_node, sparse_indices, async_op = node.args[:5]
        timeout = node.args[5] if len(node.args) > 5 else -1
        if sparse_indices is not None or timeout != -1:
            raise RuntimeError(
                "standalone_compile only supports basic c10d.allreduce_ rewrites"
            )
        if async_op:
            raise RuntimeError(
                "standalone_compile does not support async c10d.allreduce_ traced via make_fx. "
                "Use functional collectives in the graph instead."
            )
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            raise RuntimeError(
                "standalone_compile only supports single-tensor c10d.allreduce_ rewrites"
            )
        tensor = tensors[0]
        if not isinstance(tensor, torch.fx.Node):
            raise RuntimeError(
                "Expected c10d.allreduce_ tensor argument to be an FX node"
            )
        group_name = _resolve_group_name(pg_node)
        reduce_op = _resolve_reduce_op_str(reduce_op_node)

        first_output = None
        work_output = None
        for user in list(node.users):
            if (
                user.op == "call_function"
                and user.target is operator.getitem
                and len(user.args) == 2
                and user.args[0] is node
            ):
                if user.args[1] == 0:
                    first_output = user
                elif user.args[1] == 1:
                    work_output = user
        if first_output is None:
            raise RuntimeError(
                "Expected c10d.allreduce_ output tuple to have a getitem(..., 0) user"
            )
        tensor_output = None
        for user in list(first_output.users):
            if (
                user.op == "call_function"
                and user.target is operator.getitem
                and len(user.args) == 2
                and user.args[0] is first_output
                and user.args[1] == 0
            ):
                tensor_output = user
                break
        if tensor_output is None:
            raise RuntimeError(
                "Expected c10d.allreduce_ tensor list output to have a getitem(..., 0) user"
            )

        with gm.graph.inserting_before(node):
            collective = gm.graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                args=(tensor, reduce_op, group_name),
            )
            wait = gm.graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default, args=(collective,)
            )
            copy_ = gm.graph.call_function(torch.ops.aten.copy_.default, (tensor, wait))

        collective.meta = dict(node.meta)
        wait.meta = dict(node.meta)
        copy_.meta = dict(node.meta)
        for key in ("val", "example_value", "tensor_meta"):
            if key in tensor.meta:
                collective.meta[key] = tensor.meta[key]
                wait.meta[key] = tensor.meta[key]
                copy_.meta[key] = tensor.meta[key]

        tensor_output.replace_all_uses_with(tensor)

        gm.graph.erase_node(tensor_output)
        if not first_output.users:
            gm.graph.erase_node(first_output)
        if work_output is not None and not work_output.users:
            gm.graph.erase_node(work_output)
        gm.graph.erase_node(node)
        _erase_get_attr_if_dead(pg_node)
        _erase_get_attr_if_dead(reduce_op_node)
        changed = True

    if changed:
        gm.delete_all_unused_submodules()
        gm.graph.lint()
        gm.recompile()
    return gm


class CompiledArtifact(ABC):
    """
    CompiledArtifact class represents the inductor cache artifacts that
    can be invoked in order to avoid repeated compilation.

    CompiledArtifact can be obtained by calling standalone_compile(gm, example_inputs)
    to create a fresh CompiledArtifact from a GraphModule and example inputs.

    Later this CompiledArtifact can be saved to disk, either as a binary or unpacked
    into the provided folder via the CompiledArtifact.save function.

    CompiledArtifact.load provides a way to create a CompiledArtifact from the
    binary or unpacked data.

    Finally, the CompiledArtifact can be invoked via the __call__ method
    to execute the cached artifact.
    """

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: tuple[bytes, CacheInfo] | None,
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    @abstractmethod
    def __call__(self, *args: Any) -> Any: ...

    @abstractmethod
    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None: ...

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        if format == "unpacked":
            # If format is unpacked, it must be a CacheCompiledArtifact
            return CacheCompiledArtifact.load(path=path, format=format)

        assert format == "binary"
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header == AOTCompiledArtifact.AOT_HEADER:
                assert reader.read_bytes() == torch_key()
                artifact = reader.read_bytes()
                assert reader.is_finished()
                return AOTCompiledArtifact.deserialize(artifact)
            # Otherwise, it's in the CacheCompiledArtifact format
            elif header == CacheCompiledArtifact.CACHE_HEADER:
                assert reader.read_bytes() == torch_key()
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                assert reader.is_finished()
                torch.compiler.load_cache_artifacts(artifact_bytes)
                return CacheCompiledArtifact._load_impl(nullcontext(), key)
            else:
                raise RuntimeError(
                    "Invalid header, expected CacheCompiledArtifact or AOTCompiledArtifact, got: "
                    + header.decode("utf-8")
                )


class CacheCompiledArtifact(CompiledArtifact):
    """
    CompiledArtifact that depends on torch.compiler.save_cache_artifacts
    """

    CACHE_HEADER = bytes("CacheCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: tuple[bytes, CacheInfo] | None,
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    def __call__(self, *args: Any) -> Any:
        return self._compiled_fn(*args)

    def is_saveable(self) -> bool:
        if self._artifacts is None:
            return False
        _, cache_info = self._artifacts
        # 0 means nothing was saved
        # >1 means multiple artifacts were saved, which is concerning
        # (we only expect one)
        return len(cache_info.aot_autograd_artifacts) == 1

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        with dynamo_timed("CompiledArtifact.save"):
            if self._artifacts is None:
                raise RuntimeError(
                    "CompiledArtifact.save failed to save since there's no artifact to save"
                )
            artifact_bytes, cache_info = self._artifacts
            if len(cache_info.aot_autograd_artifacts) == 0:
                raise RuntimeError(
                    f"CompiledArtifact.save failed to save due to no aot_autograd artifacts. "
                    f"This likely means there was something that was not serializable in the "
                    f"graph passed to standalone_compile. This can generally be fixed by "
                    f"ensuring that your model only uses constructs that are serializable. "
                    f"{cache_info}"
                )
            if len(cache_info.aot_autograd_artifacts) > 1:
                raise AssertionError(
                    f"CompiledArtifact.save failed to save because there was more than one "
                    f"artifact but we only expected one. {cache_info}"
                )
            key = cache_info.aot_autograd_artifacts[0]

            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                assert not os.path.isdir(path)

                from torch.utils._appending_byte_serializer import BytesWriter

                from .codecache import torch_key

                writer = BytesWriter()
                writer.write_bytes(CacheCompiledArtifact.CACHE_HEADER)
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
    def _load_impl(
        cache_dir_ctx: AbstractContextManager[Any], key: str
    ) -> CompiledArtifact:
        with (
            cache_dir_ctx,
            config.patch(unsafe_skip_cache_dynamic_shape_guards=True),
        ):
            with torch._functorch.config.patch(strict_autograd_cache=True):
                from torch._functorch._aot_autograd.autograd_cache import (
                    AOTAutogradCache,
                )

                result = AOTAutogradCache._lookup(
                    key,
                    local=True,
                    remote=False,
                    args=[],
                    cache_info={},
                    aot_config=None,
                )

            assert result is not None
            (entry, _) = result

            from .compile_fx import _CompileFxKwargs

            fx_config = _CompileFxKwargs(
                cudagraphs=BoxedBool(False),
                boxed_forward_device_index=BoxedDeviceIndex(0),
            )

            context = torch._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
            with torch._guards.tracing(context):
                compiled_fn = entry.wrap_post_compile(
                    [], entry.sanitized_aot_config, fx_config
                )
        return CacheCompiledArtifact(lambda *args: compiled_fn(list(args)), None)

    @staticmethod
    def _prepare_load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> tuple[str, AbstractContextManager[Any]]:
        """
        Do format specific prep and loads, return a context manager and key
        """
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
                return key, nullcontext()
            else:
                assert format == "unpacked"
                assert os.path.isdir(path)
                autograd_cache_dir = os.path.join(path, "aotautograd")
                assert os.path.isdir(autograd_cache_dir)
                files = list(os.listdir(autograd_cache_dir))
                assert len(files) == 1
                key = files[0]
                cache_dir_ctx = temporary_cache_dir(path)
                return key, cache_dir_ctx

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        key, cache_dir_ctx = CacheCompiledArtifact._prepare_load(
            path=path, format=format
        )
        return CacheCompiledArtifact._load_impl(cache_dir_ctx, key)


class AOTCompiledArtifact(CompiledArtifact):
    """
    Similar to CompiledArtifact, but the object is a single, bundled precompiled function.
    This object is always a serializable callable function.

    This object is essentially a wrapper for BundledAOTAutogradSerializableCallable, which
    is used by torch._dynamo.aot_compile for AOT Precompilation.
    """

    AOT_HEADER = bytes("AOTCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
    ):
        self.inner_fn = BundledAOTAutogradSerializableCallable(compiled_fn)
        self._artifacts = (
            None  # We don't need artifacts, the inner object handles everything
        )

    @staticmethod
    def from_bundled_callable(
        bundled_fn: BundledAOTAutogradSerializableCallable,
    ) -> AOTCompiledArtifact:
        return AOTCompiledArtifact(bundled_fn.compiled_fn)

    def __call__(self, *args: Any) -> Any:
        return self.inner_fn(*args)

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        result_bytes = self.serialize()
        from torch.utils._appending_byte_serializer import BytesWriter

        from .codecache import torch_key

        writer = BytesWriter()
        writer.write_bytes(AOTCompiledArtifact.AOT_HEADER)
        writer.write_bytes(torch_key())
        writer.write_bytes(result_bytes)

        from torch._inductor.codecache import write_atomic

        # Save a sentinel file to indicate that this is AOT
        write_atomic(path, writer.to_bytes())

    def serialize(self) -> bytes:
        return BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
            self.inner_fn
        )

    @staticmethod
    def deserialize(result_bytes: bytes) -> AOTCompiledArtifact:
        deserialized = (
            BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                result_bytes
            )
        )
        assert isinstance(deserialized, BundledAOTAutogradSerializableCallable)
        return AOTCompiledArtifact.from_bundled_callable(deserialized)

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            assert header == AOTCompiledArtifact.AOT_HEADER
            assert reader.read_bytes() == torch_key()
            artifact = reader.read_bytes()
            assert reader.is_finished()
            return AOTCompiledArtifact.deserialize(artifact)


def _resolve_ignore_shape_env(dynamic_shapes: DynamicShapesType):
    # tells compile_fx to ignore the shape_envs on the ambient context
    # and the graph_module.
    return dynamic_shapes == "from_example_inputs"


def _resolve_fake_mode(
    gm: GraphModule, dynamic_shapes: DynamicShapesType
) -> FakeTensorMode:
    if dynamic_shapes == "from_example_inputs":
        return FakeTensorMode(shape_env=ShapeEnv())
    elif dynamic_shapes == "from_tracing_context":
        # Reuse fake_mode from the TracingContext.
        # NB: The TracingContext only exists if we're currently in a torch.compile backend.
        context = torch._guards.TracingContext.get()
        assert context.fake_mode is not None
        return context.fake_mode
    elif dynamic_shapes == "from_graph":
        # Strategy: find a FakeTensor in the graph output, grab its FakeTensorMode.
        # The graph passed to standalone_compile must be an Inductor-approved graph,
        # which means that there is at least one Tensor output and the output node
        # contains a flat list of Tensors.
        last_node = next(iter(reversed(gm.graph.nodes)))
        assert last_node.op == "output"
        assert len(last_node.args) == 1

        # If gm came from Dynamo, then last_node.args[0] is always a list,
        # even in single-Tensor returns.
        #
        # It's possible to get into a situation where last_node.args[0]
        # is a Node (and not a list!). This happens if you call split_module
        # on the graph. We allow for this case since it is common.
        nodes = (
            [last_node.args[0]]
            if isinstance(last_node.args[0], torch.fx.Node)
            else last_node.args[0]
        )
        for node in nodes:
            if "example_value" in node.meta:
                maybe_tensor = node.meta["example_value"]
                if isinstance(maybe_tensor, torch._subclasses.fake_tensor.FakeTensor):
                    return maybe_tensor.fake_mode

        return FakeTensorMode(shape_env=ShapeEnv())
    else:
        raise ValueError(
            f"standalone_compile got unsupported `dynamic_shapes` value: dynamic_shapes={dynamic_shapes}."
        )


@contextlib.contextmanager
def _standalone_context(gm: GraphModule, dynamic_shapes: DynamicShapesType, aot: bool):
    from torch.compiler._cache import CacheArtifactManager

    fake_mode = _resolve_fake_mode(gm, dynamic_shapes)
    tracing_context = torch._guards.TracingContext(fake_mode)
    with (
        torch._guards.tracing(tracing_context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
        torch._functorch.config.patch("bundled_autograd_cache", aot),
    ):
        yield


def standalone_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: DynamicShapesType,
    options: Any,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    donate_graph_module: bool = False,
) -> CompiledArtifact:
    """
    Implementation of torch.inductor.standalone_compile
    """
    from .compile_fx import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(gm, dynamic_shapes, aot):
        gm = _rewrite_legacy_collectives(gm)
        # compile_fx takes ownership of gm and may mutate it on cache miss.
        if not donate_graph_module:
            gm = copy.deepcopy(gm)
        compiled_fn = compile_fx(
            gm, example_inputs, ignore_shape_env=ignore_shape_env, **options
        )
        assert callable(compiled_fn)
        if aot:
            if not hasattr(compiled_fn, "serialize"):
                raise RuntimeError(
                    "Compiled function should have serialize method when aot=True"
                )
            return AOTCompiledArtifact(compiled_fn)
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            log.warning(
                "standalone_compile artifact generation failed, cannot save. "
                "Run with TORCH_LOGS=+torch._inductor.codecache to identify the problem"
            )

    return CacheCompiledArtifact(compiled_fn, artifacts)


def autograd_cache_key(
    graph,
    example_inputs,
    dynamic_shapes: DynamicShapesType,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
):
    from . import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(graph, dynamic_shapes, aot):
        return compile_fx.autograd_cache_key(
            graph,
            example_inputs,
            ignore_shape_env=ignore_shape_env,
        )
