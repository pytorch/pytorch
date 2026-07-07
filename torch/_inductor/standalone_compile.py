from __future__ import annotations

import contextlib
import copy
import logging
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
from torch.fx.graph_module import _share_torchbind_and_process_group_on_deepcopy

from . import config
from ._functionalize_collectives import (
    _functionalize_inplace_collectives,
    _unbox_process_group_torchbinds,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.compiler._cache import CacheInfo
    from torch.fx import GraphModule


log = logging.getLogger(__name__)


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

        if format != "binary":
            raise AssertionError(f"expected format == 'binary', got {format}")
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header == AOTCompiledArtifact.AOT_HEADER:
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                artifact = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")
                return AOTCompiledArtifact.deserialize(artifact)
            # Otherwise, it's in the CacheCompiledArtifact format
            elif header == CacheCompiledArtifact.CACHE_HEADER:
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")
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

    def _to_binary_bytes(self) -> bytes:
        """Serialize this artifact to the in-memory ``binary`` byte format.

        Produces exactly the bytes that ``save(format="binary")`` writes to disk, so
        callers that only need the bytes (rather than a file on disk) can reuse this
        without going through ``save``. The byte layout is the one
        ``load(format="binary")`` reads back: header, ``torch_key``, the autograd-cache
        ``key`` string, then the opaque ``artifact_bytes``.
        """
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

        from torch.utils._appending_byte_serializer import BytesWriter

        from .codecache import torch_key

        writer = BytesWriter()
        writer.write_bytes(CacheCompiledArtifact.CACHE_HEADER)
        writer.write_bytes(torch_key())
        writer.write_str(key)
        writer.write_bytes(artifact_bytes)
        return writer.to_bytes()

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        with dynamo_timed("CompiledArtifact.save"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                if os.path.isdir(path):
                    raise AssertionError(f"expected path to not be a dir: {path}")

                from torch._inductor.codecache import write_atomic

                # ``_to_binary_bytes`` is the single source of the None / empty /
                # multiple aot_autograd_artifacts validation, so the binary branch
                # does not repeat those checks here.
                write_atomic(path, self._to_binary_bytes())
            else:
                if format != "unpacked":
                    raise AssertionError(f"expected format == 'unpacked', got {format}")
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
                if os.path.exists(path):
                    if not os.path.isdir(path):
                        raise AssertionError(f"expected path to be a dir: {path}")
                    shutil.rmtree(path, ignore_errors=True)

                from .codecache import FxGraphCache

                with temporary_cache_dir(path):
                    # This function unpacks the cache artifacts to disk
                    loaded_cache_info = torch.compiler.load_cache_artifacts(
                        artifact_bytes
                    )
                    if loaded_cache_info is None:
                        raise AssertionError(
                            "expected loaded_cache_info to not be None"
                        )
                    # Now write all the output_code artifacts to disk so that
                    # they can be inspected and modified
                    for key in loaded_cache_info.inductor_artifacts:
                        subdir = FxGraphCache._get_tmp_dir_for_key(key)
                        if not os.path.exists(subdir):
                            raise AssertionError(f"expected subdir to exist: {subdir}")
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

            if result is None:
                raise AssertionError(
                    "expected AOTAutogradCache lookup result to not be None"
                )
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
                if os.path.isdir(path):
                    raise AssertionError(f"expected path to not be a dir: {path}")
                with open(path, "rb") as file:
                    artifacts = file.read()
                from torch.utils._appending_byte_serializer import BytesReader

                from .codecache import torch_key

                reader = BytesReader(artifacts)
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")

                torch.compiler.load_cache_artifacts(artifact_bytes)
                return key, nullcontext()
            else:
                if format != "unpacked":
                    raise AssertionError(f"expected format == 'unpacked', got {format}")
                if not os.path.isdir(path):
                    raise AssertionError(f"expected path to be a dir: {path}")
                autograd_cache_dir = os.path.join(path, "aotautograd")
                if not os.path.isdir(autograd_cache_dir):
                    raise AssertionError(
                        f"expected autograd_cache_dir to be a dir: {autograd_cache_dir}"
                    )
                files = list(os.listdir(autograd_cache_dir))
                if len(files) != 1:
                    raise AssertionError(f"expected exactly 1 file, got {len(files)}")
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
        if not isinstance(deserialized, BundledAOTAutogradSerializableCallable):
            raise AssertionError(
                f"expected BundledAOTAutogradSerializableCallable, got {type(deserialized)}"
            )
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
            if header != AOTCompiledArtifact.AOT_HEADER:
                raise AssertionError("expected AOTCompiledArtifact header")
            if reader.read_bytes() != torch_key():
                raise AssertionError("torch_key mismatch in serialized artifact")
            artifact = reader.read_bytes()
            if not reader.is_finished():
                raise AssertionError("expected reader to be finished")
            return AOTCompiledArtifact.deserialize(artifact)


def _resolve_ignore_shape_env(dynamic_shapes: DynamicShapesType):
    # tells compile_fx to ignore the shape_envs on the ambient context
    # and the graph_module.
    return dynamic_shapes == "from_example_inputs"


def _resolve_fake_mode(
    gm: GraphModule,
    dynamic_shapes: DynamicShapesType,
    fake_mode: FakeTensorMode | None = None,
) -> FakeTensorMode:
    if dynamic_shapes == "from_example_inputs":
        if fake_mode is not None:
            if fake_mode.shape_env is None:
                raise ValueError(
                    "standalone_compile requires `fake_mode` to have a ShapeEnv "
                    'when `dynamic_shapes="from_example_inputs"`.'
                )
            return fake_mode
        return FakeTensorMode(shape_env=ShapeEnv())
    elif fake_mode is not None:
        raise ValueError(
            "standalone_compile only supports passing `fake_mode` when "
            '`dynamic_shapes="from_example_inputs"`.'
        )
    elif dynamic_shapes == "from_tracing_context":
        # Reuse fake_mode from the TracingContext.
        # NB: The TracingContext only exists if we're currently in a torch.compile backend.
        context = torch._guards.TracingContext.get()
        if context.fake_mode is None:
            raise AssertionError("expected TracingContext.fake_mode to not be None")
        return context.fake_mode
    elif dynamic_shapes == "from_graph":
        # Strategy: find a FakeTensor in the graph output, grab its FakeTensorMode.
        # The graph passed to standalone_compile must be an Inductor-approved graph,
        # which means that there is at least one Tensor output and the output node
        # contains a flat list of Tensors.
        last_node = next(iter(reversed(gm.graph.nodes)))
        if last_node.op != "output":
            raise AssertionError(
                f"expected last node op == 'output', got {last_node.op}"
            )
        if len(last_node.args) != 1:
            raise AssertionError(
                f"expected last node to have 1 arg, got {len(last_node.args)}"
            )

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
def _standalone_context(
    gm: GraphModule,
    dynamic_shapes: DynamicShapesType,
    aot: bool,
    fake_mode: FakeTensorMode | None = None,
):
    from torch.compiler._cache import CacheArtifactManager

    resolved_fake_mode = _resolve_fake_mode(gm, dynamic_shapes, fake_mode)
    tracing_context = torch._guards.TracingContext(resolved_fake_mode)
    with (
        torch._guards.tracing(tracing_context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
        torch._functorch.config.patch(
            {
                "bundled_autograd_cache": aot,
                # Standalone artifacts are saved immediately after compile_fx
                # returns. Training graphs normally lower the backward lazily on
                # first backward(), so force it while the artifact recorder is
                # still active.
                "force_non_lazy_backward_lowering": True,
            }
        ),
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
    fake_mode: FakeTensorMode | None = None,
) -> CompiledArtifact:
    """
    Implementation of torch.inductor.standalone_compile
    """
    from .compile_fx import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(gm, dynamic_shapes, aot, fake_mode):
        # compile_fx takes ownership of gm and may mutate it on cache miss.
        # Deepcopy first so the rewrites below land on the owned copy rather
        # than the caller's gm. The gm may carry a non-pickleable torchbind
        # ProcessGroup (or, after a previous unbox, a Python
        # ``dist.ProcessGroup``); smuggle it through deepcopy as a shared
        # reference instead of crashing.
        if not donate_graph_module:
            with _share_torchbind_and_process_group_on_deepcopy():
                gm = copy.deepcopy(gm)
        # ``make_fx`` traces ``dist.*`` collectives as opaque ``c10d.{op}_``
        # calls. Inductor's collective machinery only recognizes the
        # ``_c10d_functional.{op}`` + ``wait_tensor`` form, so rewrite here
        # before compile_fx runs. Also unbox any torchbind ProcessGroup
        # attrs into Python ``dist.ProcessGroup`` so the runtime collective
        # op accepts them (raw torchbind is rejected).
        gm = _functionalize_inplace_collectives(gm)
        gm = _unbox_process_group_torchbinds(gm)
        compiled_fn = compile_fx(
            gm, example_inputs, ignore_shape_env=ignore_shape_env, **options
        )
        if not callable(compiled_fn):
            raise AssertionError("expected compiled_fn to be callable")
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


class NoRunnableInductorModuleError(RuntimeError):
    """Raised by ``compile_to_python`` when the graph yields no runnable Inductor
    output module -- it has no compute to lower (returns inputs/constants unchanged),
    so the compiled artifact carries no output-module source. Callers (e.g.
    torch.compiler.precompile) convert this to a clear user-facing error suggesting an
    alternative.
    """


def _runnable_source(artifact: CompiledArtifact) -> str:
    """Return the final Inductor output-module source for a compiled ``artifact``.

    Inductor stashes the wrapper-module source on the compiled callable as
    ``source_code`` (a field of the underlying ``CompiledFxGraph`` propagated onto the
    returned callable), populated identically whether the graph was freshly codegen'd
    or restored from a warm in-process / out-of-process cache. It is the FINAL selected
    module only: the throwaway max_autotune benchmark lowerings compile to their own
    modules during autotuning and never become the returned artifact, so no filtering
    is needed. A graph with no compute (returns inputs/constants unchanged) produces no
    such source, which we surface as ``NoRunnableInductorModuleError``.
    """
    source = getattr(artifact._compiled_fn, "source_code", None)
    if not source:
        raise NoRunnableInductorModuleError(
            "the compiled graph produced no runnable Inductor output module: it has no "
            "compute to lower (returns inputs/constants unchanged)."
        )
    return source


def _binary_cache_bytes(artifact: CompiledArtifact) -> bytes | None:
    """Serialize the artifact to opaque cache bytes, or None if it is not
    serializable (e.g. graphs with input mutations currently do not produce a
    saveable aot_autograd artifact). The source still runs standalone without it."""
    # The only EXPECTED no-cache case is an artifact with no saveable aot_autograd
    # entry (e.g. certain input-mutating graphs); detect it up front and skip the
    # save so a genuine ``save`` regression surfaces below at warning instead of
    # being silently downgraded to an "uncacheable" fallback (which would only show
    # up later as a missing FxGraphCache hit on reload).
    if not isinstance(artifact, CacheCompiledArtifact) or not artifact.is_saveable():
        log.debug("standalone artifact has no saveable cache entry; no cache")
        return None
    # Serialize fully in memory: these are exactly the bytes ``save(format="binary")``
    # would write, and ``CompiledArtifact.load`` reads them back via
    # ``load_cache_artifacts``, so the cache round-trips. Avoiding a temp file here
    # means a SIGKILL mid-compile leaves no stray ``.bin`` in the cache dir. The
    # expensive serialization already happened at compile time (``save_cache_artifacts``
    # captured the bytes and ``is_saveable`` gated the one expected no-cache case), so
    # this only re-frames known-good bytes: any exception here is a genuine bug and is
    # allowed to propagate rather than being silently downgraded to a cache miss.
    return artifact._to_binary_bytes()


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: DynamicShapesType = "from_example_inputs",
    options: dict[str, Any] | None = None,
) -> tuple[str, bytes | None]:
    """Compile ``gm`` and return ``(inner_python, cache)`` -- the INNER half of the
    backend contract behind ``torch.compiler.precompile``.

    This is an INTERNAL layered-contract entry point, not an end-user API. End users
    should call ``torch.compiler.precompile``; this function only emits the inductor
    piece of the artifact and assumes its caller (the AOT layer) wraps it. It lives in
    ``torch._inductor`` (a private, leading-underscore module), so it is exposed for
    the AOT layer to import, not as a stable public surface.

    ``inner_python`` is the Inductor output module exposing ``call(args) -> outs``
    for the post-AOTAutograd inner graph (dense, functionalized). It is the inductor
    piece only: it carries NO prelude/epilogue (subclass flatten/unflatten, input-
    mutation copy-back, output-alias regen, grad disabling). Those belong to the AOT
    layer -- a companion change in ``torch._functorch.aot_autograd`` wraps this and
    composes AOTAutograd's codegen'd runtime wrappers around the result.
    Callers must run ``call`` under ``torch.no_grad()`` (the kernels use out= ops).

    Caller preconditions (this layer does not re-derive them; passing a graph that
    violates them yields wrong/unrunnable output rather than a clean error):

    - ``gm`` is a post-AOTAutograd dense, functionalized inner graph (the dense
      forward/backward AOTAutograd hands to its inductor backend), NOT a raw Dynamo
      or pre-dispatch graph still carrying subclasses or autograd.
    - ``gm`` lowers to a runnable inductor output module. A graph with no compute
      raises ``NoRunnableInductorModuleError``.

    ``dynamic_shapes`` defaults to ``"from_example_inputs"`` here, which DIFFERS from
    the sibling ``torch._inductor.standalone_compile`` default of ``"from_graph"``.
    The reason: this entry point is fed the post-AOTAutograd inner graph, whose
    example-value metadata is the source of truth AOTAutograd already resolved, so
    specializing on ``example_inputs`` matches what the AOT layer expects; the public
    ``standalone_compile`` instead trusts the dynamic shapes baked into the passed-in
    graph. See ``DynamicShapesType`` for the full set of modes.

    ``options`` is an optional inductor config-override dict (``None`` means no
    overrides), the same shape as ``torch._inductor.compile``'s ``options``. The
    keys are inductor config names: they are merged into the ``config.patch``
    block this function already wraps the compile in, so they take effect as
    config rather than being forwarded as ``compile_fx`` kwargs. The two
    capture-critical pins below (``benchmark_harness``, ``cpp_wrapper``) always win
    over a conflicting user option, since the source-capture contract depends on
    them and a caller cannot be allowed to break the emitted module.

    The kernels JIT-compile from the inlined source on first call, so ``inner_python``
    needs no cache. ``cache`` is an opaque acceleration (or ``None`` when the graph
    is not serializable, e.g. some input-mutating graphs, or when caches are disabled --
    the bytes come from the AOTAutograd cache, so any of ``force_disable_caches``,
    ``fx_graph_cache=False``, or ``enable_autograd_cache=False`` yields ``None``).

    ``inner_python`` is read off the compiled artifact -- Inductor stashes the final
    wrapper-module source on the returned callable as ``source_code`` -- so it reflects
    the FINAL selected module and is valid on a cold compile, a warm in-process or
    out-of-process cache restore, or with caches disabled. No process-global codegen
    hook is involved, and the throwaway max_autotune benchmark lowerings (which compile
    to their own modules during autotuning) never reach the artifact, so nothing needs
    to be filtered out.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            f"compile_to_python expects a post-AOTAutograd torch.fx.GraphModule, "
            f"got {type(gm)}. This is an internal entry point wrapped by a higher AOT "
            f"layer and is not meant to be called directly."
        )
    # The experimental TORCHINDUCTOR_FX_COMPILE_MODE=async+/progressive+ schemes make
    # compile_fx return an _AsyncOutputCode/_ProgressiveOutputCode that carries only
    # _boxed_call and never surfaces the wrapper ``source_code`` this contract reads off.
    # These are module-level globals resolved from the env at import time, not
    # config-patchable like the pins below, so detect them up front and fail with a
    # distinct error instead of the no-compute NoRunnableInductorModuleError -- the graph
    # is runnable, the async scheme just did not surface its source.
    from torch._inductor import compile_fx as _compile_fx

    if _compile_fx.fx_compile_async or _compile_fx.fx_compile_progressive:
        raise RuntimeError(
            "compile_to_python needs synchronous source capture and does not yet "
            "support TORCHINDUCTOR_FX_COMPILE_MODE=async+/progressive+ (the async "
            "output code does not surface the wrapper source)."
        )
    # Treat ``options`` as inductor config overrides and fold them into the same
    # ``config.patch`` we wrap the compile in. The two output pins are applied AFTER the
    # user options so they override any conflicting key: benchmark_harness=False keeps
    # the emitted module runnable (no get_args()/benchmark_compiled_module()/__main__),
    # and cpp_wrapper=False keeps it a python wrapper (the C++ backend emits a C++
    # ``call`` we cannot inline). A caller must not be able to break either.
    config_patches: dict[str, Any] = dict(options) if options is not None else {}
    config_patches.update(
        {
            "benchmark_harness": False,
            "cpp_wrapper": False,
        }
    )
    # Compile under no_grad so AOTAutograd takes the inference (forward-only) path.
    # compile_fx re-enters AOTAutograd, and with grad-requiring inputs it would otherwise
    # build a JOINT graph and emit BOTH a forward and a backward output module, leaving
    # "the" runnable module ambiguous. The contract feeds one dense post-AOTAutograd inner
    # graph at a time (the AOT layer drives forward and backward as separate calls), so
    # pinning inference is the explicit choice: it yields exactly one module whose outputs
    # are this graph's own, and we never have to second-guess the strides or tensor
    # subclasses of a backward AOTAutograd would otherwise synthesize.
    with torch.no_grad(), config.patch(config_patches):
        # Go through ``standalone_compile`` (full ``compile_fx`` + AOTAutograd) rather
        # than driving ``_compile_fx_inner`` on the dense graph directly: the ``cache``
        # half of the contract is the bundled AOTAutograd-level artifact captured by
        # ``save_cache_artifacts()``, which only exists on the AOTAutograd path. The
        # ``no_grad`` pin above tames the re-entry (forces ``aot_dispatch_base``, one
        # forward module), so the second AOTAutograd pass over the already-dense graph is
        # idempotent for the tested shapes.
        # ``options`` are already applied above as config, so pass none down to
        # ``standalone_compile`` (it forwards options as ``compile_fx`` kwargs, which a
        # config key like ``max_autotune`` is not).
        artifact = standalone_compile(
            gm,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={},
        )
    inner_python = _runnable_source(artifact)
    cache = _binary_cache_bytes(artifact)
    return inner_python, cache


def autograd_cache_key(
    graph,
    example_inputs,
    dynamic_shapes: DynamicShapesType,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    fake_mode: FakeTensorMode | None = None,
):
    from . import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(graph, dynamic_shapes, aot, fake_mode):
        return compile_fx.autograd_cache_key(
            graph,
            example_inputs,
            ignore_shape_env=ignore_shape_env,
        )
