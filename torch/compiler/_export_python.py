"""``torch.compiler.export_python``: a decorator whose compile cache is a file you own."""

import ast
import base64
import binascii
import contextlib
import functools
import logging
import os
import re
import textwrap
import threading
import zlib
from collections.abc import Callable
from typing import cast, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree
from torch._precompile import (
    _matching_cache,
    _parse_artifact_metadata,
    _runnable_from_pair,
    _source_literal,
    capture,
    DynamoTracer,
    PrecompiledRunnable,
    PrecompileError,
)
from torch.compiler._no_compile import is_compilation_forbidden


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def export_python(
    path: str | os.PathLike[str],
    *,
    backend: str = "inductor",
    dynamic: bool | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Compile a function once into a Python file, and run that file from then on.

    .. warning::

        This is a prototype API. Its signature, error types and file format may
        change between releases without a deprecation cycle.

    The function must trace as one graph with no graph breaks, as under
    ``torch.compile(fullgraph=True)``, and must not need a backward graph: export an
    inference call (under :func:`torch.no_grad` when a tensor it takes requires
    grad). ``path`` is the whole cache key. If it does not exist, the first call is
    captured with :func:`torch.compiler.precompile.capture` and a
    :class:`~torch.compiler.precompile.DynamoTracer`, returns its real result, and
    writes ``path`` and its acceleration cache ``path + ".cache"``. Every later call,
    in this process or any other, is served from ``path``: it runs where the captured
    guards hold, without tracing or compiling, and a call they do not cover raises
    :class:`~torch.compiler.PrecompileError` rather than recompiling. Nothing hashes
    the function, its config or the machine, so editing the function does not
    recapture it: delete ``path`` to recapture. Concurrent first calls in one process
    capture once; processes that share ``path`` (the ranks of one job, say) should
    let one of them make the first call::

        @torch.compiler.export_python(path="artifacts/step.py")
        def step(x, *, scale=2.0):
            return (x * scale).relu()


        y = step(x, scale=3.0)  # first run: captures and writes artifacts/step.py
        y = step(x, scale=3.0)  # served from the file, here or in a new process

    ``path`` is the graph's Inductor code, and that code is what runs; the rest of
    what Dynamo captured (the guards, the frame's bytecode and AOTAutograd's wrapper)
    is one compressed line at its end. An edited file is served without its cache,
    which was built for the captured code, so its edited kernels compile;
    :func:`~torch.compiler.precompile.no_compilation` refuses that. The stale cache
    stays until a recapture replaces it. A process serves one version of the file:
    run a new process to pick up an edit, since loading an edited file where an
    earlier version of it is loaded raises.

    With the inductor backend the capture tunes its Triton kernels at compile time (it
    compiles as inductor does under ``triton.autotune_at_compile_time``), so the graph
    code carries a ``KERNEL_CONFIGS`` table of the config each kernel inductor
    generated launches with: editing a row changes that kernel's launch config. A
    template (flex attention, say) or a user-defined kernel keeps its config in its
    own decorator instead, editable the same way.

    The file is trusted executable Python, locked to the PyTorch build and Python
    version that produced it. Decorate a function defined in an importable module:
    serving the file imports the module the function was captured from.

    Args:
        path: Where the file lives. Its parent directories are created on capture.
        backend: ``"inductor"`` (default) or ``"eager"``.
        dynamic: Forwarded to :class:`~torch.compiler.precompile.DynamoTracer`, as
            ``torch.compile(dynamic=)``.
    """

    torch._C._log_api_usage_once("torch.compiler.export_python")

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        exported = _ExportedPython(fn, path, backend, dynamic)

        @functools.wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return cast(_R, exported(*args, **kwargs))

        return wrapped

    return decorator


# The file is a short header, the graph's Inductor code between these two lines, and
# one line holding the rest of the precompile artifact the capture wrote, compressed,
# with a placeholder where that artifact carried the graph code.
_CODE_BEGIN = "# " + "-" * 75 + " graph code"
_CODE_END = "# " + "-" * 71 + " end graph code"
_GRAPH_CODE = "__export_python_graph_code__"
_CAPTURED = re.compile(r"^_CAPTURED = '([A-Za-z0-9+/=]*)'$", re.MULTILINE)


def _comment(text: str) -> list[str]:
    return textwrap.wrap(text, 86, initial_indent="# ", subsequent_indent="# ")


def _exported_file(artifact: str) -> str:
    """The file export_python writes for the precompile ``artifact`` of one graph."""
    meta = _parse_artifact_metadata(artifact)
    sources = [
        node.value
        for node in ast.parse(artifact).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and getattr(node.targets[0].value, "id", None) == "GRAPH_SOURCES"
    ]
    tracer, mode = meta.get("TRACER"), meta.get("SERVING_MODE")
    if tracer != "dynamo" or mode != "standalone" or len(sources) > 1:
        raise PrecompileError(
            f"export_python: this capture is not one graph from the Dynamo tracer "
            f"(TRACER={tracer!r}, SERVING_MODE={mode!r}, {len(sources)} graphs)."
        )
    name = f"{meta['ENTRY_MODULE']}.{meta['FN_NAME']}"
    py = ".".join(map(str, cast(tuple[int, ...], meta["_DYNAMO_PYTHON_VERSION"])))
    lines = _comment(
        f"Exported by torch.compiler.export_python from {name}, one graph, captured "
        f"with torch {meta['TORCH_VERSION']} and Python {py}. The decorator serves "
        "this file; delete it to recapture."
    )
    if pinned := cast(tuple[str, ...], meta["WONT_GENERALIZE"]):
        lines += _comment(
            f"Served only with the captured values of: {', '.join(pinned)}."
        )
    groups = ("DROPPED_GUARDS", "RISKY_DROPPED_GUARDS", "POLICY_DROPPED_GUARDS")
    unchecked = [
        n for group in groups for _, n in cast(list[list[str]], meta.get(group, []))
    ]
    if unchecked:
        lines += _comment(f"Guards not checked when served: {', '.join(unchecked)}.")
    if sources:
        segment = ast.get_source_segment(artifact, sources[0])
        if segment is None or artifact.count(segment) != 1:
            raise PrecompileError(
                "export_python: the artifact's graph code literal is not unique"
            )
        code = ast.literal_eval(segment)
        about = (
            "The graph code below is Inductor's output and is what runs: edit it, then "
            "rerun in a new process."
        )
        if 'if __name__ == "__main__":' in code:
            about += " Running this file benchmarks it on random inputs."
        lines += ["#"] + _comment(about)
        artifact = artifact.replace(segment, _GRAPH_CODE)
        lines += ["", _CODE_BEGIN, code, _CODE_END]
    blob = base64.b64encode(zlib.compress(artifact.encode(), 9)).decode()
    lines += [""] + _comment(
        "The rest of what Dynamo captured (the guards, the frame's bytecode and "
        "AOTAutograd's wrapper around call()), compressed. Not meant to be edited."
    )
    return "\n".join([*lines, f"_CAPTURED = {blob!r}", ""])


def _captured_artifact(text: str, path: str) -> str:
    """The precompile artifact export_python's file ``text`` was written from, with the
    graph code the file holds now."""
    match = _CAPTURED.search(text)
    if match is None:
        raise PrecompileError(
            f"export_python: {path} has no _CAPTURED line, so it is not a file "
            "export_python wrote. Delete it to recapture."
        )
    try:
        artifact = zlib.decompress(base64.b64decode(match.group(1))).decode()
    except (binascii.Error, zlib.error, UnicodeDecodeError) as e:
        raise PrecompileError(
            f"export_python: the _CAPTURED line of {path} is corrupt ({e}). Delete "
            "it to recapture."
        ) from e
    if _GRAPH_CODE not in artifact:
        return artifact
    begin, end = text.find(_CODE_BEGIN + "\n"), text.rfind("\n" + _CODE_END)
    if begin < 0 or end < begin:
        raise PrecompileError(
            f"export_python: {path} lost the lines that delimit its graph code "
            f"({_CODE_BEGIN!r} and {_CODE_END!r}). Restore them, or delete the file "
            "to recapture."
        )
    code = text[begin + len(_CODE_BEGIN) + 1 : end]
    return artifact.replace(_GRAPH_CODE, _source_literal(code), 1)


class _ExportedPython:
    def __init__(
        self,
        fn: Callable[..., object],
        path: str | os.PathLike[str],
        backend: str,
        dynamic: bool | None,
    ) -> None:
        if getattr(fn, "__module__", None) == "__main__":
            raise PrecompileError(
                f"export_python: {getattr(fn, '__qualname__', fn)!r} is defined in the "
                "__main__ script, which a later process cannot import to serve its "
                "file. Define it in an importable module."
            )
        self._fn = fn
        self._name = getattr(fn, "__qualname__", repr(fn))
        self._path = os.fspath(path)
        self._cache_path = self._path + ".cache"
        self._backend = backend
        self._tracer = DynamoTracer(dynamic=dynamic, fullgraph=True)
        self._runnable: PrecompiledRunnable | None = None
        self._lock = threading.Lock()

    def __call__(self, *args: object, **kwargs: object) -> object:
        if self._runnable is None:
            with self._lock:
                if self._runnable is None:
                    if not os.path.exists(self._path):
                        return self._capture(*args, **kwargs)
                    self._runnable = self._load()
        return self._runnable(*args, **kwargs)

    def _load(self, *, trusted: bool = False) -> PrecompiledRunnable:
        with open(self._path, "rb") as f:
            python_code = _captured_artifact(f.read().decode(), self._path)
        meta = _parse_artifact_metadata(python_code)
        captured = f"{meta['ENTRY_MODULE']}.{meta['FN_NAME']}"
        decorated = f"{self._fn.__module__}.{self._name}"
        if captured != decorated:
            raise PrecompileError(
                f"export_python: {self._path} was captured from {captured}, not "
                f"{decorated}: two functions cannot share a path. Give {decorated} "
                "a path of its own."
            )
        cache = _matching_cache(python_code, self._cache_path)
        if cache is None:
            if os.path.exists(self._cache_path):
                stale = (
                    f"{self._path} was edited after capture, so its cache "
                    f"{self._cache_path} does not match it"
                )
            else:
                stale = f"{self._path} has no cache at {self._cache_path}"
            # The artifact's own backend: the decorator's may have changed since.
            if is_compilation_forbidden() and meta["BACKEND"] == "inductor":
                raise PrecompileError(
                    f"export_python: {stale}, and no_compilation() serves an inductor "
                    "artifact only with its own cache. Serve it outside "
                    f"no_compilation(), or delete {self._path} to recapture."
                )
            # An edit (or a missing cache) leaves nothing to accelerate: the edited
            # graph compiles the kernels its code defines.
            log.info("export_python: %s; serving it without the cache", stale)
        return _runnable_from_pair(python_code, cache, _trusted=trusted)

    def _capture(self, *args: object, **kwargs: object) -> object:
        from torch._dynamo.exc import Unsupported

        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        # The precompile artifact the file is made from; its cache is the file's.
        artifact = f"{self._path}.capture"
        written = f"{self._path}.{os.getpid()}.tmp"
        cap = capture(
            self._fn,
            artifact_path=artifact,
            cache_path=self._cache_path,
            tracer=self._tracer,
            backend=self._backend,
        )
        try:
            with contextlib.ExitStack() as stack:
                if self._backend == "inductor":
                    from torch._inductor import config as inductor_config

                    # Tuning at compile time writes the graph's KERNEL_CONFIGS table:
                    # the config each inductor-generated kernel tuned to and launches
                    # with.
                    stack.enter_context(
                        inductor_config.patch({"triton.autotune_at_compile_time": True})
                    )
                stack.enter_context(cap)
                try:
                    result = cap(*args, **kwargs)
                except Unsupported as e:
                    raise PrecompileError(
                        f"export_python: {self._name} does not trace as one graph, "
                        "which export_python requires (no graph breaks, as under "
                        f"torch.compile(fullgraph=True)): {e}"
                    ) from e
                leaves = pytree.tree_leaves(result)
                if any(isinstance(t, torch.Tensor) and t.requires_grad for t in leaves):
                    raise PrecompileError(
                        f"export_python: {self._name} returned a tensor that requires "
                        "grad, so serving it would take a backward graph too, and "
                        "export_python exports one graph. Export an inference call: "
                        "make it under torch.no_grad()."
                    )
            with open(artifact, "rb") as f:
                text = _exported_file(f.read().decode())
            with open(written, "wb") as f:
                f.write(text.encode())
            os.replace(written, self._path)
        except BaseException:
            # Nothing serves without the file, so no half of it is left behind.
            for leftover in (self._cache_path, written):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(leftover)
            raise
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(artifact)
        # This process just wrote the file, so it skips the exec warning.
        self._runnable = self._load(trusted=True)
        return result


# Public as torch.compiler.export_python; rehomed like torch.compiler.PrecompileError.
export_python.__module__ = "torch.compiler"
