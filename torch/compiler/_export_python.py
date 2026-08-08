"""Path-cached ``torch.compiler.export_python`` decorator.

``torch.compiler.precompile`` captures a function ahead of time and lowers it to a
self-contained, human-readable Python source artifact (see
``torch/_precompile.py``). ``torch.compiler.export_python`` wraps that in a
decorator keyed off a file on disk: the first run writes the emitted
``python_code`` to ``path``; every later run reads the ``.py`` back and executes
it directly instead of recompiling.

Because the artifact is self-contained, re-executable Python, ``path`` is meant to
be committed and hand-edited: an engineer or agent can "hill-climb" the generated
kernel in place. This is ejectable compilation -- the emitted source is the source
of truth and is always exec'd, so hand edits always take effect. There is no
acceleration cache and no ``precompile.load`` round-trip: the source is exec'd as
written, so keeping the edited source correct is the caller's responsibility.
"""

import copy
import functools
import inspect
import logging
import operator
import os
import tempfile
import threading
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.utils._pytree as pytree


log = logging.getLogger(__name__)

# Written as the artifact's first line so a later load can detect it was produced
# by a different torch (see _warn_on_version_skew). It is a comment, so it does not
# affect exec; a hand-edit that drops it just disables the skew warning, so
# hill-climbing an artifact never triggers a spurious version warning.
_VERSION_TAG = "# torch.compiler.export_python torch-version: "


def _atomic_write(path: str, data: bytes) -> None:
    # Write to a temp file in the same directory, fsync it, then rename it into
    # place. os.replace is atomic, so an interrupted or concurrent writer never
    # leaves a half-written artifact that the presence-only load gate would read
    # as valid.
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates the temp file 0600 and os.replace preserves that mode,
        # but this artifact is meant to be committed and hand-edited, so it needs
        # conventional world-readable perms rather than the private tempfile mode.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# The buffer-donation contract, stated once here because out= is the one part of
# export_python that needs anything of the artifact beyond "it is runnable python":
#
#   An artifact supports out= iff it takes every buffer it allocates from a callable
#   bound at module level in its own namespace under a name starting with
#   _ALLOCATOR_PREFIX. Rebinding those names is then enough to hand the generated code
#   memory instead of letting it allocate (see _BufferDonationPool).
#
# Nothing here is keyed on the backend: an artifact that meets the contract donates,
# one that does not is rejected at the first out= call. The inductor backend meets it
# by binding one allocator per device type (empty_strided_cuda, empty_strided_cpu,
# empty_strided_cpu_pinned, ...) regardless of which the graph uses; the eager backend
# emits a bare graph call and binds none. Allocators are matched by prefix rather than
# by an explicit list because the pool replays allocations BY ORDER: missing one
# device's allocator would desync the whole plan rather than fail. Both halves of that
# are pinned by tests -- test_allocator_prefix_matches_codegen_call_sites checks every
# allocator inductor's codegen can emit, and test_pool_intercepts_every_artifact_allocator
# checks every one a real artifact binds.
_ALLOCATOR_PREFIX = "empty_strided"


def _precompile_error(msg: str) -> Exception:
    from torch._precompile import PrecompileError

    return PrecompileError(msg)


class _BufferDonationPool:
    """Serves the artifact's buffer allocations when the caller donates outputs.

    Takes the artifact as satisfying the donation contract above ``_ALLOCATOR_PREFIX``:
    every buffer comes from a module-level allocator in its own namespace, so rebinding
    those names is enough to hand it memory rather than let it allocate. The first
    donating call records the ordered requests and which of them come back as graph
    outputs; later donating calls replay that recording, serving scratch buffers from
    the pool and outputs from the caller, so a steady-state donating call allocates
    nothing.

    Calls that do not donate go straight through to the real allocator, so adding an
    ``out=`` call site never changes what the plain call path does.
    """

    def __init__(self, ns: dict[str, Any]) -> None:
        allocators = [
            n for n, v in ns.items() if n.startswith(_ALLOCATOR_PREFIX) and callable(v)
        ]
        if not allocators:
            raise _precompile_error(
                "torch.compiler.export_python: out= needs the artifact to allocate its "
                f"buffers through {_ALLOCATOR_PREFIX}* callables bound in its own "
                "namespace, and this one binds none (backend='eager' emits a bare graph "
                "call). Use backend='inductor', or drop out=."
            )
        # Every device's allocator is wrapped, each keeping its OWN real function: the
        # preamble binds all of them regardless of which the graph uses, and falling
        # back through the wrong one would hand a CPU kernel device memory.
        for name in allocators:
            ns[name] = functools.partial(self._alloc, ns[name])
        self.recorded = False
        # (buffer, -1) for a pooled scratch slot, (None, position) for one the caller
        # donates. Replayed in allocation order, which the specialized graph fixes.
        self._plan: list[tuple[torch.Tensor | None, int]] = []
        self._slots: list[torch.Tensor] = []
        self._recording = False
        self._donated: Sequence[torch.Tensor] | None = None
        self._i = 0

    def _alloc(
        self, real: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if self._recording:
            buf = real(*args, **kwargs)
            self._slots.append(buf)
            return buf
        if self._donated is None:
            return real(*args, **kwargs)
        i = self._i
        self._i = i + 1
        try:
            buf, pos = self._plan[i]
        except IndexError:
            # The graph is shape-specialized, so the recorded request sequence is
            # fixed; anything past it is not something to serve from the plan.
            return real(*args, **kwargs)
        return self._donated[pos] if pos >= 0 else buf  # type: ignore[return-value]

    def record(
        self, call: Callable[[list[Any]], Any], args: list[Any], out: Sequence[Any]
    ) -> list[Any]:
        """Run one recording call, then fix the plan and validate the donated tensors."""
        self._recording = True
        try:
            outs = list(call(args))
        finally:
            self._recording = False
        if len(out) != len(outs):
            raise _precompile_error(
                f"torch.compiler.export_python: out= has {len(out)} tensors but the "
                f"artifact returns {len(outs)}."
            )
        slot_of = {id(t): i for i, t in enumerate(self._slots)}
        out_slot: dict[int, int] = {}
        for pos, produced in enumerate(outs):
            i = slot_of.get(id(produced))
            if i is None:
                raise _precompile_error(
                    f"torch.compiler.export_python: output {pos} is not a buffer the "
                    "artifact allocates (it is a view of one, or an input passed "
                    "through), so it cannot be donated. Drop out= for this artifact."
                )
            out_slot[i] = pos
            _check_donor(out[pos], produced, pos)
        # The plan holds the scratch buffers from here on, and holds no reference to
        # the output slots (the caller owns those), so the recording list goes away.
        self._plan = [
            (None, out_slot[i]) if i in out_slot else (buf, -1)
            for i, buf in enumerate(self._slots)
        ]
        self._slots = []
        self.recorded = True
        # This one call allocated its own outputs, so the donated tensors have to be
        # filled by hand. Every later donating call writes into them directly.
        for donor, produced in zip(out, outs):
            donor.copy_(produced)
        return list(out)

    def begin(self, donated: Sequence[torch.Tensor]) -> None:
        self._i = 0
        self._donated = donated

    def end(self) -> None:
        self._donated = None

    def check_complete(self) -> None:
        # The plan is replayed by allocation ORDER, so a call that asked for a
        # different number of buffers than was recorded served the wrong memory. That
        # cannot happen for a shape-specialized graph, but if it ever does it must be
        # loud on the first donating call rather than quietly wrong forever. Checked
        # only after a call that returned, so it never masks a real exception.
        if self._i != len(self._plan):
            raise _precompile_error(
                "torch.compiler.export_python: the donated call requested "
                f"{self._i} buffers but {len(self._plan)} were recorded; the buffer "
                "donation plan is out of sync. Drop out= for this artifact."
            )


def _check_donor(donor: Any, produced: torch.Tensor, pos: int) -> None:
    # Checked once, on the recording call. Afterwards a donated tensor is written to
    # by a kernel with baked sizes and strides, so a later mismatch is undefined
    # behavior -- the same contract as the rest of unsafe_reduce_overhead.
    if (
        not isinstance(donor, torch.Tensor)
        or donor.shape != produced.shape
        or donor.stride() != produced.stride()
        or donor.dtype != produced.dtype
        or donor.device != produced.device
    ):
        raise _precompile_error(
            f"torch.compiler.export_python: out[{pos}] must be a tensor matching the "
            f"artifact's output exactly (shape {tuple(produced.shape)}, stride "
            f"{produced.stride()}, {produced.dtype}, {produced.device}); got {donor!r}."
        )


def _lean_entry(forward: Callable[..., Any]) -> Callable[..., Any] | None:
    """Bind the artifact's compiled ``call`` directly, skipping the driver's guards.

    The emitted ``forward`` spends most of its time re-verifying the precompile
    invariants on every call (pytree round-trip, per-input shape/dtype/device guards,
    module structure). Everything it guards is fixed at capture, so a caller that
    accepts the invariants can call the compiled ``call`` directly. Returns None when
    the driver is doing real marshalling rather than checking -- an ``nn.Module``
    argument to lift params out of, a gradient to scatter, or an input/output
    structure that is not a flat sequence of leaves -- in which case there is nothing
    safe to strip and the caller keeps ``forward``.
    """
    # forward was exec'd into the artifact's own module namespace, so its __globals__
    # IS that namespace: the composed ``call`` and the baked calling convention are
    # reachable through it without re-exec'ing or re-parsing the source.
    ns = forward.__globals__
    if ns.get("MODULE_POSITIONS") or ns.get("GRAD_PARAM_INDICES"):
        return None
    try:
        # ``call`` is AOTAutograd's composed entry point, NOT the raw inductor one:
        # it still reflects input mutation, unwraps subclasses and disables grad. Only
        # the precompile driver's guards are dropped, never AOTAutograd's semantics.
        call = ns["call"]
        in_spec_str, out_spec_str = ns["IN_SPEC"], ns["OUT_SPEC"]
    except KeyError:
        return None
    if in_spec_str is None or out_spec_str is None:
        return None
    in_spec = pytree.treespec_loads(in_spec_str)
    if in_spec.type is not tuple or not all(c.is_leaf() for c in in_spec.children()):
        return None
    out_spec = pytree.treespec_loads(out_spec_str)
    if out_spec.is_leaf():
        rebuild: Callable[[list[Any]], Any] = operator.itemgetter(0)
    elif out_spec.type in (tuple, list) and all(
        c.is_leaf() for c in out_spec.children()
    ):
        rebuild = out_spec.type
    else:
        return None
    num_args = in_spec.num_children
    set_grad, grad_enabled = torch._C._set_grad_enabled, torch.is_grad_enabled
    pool: _BufferDonationPool | None = None

    def lean_forward(*args: Any, out: Sequence[torch.Tensor] | None = None) -> Any:
        nonlocal pool
        # Arity is checked because it costs ~30ns and the alternative is an unpack
        # ValueError raised from generated source. Nothing else is: shapes, dtypes,
        # devices and layouts are the caller's responsibility in this mode.
        if len(args) != num_args:
            raise _precompile_error(
                f"precompile: expected {num_args} positional args (the same as the "
                f"traced fn), got {len(args)}."
            )
        if out is not None and pool is None:
            pool = _BufferDonationPool(ns)
        # forward runs the call under no_grad and an eager-backend ``call`` is a bare
        # graph call, so grad has to be disabled here too -- but with the C setter, as
        # the torch.no_grad context manager costs more than everything else here.
        grad = grad_enabled()
        if grad:
            set_grad(False)
        try:
            if out is None:
                return rebuild(call(list(args)))
            if not pool.recorded:  # type: ignore[union-attr]
                return rebuild(pool.record(call, list(args), out))  # type: ignore[union-attr]
            pool.begin(out)  # type: ignore[union-attr]
            try:
                result = rebuild(call(list(args)))
            finally:
                pool.end()  # type: ignore[union-attr]
            pool.check_complete()  # type: ignore[union-attr]
            return result
        finally:
            if grad:
                set_grad(True)

    return lean_forward


class ExportedPythonArtifact:
    """Materializes and disk-caches a ``torch.compiler.precompile`` artifact.

    Materialization is lazy and happens on the first call: if ``path`` exists the
    emitted Python is read from disk, otherwise the wrapped ``fn`` is precompiled
    against the example inputs and the emitted source is written to disk. Either
    way the source is exec'd directly to build the runnable. The loaded callable is
    reused for all subsequent calls in the process; a later process re-reads
    whatever is on disk.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        path: str,
        backend: str,
        tracer: str,
        decompositions: dict | None,
        example_inputs: Sequence[object] | None,
        unsafe_reduce_overhead: bool = False,
    ) -> None:
        self._fn = fn
        self._signature = inspect.signature(fn)
        self._path = path
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._example_inputs = None if example_inputs is None else tuple(example_inputs)
        self._unsafe_reduce_overhead = unsafe_reduce_overhead
        # Whether the lean entry point (the one that accepts out=) is what got bound;
        # unsafe_reduce_overhead can fall back to the checked forward at load time.
        self._lean_bound = False
        self._loaded: Callable[..., Any] | None = None
        self._lock = threading.Lock()
        functools.update_wrapper(self, fn)

    def _precompile_and_save(self, args: tuple[Any, ...]) -> str:
        example = self._example_inputs
        if example is None:
            # Capture runs fn once on the example inputs (real-mode make_fx), which
            # mutates them; deep-copy the live call args so capture side effects (in-
            # place input mutation, module buffer updates) do not leak onto the
            # caller before the artifact itself runs on the real args exactly once.
            try:
                example = copy.deepcopy(args)
            except Exception as e:
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    "torch.compiler.export_python could not deep-copy the "
                    "first-call arguments to capture without mutating them (e.g. a "
                    "non-leaf tensor or a weight_norm module). Pass explicit "
                    "example_inputs=... to precompile against dedicated inputs."
                ) from e
        # precompile returns (python_code, cache); the cache is an acceleration
        # artifact that export_python does not use -- the emitted source is
        # self-contained and always exec'd -- so only the code is written to disk.
        code, _cache = torch.compiler.precompile(
            self._fn,
            *example,
            backend=self._backend,
            tracer=self._tracer,
            decompositions=self._decompositions,
        )
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Stamp the producing torch version as the artifact's first line so a load
        # by a different torch can warn about the skew. It is a comment (exec-inert)
        # and a hand-edit may freely drop it (see _warn_on_version_skew).
        code = f"{_VERSION_TAG}{torch.__version__}\n{code}"
        # The write is atomic, so two first-call processes that both pass the
        # presence gate each write a complete, self-contained artifact and the last
        # rename wins; neither leaves a half-written file behind the gate.
        _atomic_write(self._path, code.encode("utf-8"))
        return code

    def _load_from_disk(self) -> str:
        with open(self._path, encoding="utf-8") as f:
            return f.read()

    def _warn_on_version_skew(self, code: str) -> None:
        # Warn (but still run) when the artifact carries a version stamp that does
        # not match the current torch, so a committed artifact gone stale across a
        # torch upgrade is visible rather than silently running old logic. A missing
        # stamp (dropped by a hand-edit) is silent, so hill-climbing never warns.
        first_line = code.split("\n", 1)[0]
        if not first_line.startswith(_VERSION_TAG):
            return
        produced = first_line[len(_VERSION_TAG) :].strip()
        if produced != torch.__version__:
            log.warning(
                "torch.compiler.export_python: the artifact at %s was produced by "
                "torch %s but the current torch is %s; running it as-is. Delete %s "
                "to regenerate against the current torch.",
                self._path,
                produced,
                torch.__version__,
                self._path,
            )

    def _load(self, code: str) -> Callable[..., Any]:
        # The emitted source is self-contained: exec it directly (no cache, no
        # precompile.load round-trip) and without the untrusted-exec warning, since
        # export_python only ever loads artifacts it produced. A clobbered hand-edit
        # (dropped forward / syntax error) and an environment or version mismatch (an
        # import that fails under the current torch) surface as distinct, actionable
        # PrecompileErrors rather than one catch-all "delete to regenerate".
        from torch._precompile import _make_inlined_forward, PrecompileError

        try:
            return _make_inlined_forward(code, warn=False)
        except (SyntaxError, KeyError) as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} could "
                "not be run as precompile source; it is not a valid "
                "torch.compiler.precompile artifact (a hand-edit may have clobbered "
                "it, e.g. dropping forward()). Delete it to regenerate."
            ) from e
        except ImportError as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} failed "
                "to import a dependency; it was likely produced by a different torch "
                f"version or environment. Delete {self._path} to regenerate against "
                "the current torch."
            ) from e
        except Exception as e:
            raise PrecompileError(
                "torch.compiler.export_python: an unexpected error occurred running "
                f"the artifact at {self._path}. Delete it to regenerate."
            ) from e

    def _materialize(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        if os.path.exists(self._path):
            code = self._load_from_disk()
            self._warn_on_version_skew(code)
        else:
            code = self._precompile_and_save(args)
        forward = self._load(code)
        if not self._unsafe_reduce_overhead:
            return forward
        # A fallback is a warning rather than an error: the artifact still runs, the
        # flag just bought nothing, and failing a working call over a missed
        # optimization would be worse than saying so.
        lean = _lean_entry(forward)
        if lean is None:
            log.warning(
                "torch.compiler.export_python: unsafe_reduce_overhead=True had no "
                "effect for %s; its calling convention needs the artifact's driver (an "
                "nn.Module argument, a gradient to scatter, or a non-flat input/output "
                "structure). Running the checked entry point.",
                self._path,
            )
            return forward
        self._lean_bound = True
        return lean

    def _bind_positional(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, ...]:
        # The artifact's forward is positional (the precompile calling convention),
        # so map any keyword call args onto fn's positional parameters -- this lets
        # callers invoke the decorated fn naturally (e.g. rope(q=..., k=...)).
        # Anything that cannot be laid out positionally is rejected below.
        sig = self._signature
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                "torch.compiler.export_python: could not bind the call arguments to "
                f"{getattr(self._fn, '__name__', 'fn')}'s signature: {e}"
            ) from e
        # bound.kwargs holds every argument bind() could not place positionally. That
        # is a keyword-only / **kwargs param (never positional), or a plain
        # positional-or-keyword param passed by keyword while an earlier one was left
        # to its default -- distinguish them so the error names the real cause.
        if bound.kwargs:
            params = sig.parameters
            kw_only = sorted(
                n
                for n in bound.kwargs
                if n in params and params[n].kind == inspect.Parameter.KEYWORD_ONLY
            )
            if kw_only:
                raise TypeError(
                    "torch.compiler.export_python does not support keyword-only "
                    f"parameters (got {kw_only}); the precompile calling convention "
                    "is positional."
                )
            # Names not declared as parameters were absorbed by a **kwargs param;
            # they are never positional, so name **kwargs as the cause rather than
            # misreporting them as a positional-or-keyword arg left to its default.
            var_kw = sorted(n for n in bound.kwargs if n not in params)
            if var_kw:
                raise TypeError(
                    "torch.compiler.export_python does not support **kwargs "
                    f"parameters (got {var_kw}); the precompile calling convention "
                    "is positional."
                )
            raise TypeError(
                "torch.compiler.export_python could not place keyword arguments "
                f"{sorted(bound.kwargs)} positionally because an earlier positional "
                "parameter was left to its default; pass those arguments positionally "
                "or provide example_inputs."
            )
        return bound.args

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # out= is intercepted before signature binding: it is export_python's own
        # calling convention, not one of fn's parameters, so binding it onto fn would
        # report it as an unexpected keyword.
        out = kwargs.pop("out", None)
        if out is not None and not self._unsafe_reduce_overhead:
            raise TypeError(
                "torch.compiler.export_python: out= requires "
                "unsafe_reduce_overhead=True; the checked entry point allocates its "
                "own outputs."
            )
        if kwargs:
            args = self._bind_positional(args, kwargs)
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    self._loaded = self._materialize(args)
        if out is None:
            return self._loaded(*args)
        if not self._lean_bound:
            raise TypeError(
                "torch.compiler.export_python: out= is not available for this "
                "artifact; unsafe_reduce_overhead had no effect for its calling "
                "convention (see the warning logged on load)."
            )
        return self._loaded(*args, out=out)


def export_python(
    *,
    path: str,
    backend: str = "inductor",
    tracer: str = "make_fx",
    decompositions: dict | None = None,
    example_inputs: Sequence[object] | None = None,
    unsafe_reduce_overhead: bool = False,
) -> Callable[[Callable[..., Any]], ExportedPythonArtifact]:
    """See :func:`torch.compiler.export_python`."""

    def decorator(fn: Callable[..., Any]) -> ExportedPythonArtifact:
        return ExportedPythonArtifact(
            fn,
            path=path,
            backend=backend,
            tracer=tracer,
            decompositions=decompositions,
            example_inputs=example_inputs,
            unsafe_reduce_overhead=unsafe_reduce_overhead,
        )

    return decorator
