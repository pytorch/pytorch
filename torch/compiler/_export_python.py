"""Path-cached ``torch.compiler.export_python`` decorator.

``torch.compiler.precompile`` captures a function ahead of time and lowers it to a
self-contained, human-readable Python source artifact (see
``torch/_precompile.py``). ``torch.compiler.export_python`` wraps that in a
decorator keyed off a file on disk: the first run writes the emitted
``python_code`` to ``path``; every later run reads the ``.py`` back and executes
it directly instead of recompiling.
"""

import ast
import copy
import errno
import functools
import inspect
import logging
import os
import secrets
import threading
from collections.abc import Callable, Sequence
from typing import Any, cast, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Every stamp must stay in the artifact's LEADING comment block: the reader stops at the
# first line that is not a comment, so inserting code above them turns every check off.
# Each checked stamp then warns per call, and the version warning is the only one that
# goes quiet.

# Written as the artifact's first line so a later load can detect it was produced
# by a different torch (see _warn_on_version_skew). It is a comment, so it does not
# affect exec; a hand-edit that drops it just disables the skew warning, so
# hill-climbing an artifact never triggers a spurious version warning.
_VERSION_TAG = "# torch.compiler.export_python torch-version: "
# The train()/eval() flags of every nn.Module argument at capture. Python control flow
# on ``self.training`` is specialized into the graph with no runtime guard, so this
# stamp is what catches an artifact captured in one mode being run in the other. Like
# the version stamp it is exec-inert, and dropping it in a hand-edit just turns the
# check off (see _check_module_training).
_MODULE_TRAINING_TAG = "# torch.compiler.export_python module-training: "
# Which input tensors overlapped in memory at capture, and which of them were literally
# the same object. make_fx bakes both into the graph with no runtime guard: aliased
# inputs change what a mutation means, and one object passed twice is deduped into a
# single graph slot. Exec-inert like the other stamps, and dropping one turns only its
# own check off. What no stamp guards is a change in HOW two aliased inputs overlap --
# capture's relative offsets stay baked in, exactly as they do under torch.compile.
_INPUT_OVERLAP_TAG = "# torch.compiler.export_python input-overlap: "
_INPUT_DUPLICATE_TAG = "# torch.compiler.export_python input-duplicates: "

# os.link failures that mean the filesystem cannot do hard links at all, as opposed to
# a real I/O problem (a full disk, a bad permission) that must not be swallowed. EINVAL
# is how Windows reports a volume without hard links (FAT, exFAT).
_NO_HARDLINK_ERRNOS = frozenset(
    getattr(errno, name)
    for name in (
        "EPERM",
        "EOPNOTSUPP",
        "ENOTSUP",
        "EXDEV",
        "EMLINK",
        "ENOSYS",
        "EINVAL",
    )
    if hasattr(errno, name)
)


def _atomic_publish(path: str, data: bytes) -> bool:
    # Publish a fully-written file, never a partial one, and report whether this call
    # is the writer that published it. A hard link is the no-replace publish: exactly
    # one concurrent writer wins and every loser loads that winner rather than exec'ing
    # its own divergent source. Only errnos that mean "this filesystem has no hard
    # links" fall back to replace (last-writer-wins, still never partial); a full disk
    # or a permissions problem must surface rather than silently weaken the guarantee.
    dir_name = os.path.dirname(path) or "."
    base = os.path.basename(path)
    tmp = os.path.join(dir_name, f".{base}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        except OSError as e:
            if e.errno not in _NO_HARDLINK_ERRNOS:
                raise
            log.warning(
                "torch.compiler.export_python: %s has no hard links (%s), so %s is "
                "published last-writer-wins; concurrent first writers may each run "
                "their own generated source.",
                dir_name,
                e.strerror,
                path,
            )
            os.replace(tmp, path)
        return True
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def _precompile_error(msg: str) -> Exception:
    from torch._precompile import PrecompileError

    return PrecompileError(msg)


def _module_training_state(
    args: Sequence[Any],
) -> list[tuple[int, list[tuple[str, bool]]]]:
    return [
        (pos, [(name, module.training) for name, module in arg.named_modules()])
        for pos, arg in enumerate(args)
        if isinstance(arg, torch.nn.Module)
    ]


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    # The [start, end) byte range the tensor can touch. Exact for a dense tensor and a
    # bounding range for a strided one, which errs toward reporting overlap.
    start = t.data_ptr()
    extent = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    return start, start + (extent + 1) * t.element_size()


def _dense_leaves(t: torch.Tensor) -> list[torch.Tensor] | None:
    """The real-memory tensors inside t, or None if its bytes cannot be located.

    A wrapper subclass (DTensor, TwoTensor, FunctionalTensor) reports data_ptr() == 0
    with a real device and is_meta False, so comparing its byte span against a plain
    tensor's would report every pair as disjoint. Decompose it instead.
    """
    if is_traceable_wrapper_subclass(t):
        try:
            attrs, _ = t.__tensor_flatten__()
        except Exception:
            return None
        leaves: list[torch.Tensor] = []
        saw_tensor = False
        for attr in attrs:
            # __tensor_flatten__ names the attributes that must be transformed, and
            # not all of them are tensors -- DTensor's list includes its DeviceMesh.
            component = getattr(t, attr, None)
            if not isinstance(component, torch.Tensor):
                continue
            saw_tensor = True
            inner = _dense_leaves(component)
            if inner is None:
                return None
            leaves.extend(inner)
        # An empty list here means every component owns no bytes, which is a real
        # answer. Only a subclass we could not see INTO is unresolvable -- returning
        # `leaves` unconditionally would make such a wrapper disjoint from everything
        # and let a donor be written over a live input.
        return leaves if saw_tensor else None
    if t.numel() == 0 or t.is_meta:
        # Owns no bytes, unlike "bytes we cannot find" (both report data_ptr 0): one empty
        # component must not make its whole wrapper alias everything.
        return []
    try:
        if t.data_ptr() == 0:
            return None
    except RuntimeError:
        return None
    return [t]


def _reports_no_bytes(t: torch.Tensor) -> bool:
    """Whether t can be ruled out from its own report, without locating its bytes.

    Only for tensors that are what they say they are. A wrapper subclass's numel and
    is_meta describe what it PRESENTS: torch.load with
    map_location={torch.device("cpu"): "meta"} -- keyed by a device OBJECT, which remaps
    the wrapper without remapping its storages -- builds one reporting meta over live
    cpu payloads, and taking that at face value says it aliases nothing, including its
    own payload. The string form {"cpu": "meta"} does the reverse and is not the case
    this guards.
    """
    return not is_traceable_wrapper_subclass(t) and (t.numel() == 0 or t.is_meta)


def _shares_memory(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors can touch the same bytes.

    NOT torch._C._overlaps, which is IValue::overlaps -- storage IDENTITY, ignoring
    offsets. That reports byte-disjoint slices of one buffer as overlapping, which
    rejects the arena / fused-QKV / KV-cache shape this API exists to serve.

    Compares addresses, not storage objects: the same bytes can be reached through
    different UntypedStorages (from_numpy on overlapping slices, frombuffer, DLPack,
    __cuda_array_interface__ onto a live arena), and a storage-identity gate reports
    those as disjoint -- a donor would then be written over a live input. data_ptr is a
    process-global address and differing devices are rejected at the LEAF below, so
    distinct allocations cannot collide. One allocation visible under two device types
    still can: mapped pinned host memory has the same address as its CUDA view, and
    this reports the pair disjoint. torch._C._overlaps answers that case identically.
    Conservative for anything whose extent cannot be computed: a bounding range for
    strided tensors, and for sparse or otherwise unlocatable tensors "aliases anything
    of the same device type" -- NOT storage identity, which is the predicate this
    function exists to avoid.
    """
    if _reports_no_bytes(a) or _reports_no_bytes(b):
        # No real memory, and every meta tensor reports data_ptr 0, which would
        # otherwise make every pair look coincident.
        return False
    a_leaves, b_leaves = _dense_leaves(a), _dense_leaves(b)
    if a_leaves is None or b_leaves is None:
        # Bytes unknown (sparse, nested, an undecomposable wrapper): assume it aliases
        # anything of the same device type, read off the leaves where there are any.
        a_types = {x.device.type for x in ([a] if a_leaves is None else a_leaves)}
        b_types = {x.device.type for x in ([b] if b_leaves is None else b_leaves)}
        return not a_types.isdisjoint(b_types)
    if is_traceable_wrapper_subclass(a) or is_traceable_wrapper_subclass(b):
        return any(_shares_memory(x, y) for x in a_leaves for y in b_leaves)
    if a.device != b.device:
        # On the leaf, never the wrapper: a subclass can report a device differing from
        # its bytes' in index (torch.load map_location="cuda" over "cuda:0") or in type.
        return False
    try:
        (a_start, a_end), (b_start, b_end) = _byte_span(a), _byte_span(b)
    except RuntimeError:
        return True
    return a_start < b_end and b_start < a_end


def _input_tensors(args: Sequence[Any]) -> list[torch.Tensor]:
    """Every tensor an artifact call can reach, in a stable order.

    Module params and buffers are included, and come after the user tensors so their
    presence does not shift the indices a previously written stamp recorded. They have
    to be here: AOTAutograd dedups a user tensor that aliases a module buffer into one
    graph slot, so an artifact captured with that alias computes -- and mutates -- the
    wrong thing when the runtime call passes independent tensors, and the reverse.
    """
    user = [
        leaf
        for arg in args
        if not isinstance(arg, torch.nn.Module)
        for leaf in pytree.tree_leaves(arg)
        if isinstance(leaf, torch.Tensor)
    ]
    module: list[torch.Tensor] = []
    seen: set[int] = set()
    for arg in args:
        if not isinstance(arg, torch.nn.Module):
            continue
        for tensor in [*arg.parameters(), *arg.buffers()]:
            if id(tensor) not in seen:
                seen.add(id(tensor))
                module.append(tensor)
    return [*user, *module]


def _span_atoms(
    t: torch.Tensor,
) -> list[tuple[torch.device, int, int]] | None:
    """t's byte ranges as (leaf device, start, end), or None if they cannot be located.

    Keyed on the leaf device, never the wrapper's, matching _shares_memory. An empty list
    means t owns no addressable bytes, so it overlaps nothing.
    """
    if _reports_no_bytes(t):
        return []
    leaves = _dense_leaves(t)
    if leaves is None:
        return None
    atoms = []
    for leaf in leaves:
        try:
            start, end = _byte_span(leaf)
        except RuntimeError:
            return None
        atoms.append((leaf.device, start, end))
    return atoms


def _input_overlaps(
    args: Sequence[Any], tensors: list[torch.Tensor] | None = None
) -> list[list[int]]:
    """Which pairs of input tensors share memory, as sorted [i, j] index pairs.

    Runs on every artifact call, so it is a sweep and not the P-choose-2 pairwise scan:
    a 32-block model has hundreds of parameters, and the quadratic form cost multiples
    of the forward it guards. Lists (not tuples) so the stamp round-trips through
    ast.literal_eval to something that compares equal.
    """
    if tensors is None:
        tensors = _input_tensors(args)
    by_device: dict[torch.device, list[tuple[int, int, int]]] = {}
    unresolved: list[int] = []
    for i, tensor in enumerate(tensors):
        atoms = _span_atoms(tensor)
        if atoms is None:
            unresolved.append(i)
            continue
        for leaf, start, end in atoms:
            by_device.setdefault(leaf, []).append((start, end, i))
    pairs: set[tuple[int, int]] = set()
    for spans in by_device.values():
        spans.sort()
        # Spans that started earlier and have not ended: exactly the ones this span can
        # overlap, since the list is sorted by start.
        open_spans: list[tuple[int, int]] = []
        for start, end, i in spans:
            open_spans = [(e, j) for e, j in open_spans if e > start]
            for _, j in open_spans:
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
            open_spans.append((end, i))
    for i in unresolved:
        for j, other in enumerate(tensors):
            if j != i and _shares_memory(tensors[i], other):
                pairs.add((min(i, j), max(i, j)))
    return [[i, j] for i, j in sorted(pairs)]


def _input_duplicates(
    args: Sequence[Any], tensors: list[torch.Tensor] | None = None
) -> list[list[int]]:
    """Which input positions hold the SAME tensor object, as [first, repeat] pairs.

    Not implied by _input_overlaps: AOTAutograd dedups arguments that are one object
    into a single graph slot, and byte overlap cannot tell that apart from two views
    that merely intersect. Both report the same pair set, so without this an artifact
    captured from overlapping views silently computes the wrong thing when handed one
    tensor twice. torch.compile guards it and recompiles ("Duplicate tensors found").
    """
    if tensors is None:
        tensors = _input_tensors(args)
    first: dict[int, int] = {}
    pairs = []
    for i, tensor in enumerate(tensors):
        seen_at = first.setdefault(id(tensor), i)
        if seen_at != i:
            pairs.append([seen_at, i])
    return pairs


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
    ) -> None:
        self._fn = fn
        self._call_signature = inspect.signature(fn)
        self._path = path
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._example_inputs = None if example_inputs is None else tuple(example_inputs)
        self._module_training: list[tuple[int, list[tuple[str, bool]]]] | None = None
        self._input_overlaps: list[list[int]] | None = None
        self._input_duplicates: list[list[int]] | None = None
        self._loaded: Callable[..., Any] | None = None
        # (pid, tid) currently inside _materialize: the re-entrancy guard the reentrant
        # capture lock cannot provide. The pid keeps it correct in a forked child, which
        # never matches a marker left by a thread it did not inherit.
        self._materializing: tuple[int, int] | None = None

    def _precompile_and_save(self, args: tuple[Any, ...]) -> tuple[str, bool]:
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
        else:
            example = self._bind_positional(example, {}, "example_inputs=")
            self._check_supported_args(example)
        # Before capture: fn may call train()/eval() itself, and the per-call check reads
        # the caller's modules before the call runs.
        module_training = _module_training_state(example)
        # Taken before _compile, which runs fn on example and can change its aliasing
        # (set_ on an input).
        example_tensors = _input_tensors(example)
        example_overlaps = _input_overlaps(example, example_tensors)
        if self._example_inputs is None and example_overlaps != _input_overlaps(args):
            from torch._precompile import PrecompileError

            raise PrecompileError(
                "torch.compiler.export_python: deep-copying the first-call "
                "arguments did not preserve how their tensors share memory, so "
                "capturing from the copy would bake in aliasing the real arguments "
                "do not have. nn.Parameter.__deepcopy__ clones, so two Parameters "
                "backed by one storage become independent in the copy. Pass "
                "example_inputs=... built with the same sharing as the real "
                "arguments to capture against those instead."
            )
        # Only the python_code is written: the emitted source is self-contained and
        # always exec'd, so export_python never builds precompile's acceleration cache.
        from torch._precompile import PrecompiledModule

        compiled = PrecompiledModule(
            self._fn,
            backend=self._backend,
            tracer=self._tracer,
            decompositions=self._decompositions,
        )
        compiled._compile(example)
        code = compiled.to_python_code()
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # The stamps lead the artifact as exec-inert comments, each guarding one thing
        # make_fx specialized without a runtime guard. A hand-edit may drop any of them;
        # each just turns its own check off.
        code = (
            f"{_VERSION_TAG}{torch.__version__}\n"
            f"{_MODULE_TRAINING_TAG}{module_training!r}\n"
            f"{_INPUT_OVERLAP_TAG}{example_overlaps!r}\n"
            f"{_INPUT_DUPLICATE_TAG}{_input_duplicates(example, example_tensors)!r}\n"
            f"{code}"
        )
        # os.link does not follow a symlink at its destination, so resolve one first: a
        # dangling symlink at path would otherwise read as a lost race forever.
        target, data = os.path.realpath(self._path), code.encode("utf-8")
        while not _atomic_publish(target, data):
            # Lost the publish race. If the winner's file is already gone (a peer
            # deleting it to force a regenerate), publish again rather than fail.
            winner = self._load_from_disk()
            if winner is not None:
                return winner, True
        return code, False

    def _load_from_disk(self) -> str | None:
        # None means "not there after all" -- the presence gate raced a peer deleting
        # the artifact to force a regenerate, which should fall through to capture
        # rather than surface a bare FileNotFoundError.
        try:
            with open(self._path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as e:
            hint = " rather than a directory" if os.path.isdir(self._path) else ""
            raise _precompile_error(
                f"torch.compiler.export_python: could not read the artifact at "
                f"{self._path} ({e}). Check that the path names a readable UTF-8 "
                f"file{hint}."
            ) from e

    @staticmethod
    def _read_raw_stamp(code: str, tag: str) -> str | None:
        for line in code.splitlines():
            if line.startswith(tag):
                return line[len(tag) :].strip() or None
            # An INDENTED comment is still a comment. The documented rule is "the
            # first non-comment line", and stopping early here silently turns every
            # later stamp's check off rather than reading it.
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                break
        return None

    @staticmethod
    def _read_stamp(code: str, tag: str) -> Any:
        raw = ExportedPythonArtifact._read_raw_stamp(code, tag)
        if raw is None:
            return None
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            # A mangled stamp is a hand-edit like any other: turn the check off rather
            # than raise a SyntaxError from a comment line that does not affect what the
            # artifact runs.
            return None

    def _check_capture_environment(self, args: tuple[Any, ...]) -> None:
        # Another thing make_fx specialized with no runtime guard. Input ALIASING
        # decides what an in-place mutation means, so an artifact captured with two
        # arguments sharing memory computes the wrong thing (and mutates the wrong
        # thing) when they are distinct at runtime -- torch.compile guards this and
        # recompiles.
        tensors: list[torch.Tensor] | None = None

        def input_tensors() -> list[torch.Tensor]:
            # Walked once and shared by the two checks below. Lazily, so an artifact
            # whose stamps were all hand-edited away does not pay for a walk no check
            # will read.
            nonlocal tensors
            if tensors is None:
                tensors = _input_tensors(args)
            return tensors

        if self._input_overlaps is None:
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "input-aliasing stamp, so a runtime call whose inputs share memory "
                "differently than capture is unchecked. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        else:
            actual = _input_overlaps(args, input_tensors())
            if actual != self._input_overlaps:
                raise _precompile_error(
                    "torch.compiler.export_python: the runtime inputs do not share "
                    f"memory the way capture did (captured overlapping index pairs "
                    f"{self._input_overlaps}, got {actual}). Aliasing is baked into the "
                    "graph, so this call would compute against the wrong assumption."
                )
        if self._input_duplicates is None:
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "input-duplicate stamp, so a runtime call that repeats a tensor object "
                "differently than capture is unchecked. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        else:
            actual_duplicates = _input_duplicates(args, input_tensors())
            if actual_duplicates != self._input_duplicates:
                raise _precompile_error(
                    "torch.compiler.export_python: the runtime inputs repeat tensor "
                    "objects differently than capture did (captured duplicate index "
                    f"pairs {self._input_duplicates}, got {actual_duplicates}). "
                    "AOTAutograd folds arguments that are one object into a single "
                    "graph slot, so this call would compute against the wrong "
                    "assumption -- byte overlap alone cannot see the difference."
                )

    def _check_module_training(self, args: tuple[Any, ...]) -> None:
        actual = _module_training_state(args)
        if not actual:
            return
        if self._module_training is None:
            # Missing stamp means a hand-edit dropped it, the same as the version
            # stamp: warn that the guard is off rather than refuse to run an artifact
            # whose source is by design the thing the caller is free to edit.
            log.warning(
                "torch.compiler.export_python: the artifact at %s takes nn.Module "
                "arguments but carries no recorded training state, so train()/eval() "
                "skew against capture is unchecked. Delete %s to regenerate the stamp.",
                self._path,
                self._path,
            )
            return
        if actual != self._module_training:
            # Name the first flip only: a per-submodule dump of a real model runs to
            # thousands of characters.
            try:
                was = {(p, n): t for p, mods in self._module_training for n, t in mods}
            except (TypeError, ValueError):
                was = {}
            flips = [
                (p, n, t)
                for p, mods in actual
                for n, t in mods
                if was.get((p, n), t) != t
            ]
            if flips:
                p, n, t = flips[0]
                more = f" and {len(flips) - 1} more" if len(flips) > 1 else ""
                detail = (
                    f"argument {p} submodule {n or '<root>'!r} was training={not t} at "
                    f"capture, is training={t} now{more}"
                )
            else:
                detail = "the nn.Module arguments' submodules differ from capture"
            raise _precompile_error(
                "torch.compiler.export_python: the runtime module training state does "
                f"not match capture ({detail}). Restore train()/eval() state or "
                "regenerate the artifact."
            )

    def _warn_on_version_skew(self, code: str) -> None:
        # Warn (but still run) when the artifact carries a version stamp that does
        # not match the current torch, so a committed artifact gone stale across a
        # torch upgrade is visible rather than silently running old logic. A missing
        # stamp (dropped by a hand-edit) is silent, so hill-climbing never warns.
        produced = self._read_raw_stamp(code, _VERSION_TAG)
        # str(): TorchVersion.__eq__ PEP-440-parses its operand and re-raises anything
        # that is not InvalidVersion, so a stamp of 4300+ digits took the whole load path
        # down with a ValueError. Comparing text keeps a mangled stamp to a warning.
        if produced is None or produced == str(torch.__version__):
            return
        log.warning(
            "torch.compiler.export_python: the artifact at %s was produced by "
            "torch %s but the current torch is %s; running it as-is. Delete %s "
            "to regenerate against the current torch.",
            self._path,
            produced,
            torch.__version__,
            self._path,
        )

    def _load(self, code: str, *, from_disk: bool) -> Callable[..., Any]:
        # The emitted source is self-contained: exec it directly (no cache, no
        # precompile.load round-trip). A broken hand-edit and an environment or version
        # mismatch (an import that fails under the current torch) surface as distinct,
        # actionable PrecompileErrors rather than one catch-all "delete to regenerate".
        from torch._precompile import _make_inlined_forward, PrecompileError

        if not from_disk:
            # Source this call just captured is precompile's own output, so a failure
            # running it is a bug to surface as-is, not a file to fix or delete.
            return _make_inlined_forward(code, warn=False, filename=self._path)
        log.warning(
            "torch.compiler.export_python is about to EXEC the artifact at %s; "
            "the file is trusted executable Python and may have been edited or "
            "replaced since export. Only load paths whose contents you trust.",
            self._path,
        )
        try:
            return _make_inlined_forward(code, warn=False, filename=self._path)
        except PrecompileError:
            raise
        except SyntaxError as e:
            if e.filename != self._path:
                # Raised by code the artifact runs (an import, a nested exec); the
                # line belongs to that file, so report it like any other failure.
                raise PrecompileError(
                    "torch.compiler.export_python: an unexpected error occurred running "
                    f"the artifact at {self._path} ({type(e).__name__}: {e}). Fix it "
                    "there, or delete it to regenerate."
                ) from e
            # Kernels are hoisted to module level, so Python reports a typo in one
            # against this file at the right line. Say so: telling someone to delete an
            # artifact they are midway through tuning is the wrong advice.
            where = f" at line {e.lineno}" if e.lineno else ""
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} does not "
                f"parse{where}: {e.msg}. Fix it there, or delete the file to regenerate "
                "from the original function."
            ) from e
        except ImportError as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} failed "
                f"to import a dependency ({e}); it was edited, or produced by a "
                "different torch version or environment. Fix it there, or delete it "
                "to regenerate against the current torch."
            ) from e
        except Exception as e:
            raise PrecompileError(
                "torch.compiler.export_python: an unexpected error occurred running "
                f"the artifact at {self._path} ({type(e).__name__}: {e}). Fix it there, "
                "or delete it to regenerate."
            ) from e

    def _materialize(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        code = self._load_from_disk() if os.path.exists(self._path) else None
        from_disk = code is not None
        if code is None:
            code, from_disk = self._precompile_and_save(args)
        if from_disk:
            self._warn_on_version_skew(code)
        self._module_training = self._read_stamp(code, _MODULE_TRAINING_TAG)
        self._input_overlaps = self._read_stamp(code, _INPUT_OVERLAP_TAG)
        self._input_duplicates = self._read_stamp(code, _INPUT_DUPLICATE_TAG)
        entry = self._load(code, from_disk=from_disk)
        self._example_inputs = None
        self._decompositions = None
        return entry

    def _materialize_once(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        # Materialization runs under the one process-wide capture lock. Capture runs
        # fn, which may call another decorated function, so any second lock taken
        # around this would give two threads two orders to acquire them in and deadlock
        # -- which is why the artifact holds no lock of its own.
        import torch._precompile as precompile_impl

        ident = (os.getpid(), threading.get_ident())
        if self._materializing == ident:
            raise _precompile_error(
                "torch.compiler.export_python: re-entrant call into "
                f"{getattr(self._fn, '__name__', 'fn')} while it is being precompiled. "
                "A decorated function cannot call itself: capture would have to run "
                "inside its own capture. Move the recursion into an undecorated helper."
            )
        with precompile_impl._CAPTURE_LOCK:
            if self._loaded is None:
                self._materializing = ident
                try:
                    self._loaded = self._serialize_first_launch(self._materialize(args))
                finally:
                    self._materializing = None
            return self._loaded

    def _serialize_first_launch(self, entry: Callable[..., Any]) -> Callable[..., Any]:
        # CachingAutotuner.run checks its launcher count outside the autotuner's lock, and
        # autotune_to_one_config (benchmark every config, release the losers, replace the
        # launchers) takes no lock, so two first launches of a kernel compile-time
        # autotuning did not pin race each other. Racing first calls take turns under the
        # capture lock until one launch succeeds, and only then see the bare entry; a
        # launch that raises leaves the next call serialized too.
        import torch._precompile as precompile_impl

        def first_launch(*args: Any) -> Any:
            with precompile_impl._CAPTURE_LOCK:
                out = entry(*args)
                self._loaded = entry
                return out

        return first_launch

    def _bind_positional(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        source: str = "the call arguments",
    ) -> tuple[Any, ...]:
        # The artifact's forward is positional (the precompile calling convention),
        # so map any keyword call args onto fn's positional parameters -- this lets
        # callers invoke the decorated fn naturally (e.g. rope(q=..., k=...)).
        # Anything that cannot be laid out positionally is rejected below.
        sig = self._call_signature
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"torch.compiler.export_python: could not bind {source} to "
                f"{getattr(self._fn, '__name__', 'fn')}'s signature: {e}"
            ) from e
        bound.apply_defaults()
        # After apply_defaults every positional-or-keyword parameter is placed, so
        # bound.kwargs holds only keyword-only and **kwargs entries.
        if bound.kwargs:
            params = sig.parameters
            kw_only = sorted(
                n
                for n in bound.kwargs
                if n in params and params[n].kind == inspect.Parameter.KEYWORD_ONLY
            )
            if kw_only:
                raise TypeError(
                    "torch.compiler.export_python does not support functions that "
                    f"declare keyword-only parameters ({kw_only}); the precompile "
                    "calling convention is positional."
                )
            raise TypeError(
                "torch.compiler.export_python does not support **kwargs parameters "
                f"(got {sorted(bound.kwargs)}); the precompile calling convention is "
                "positional."
            )
        return bound.args

    def _check_supported_args(self, args: tuple[Any, ...]) -> None:
        # args is the bound positional layout: the named positional parameters in
        # order, then any *args values.
        P = inspect.Parameter
        params = self._call_signature.parameters.values()
        positional = (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)
        names = [p.name for p in params if p.kind in positional]
        var = next((p.name for p in params if p.kind == P.VAR_POSITIONAL), None)
        for pos, arg in enumerate(args):
            if isinstance(arg, torch.nn.Module):
                continue
            unsupported = [
                leaf
                for leaf in pytree.tree_leaves(arg)
                if not isinstance(leaf, torch.Tensor)
            ]
            if not unsupported:
                continue
            name = names[pos] if pos < len(names) else f"{var}[{pos - len(names)}]"
            # These two land often enough that the generic "close the constant over"
            # advice is actively wrong for them: a module must stay an argument, and an
            # optional parameter has no constant to close over in the first place.
            if any(isinstance(leaf, torch.nn.Module) for leaf in unsupported):
                raise TypeError(
                    "torch.compiler.export_python: nn.Module arguments must be passed "
                    f"directly, not nested inside a container (parameter {name!r}). "
                    "Pass the module itself as its own positional argument."
                )
            if all(leaf is None for leaf in unsupported):
                raise TypeError(
                    "torch.compiler.export_python does not support None arguments "
                    f"(parameter {name!r}); make_fx specializes the None branch without "
                    "a runtime guard. Split the function, or pass a tensor."
                )
            raise TypeError(
                "torch.compiler.export_python supports only Tensor pytrees and "
                "nn.Module positional arguments; Python scalar/config values are "
                "specialized by make_fx without runtime guards. Close constants "
                f"over in the function instead of passing parameter {name!r} "
                f"({unsupported[0]!r})."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        args = self._bind_positional(args, kwargs)
        self._check_supported_args(args)
        loaded = self._loaded
        if loaded is None:
            loaded = self._materialize_once(args)
        self._check_capture_environment(args)
        self._check_module_training(args)
        return loaded(*args)


def export_python(
    *,
    path: str,
    backend: str = "inductor",
    tracer: str = "make_fx",
    decompositions: dict | None = None,
    example_inputs: Sequence[object] | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """See :func:`torch.compiler.export_python`."""

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        artifact = ExportedPythonArtifact(
            fn,
            path=path,
            backend=backend,
            tracer=tracer,
            decompositions=decompositions,
            example_inputs=example_inputs,
        )

        @functools.wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return cast("_R", artifact(*args, **kwargs))

        return wrapped

    return decorator
