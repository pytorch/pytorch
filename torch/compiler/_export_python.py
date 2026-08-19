"""Path-cached ``torch.compiler.export_python`` decorator.

``torch.compiler.precompile`` captures a function ahead of time and lowers it to a
self-contained, human-readable Python source artifact (see
``torch/_precompile.py``). ``torch.compiler.export_python`` wraps that in a
decorator keyed off a file on disk: the first run writes the emitted
``python_code`` to ``path``; every later run reads the ``.py`` back and executes
it directly instead of recompiling.

Because the artifact is self-contained, re-executable Python, ``path`` is meant to be
committed and shipped -- and, when a kernel starts to matter, hand-edited in place by
an engineer or an agent. This is ejectable compilation: the emitted source is the
source of truth and is always exec'd, so an edit is simply what runs from then on, in
production as much as in development. There is no acceleration cache and no
``precompile.load`` round-trip, so keeping the edited source correct is the caller's
responsibility.
"""

import ast
import copy
import errno
import functools
import inspect
import logging
import os
import re
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

# Written as the artifact's first line so a later load can detect it was produced
# by a different torch (see _warn_on_version_skew). It is a comment, so it does not
# affect exec; a hand-edit that drops it just disables the skew warning, so
# hill-climbing an artifact never triggers a spurious version warning.
# All seven stamps must stay in the artifact's LEADING comment block: the reader stops
# at the first line that is not a comment, so inserting code above them turns every
# check off -- each checked stamp warns per call while it is missing, and the version
# warning is the only one that goes quiet.
_VERSION_TAG = "# torch.compiler.export_python torch-version: "
# The train()/eval() flags of every nn.Module argument at capture. Python control flow
# on ``self.training`` is specialized into the graph with no runtime guard, so this
# stamp is what catches an artifact captured in one mode being run in the other. Like
# the version stamp it is exec-inert, and dropping it in a hand-edit just turns the
# check off (see _check_module_training).
_MODULE_TRAINING_TAG = "# torch.compiler.export_python module-training: "
# Which input tensors overlapped in memory at capture, which of them were literally the
# same object, and the ambient autocast state. make_fx bakes all three into the graph
# with no runtime guard: aliased inputs change what a mutation means, one object passed
# twice is deduped into a single graph slot, and autocast changes the dtypes the kernels
# were specialized for. Exec-inert like the other stamps, and dropping one turns only
# its own check off. What no stamp guards is a change in HOW two aliased inputs overlap
# -- capture's relative offsets stay baked in, exactly as they do under torch.compile.
_INPUT_OVERLAP_TAG = "# torch.compiler.export_python input-overlap: "
_INPUT_DUPLICATE_TAG = "# torch.compiler.export_python input-duplicates: "
_AUTOCAST_TAG = "# torch.compiler.export_python autocast: "
# Ambient process state the generated code bakes that no other stamp covers: the default
# dtype and device a factory op with no explicit argument resolves against, and whether
# deterministic algorithms were on when inductor chose between a deterministic and an
# atomic lowering.
_GLOBAL_STATE_TAG = "# torch.compiler.export_python global-state: "
# The CPU vector ISA the artifact's C++ kernels were generated against. Inductor bakes
# the host's vector width into a C++ loop's stride -- a reduction emitted on AVX-512
# steps and stores 16 floats at a time -- while the ISA itself is re-picked when the
# artifact is compiled on whichever machine loads it. Replay under a narrower vector unit
# and each store covers half of its own step, leaving the rest of the output as whatever
# the allocator held: no error, no warning, and a result that is not close to anything.
# Unlike the CUDA case there is no kernel-image error to fall back on, and no other stamp
# covers it, so this one raises rather than warns.
_CPU_ISA_TAG = "# torch.compiler.export_python cpu-vec-isa: "

# os.link failures that mean the filesystem cannot do hard links at all, as opposed to
# a real I/O problem (a full disk, a bad permission) that must not be swallowed.
_NO_HARDLINK_ERRNOS = frozenset(
    getattr(errno, name)
    for name in ("EPERM", "EOPNOTSUPP", "ENOTSUP", "EXDEV", "EMLINK", "ENOSYS")
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
    tmp = os.path.join(
        dir_name,
        f".{base}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(8)}.tmp",
    )
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


# Elements below this fraction of the reference's peak are too small for a relative diff
# to say anything; they are covered by the absolute term instead.
_REL_REPORT_FLOOR = 1e-6


def _allclose(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> bool:
    """torch.allclose, but tolerant of dtypes that have no comparison kernel.

    float8 reaches allclose and dies inside it on a missing mul (as NotImplementedError,
    which is a RuntimeError); promote and compare there rather than failing an artifact
    that is perfectly honest.
    """
    try:
        return bool(torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    except RuntimeError:
        return bool(
            torch.allclose(a.double(), b.double(), rtol=rtol, atol=atol, equal_nan=True)
        )


def _finite_mask(t: torch.Tensor) -> torch.Tensor | None:
    """Where ``t`` is finite, or None for a dtype that cannot answer.

    float8 is is_floating_point() but has no isfinite kernel, so asking crashes the
    check on an artifact that is perfectly honest.
    """
    if not t.is_floating_point():
        return None
    try:
        return torch.isfinite(t)
    except RuntimeError:
        return None


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
        # Owns no bytes, which is not the same as "bytes we cannot find". Both report
        # data_ptr 0, so without this one such component (an absent bias, an empty KV
        # cache, an uneven shard, a meta-initialized slot) would poison its whole
        # wrapper into "assume it aliases everything" and refuse every donation against
        # it. This is about a leaf that genuinely IS meta; a wrapper merely presenting
        # meta over live payloads is still distrusted, in _reports_no_bytes.
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
        # An addressless or undecomposable tensor (a wrapper subclass whose components
        # we cannot reach, sparse, nested). Its bytes are unknown, so the only safe
        # answer is "assume it aliases" -- NOT storage identity, which is the predicate
        # that made byte-disjoint arena slices look aliased in the first place. Device
        # type is all that can still rule a pair out. Note this reads the OUTER device
        # of both operands, including one that did resolve, because there is no leaf on
        # the unresolved side to pair its leaves against; a wrapper misreporting its
        # type is therefore still taken at its word here.
        return a.device.type == b.device.type
    if not (
        len(a_leaves) == 1
        and a_leaves[0] is a
        and len(b_leaves) == 1
        and b_leaves[0] is b
    ):
        return any(_shares_memory(x, y) for x in a_leaves for y in b_leaves)
    if a.device != b.device:
        # Compared here, on the LEAF, and never on the wrapper: a subclass can report a
        # device that differs from the one holding its bytes in index (torch.load with
        # map_location="cuda" over a "cuda:0" payload) or in TYPE (map_location=
        # {"cuda:0": "cpu"}). Gating above reported such a tensor as sharing memory with
        # nothing -- including with its own payload.
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

    Keyed on the LEAF device only, matching _shares_memory: a wrapper subclass can
    report a device that differs from the one holding its bytes, in index (torch.load
    with map_location="cuda" over a "cuda:0" payload) or in type (map_location=
    {"cuda:0": "cpu"}), so the wrapper's own report decides nothing. An empty list means
    t owns no addressable bytes, so it overlaps nothing.
    """
    if _reports_no_bytes(t):
        return []
    leaves = _dense_leaves(t)
    if leaves is None:
        return None
    atoms = []
    for leaf in leaves:
        if leaf.numel() == 0 or leaf.is_meta:
            continue
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


def _code_devices(code: str) -> set[str]:
    """The device types the emitted artifact names, from its own source.

    The stamp and the per-call check have to agree, so this is computed from the source
    once at capture and once at load rather than from anything ambient.
    """
    found = set(re.findall(r"empty_strided_(\w+)\(", code))
    found.update(re.findall(r"device\(type=[\'\"](\w+)[\'\"]", code))
    found.update(re.findall(r"\.to\([\'\"](\w+)[\'\"]", code))
    devices = set()
    for name in found:
        # The allocator names are not all device types -- empty_strided_cpu_pinned is a
        # cpu allocator -- and a hand-edited artifact can contain anything at all.
        try:
            devices.add(torch.device(name).type)
        except (RuntimeError, ValueError):
            continue
    return devices


def _autocast_state(
    args: Sequence[Any],
    tensors: list[torch.Tensor] | None = None,
    code_devices: set[str] | None = None,
) -> list[list[Any]]:
    """Ambient autocast for every device type this artifact can compute on.

    The input devices alone are not enough: a graph whose inputs are all on CPU can still
    run its matmuls on an accelerator, and keying only on inputs recorded [] for it in
    both processes, so the check passed while the kernels had been built for autocast
    dtypes. Widened with the device types the emitted code names. Still not every device
    type in the process: a CPU helper first called inside a `with torch.autocast("cuda")`
    training region must not be locked to that region, and its graph never names cuda.
    """
    if tensors is None:
        tensors = _input_tensors(args)
    devices = {t.device.type for t in tensors}
    if code_devices is not None:
        devices.update(code_devices)
    return [
        [device, str(torch.get_autocast_dtype(device))]
        for device in sorted(devices)
        if torch.amp.is_autocast_available(device) and torch.is_autocast_enabled(device)
    ]


def _cpu_vec_isa(code: str) -> str | None:
    """The CPU vector ISA a C++ kernel in ``code`` was generated against, or None.

    None for an artifact with no C++ kernel, where the question does not arise -- a CUDA
    artifact must not start refusing to run because it moved between two hosts whose CPUs
    differ in a way it never depended on.
    """
    if not re.search(r"cpp_pybinding|cpp_fused", code):
        return None
    from torch._inductor.cpu_vec_isa import pick_vec_isa

    return str(pick_vec_isa())


def _global_state() -> list[list[Any]]:
    """Ambient globals the emitted code resolves against, as sorted [key, value] pairs.

    Recorded as strings so the stamp round-trips through ast.literal_eval. Both are read
    at CAPTURE and baked: a factory op with no dtype= takes the default dtype then, and
    inductor picks a deterministic or an atomic-add lowering from the determinism flag.
    The artifact never re-consults either, so a process that changes one and replays gets
    capture's answer with no error.

    float32_matmul_precision is deliberately NOT here. torch.get_float32_matmul_precision
    raises outright in a process that has used the per-backend fp32_precision API, so
    stamping it killed capture there; and on the default config the artifact reaches
    extern_kernels.mm, which re-reads the setting at run time, so the check refused calls
    the artifact would have served correctly. Only a max_autotune Triton GEMM template
    bakes it as a tl.constexpr.
    """
    return [
        ["default_dtype", str(torch.get_default_dtype())],
        ["default_device", str(torch.get_default_device())],
        ["deterministic", str(torch.are_deterministic_algorithms_enabled())],
    ]


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
        self._signature = inspect.signature(fn)
        self._call_signature = self._signature
        self._path = path
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._example_inputs = None if example_inputs is None else tuple(example_inputs)
        self._module_training: list[tuple[int, list[tuple[str, bool]]]] | None = None
        self._input_overlaps: list[list[int]] | None = None
        self._input_duplicates: list[list[int]] | None = None
        self._global_state: list[list[str]] | None = None
        self._cpu_isa: str | None = None
        self._cpu_isa_stamped = False
        self._code_devices: set[str] = set()
        self._autocast: list[list[Any]] | None = None
        self._loaded: Callable[..., Any] | None = None
        # (pid, tid) currently inside _materialize. There is deliberately no
        # per-artifact lock: capture is already serialized process-wide, so a second
        # lock would add nothing but a second acquisition order to deadlock against.
        # This is the re-entrancy guard the (reentrant) capture lock cannot provide.
        # The pid makes it fork-safe without a registry -- a child never matches a
        # marker left by a thread it did not inherit.
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
            if _input_overlaps(example) != _input_overlaps(args):
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
        else:
            example = self._bind_positional(example, {}, "example_inputs=")
            self._check_supported_args(example)
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
        # The stamps lead the artifact as exec-inert comments, each guarding one thing
        # make_fx specialized without a runtime guard. A hand-edit may drop any of them;
        # each just turns its own check off.
        example_tensors = _input_tensors(example)
        code = (
            f"{_VERSION_TAG}{torch.__version__!r}\n"
            f"{_MODULE_TRAINING_TAG}{_module_training_state(example)!r}\n"
            f"{_INPUT_OVERLAP_TAG}{_input_overlaps(example, example_tensors)!r}\n"
            f"{_INPUT_DUPLICATE_TAG}{_input_duplicates(example, example_tensors)!r}\n"
            f"{_AUTOCAST_TAG}"
            f"{_autocast_state(example, example_tensors, _code_devices(code))!r}\n"
            f"{_GLOBAL_STATE_TAG}{_global_state()!r}\n"
            f"{_CPU_ISA_TAG}{_cpu_vec_isa(code)!r}\n{code}"
        )
        if _atomic_publish(self._path, code.encode("utf-8")):
            return code, False
        # Lost the publish race; the winner's file is complete and already linked.
        winner = self._load_from_disk()
        if winner is None:
            raise _precompile_error(
                f"torch.compiler.export_python: another writer published {self._path} "
                "and it was deleted before this call could load it. Retry."
            )
        return winner, True

    def _load_from_disk(self) -> str | None:
        # None means "not there after all" -- the presence gate raced a peer deleting
        # the artifact to force a regenerate, which should fall through to capture
        # rather than surface a bare FileNotFoundError.
        try:
            with open(self._path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise _precompile_error(
                f"torch.compiler.export_python: could not read the artifact at "
                f"{self._path} ({e.strerror}). Check that the path names a readable "
                "file rather than a directory."
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
        for line in code.splitlines():
            if line.startswith(tag):
                try:
                    return ast.literal_eval(line[len(tag) :])
                except (ValueError, SyntaxError):
                    # A mangled stamp is a hand-edit like any other: turn the check off
                    # rather than raise a SyntaxError from a comment line that does not
                    # affect what the artifact runs.
                    return None
            # An INDENTED comment is still a comment. The documented rule is "the
            # first non-comment line", and stopping early here silently turns every
            # later stamp's check off rather than reading it.
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                break
        return None

    def _check_capture_environment(self, args: tuple[Any, ...]) -> None:
        # Two more things make_fx specialized with no runtime guard. Input ALIASING
        # decides what an in-place mutation means, so an artifact captured with two
        # arguments sharing memory computes the wrong thing (and mutates the wrong
        # thing) when they are distinct at runtime -- torch.compile guards this and
        # recompiles. Ambient AUTOCAST decides the dtypes the kernels were built for.
        tensors: list[torch.Tensor] | None = None

        def input_tensors() -> list[torch.Tensor]:
            # Walked once and shared by the three checks below. Lazily, so an artifact
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
        if self._global_state is None:
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "global-state stamp, so calling it under a different default dtype, "
                "default device or determinism setting than capture "
                "is unchecked. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        else:
            actual_state = dict(_global_state())
            try:
                recorded = dict(self._global_state)
            except (TypeError, ValueError):
                # A hand-edit can leave a literal that is not [key, value] pairs. Every
                # other stamp degrades to warn-and-continue for that; this one must not
                # be the only one that raises a raw unpack error out of the load path.
                log.warning(
                    "torch.compiler.export_python: the global-state stamp at %s is not a "
                    "list of [key, value] pairs, so that check is off. Delete %s to "
                    "regenerate it.",
                    self._path,
                    self._path,
                )
                recorded = {}
            for key, captured in recorded.items():
                if key not in actual_state:
                    continue  # a key this torch no longer records
                live = actual_state[key]
                if live == captured:
                    continue
                # Determinism is one-sided: capture with it OFF and replay with it ON
                # means the artifact keeps a lowering the caller has asked not to run.
                # ON at capture and OFF at replay is conservative, so it is not an error.
                if key == "deterministic" and captured == "True":
                    continue
                raise _precompile_error(
                    f"torch.compiler.export_python: {key} is {live} but the artifact "
                    f"was captured with {key} {captured}. It is resolved when the code "
                    "is generated and baked in, so this call would silently get "
                    "capture's answer. Set it to the captured value, or delete "
                    f"{self._path} to recapture."
                )
        if self._autocast is None:
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "autocast stamp, so calling it under a different autocast context than "
                "capture is unchecked. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        else:
            actual_autocast = _autocast_state(args, input_tensors(), self._code_devices)
            if actual_autocast != self._autocast:
                raise _precompile_error(
                    "torch.compiler.export_python: the runtime autocast state does not "
                    f"match capture (captured {self._autocast}, got {actual_autocast}). "
                    "Autocast dtypes are baked into the artifact; capture under the "
                    "same autocast context you call it in."
                )
        if not self._cpu_isa_stamped:
            # Every other checked stamp warns per call while it is missing. This one was
            # silent, which made it the only guard a hand-edit could switch off without
            # saying anything -- and it is the one guarding the loudest failure, a C++
            # kernel whose baked vector width does not match the host's.
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "cpu-vec-isa stamp, so running a C++ kernel built for a different "
                "vector width than this host's is unchecked, and that failure is "
                "silent. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        elif self._cpu_isa is not None:
            live_isa = _cpu_vec_isa("cpp_fused")
            if live_isa is not None and live_isa != self._cpu_isa:
                raise _precompile_error(
                    "torch.compiler.export_python: this machine's CPU vector ISA is "
                    f"{live_isa!r} but the artifact's C++ kernels were generated for "
                    f"{self._cpu_isa!r}. The vector width is baked into their loop "
                    "strides while the ISA is re-picked at compile time, so running "
                    "them here would write past the end of an output or leave part of "
                    f"it uninitialized. Delete {self._path} to regenerate on this "
                    "machine, or run where the captured ISA is available."
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
            raise _precompile_error(
                "torch.compiler.export_python: the runtime module training state does "
                f"not match capture (expected {self._module_training!r}, got {actual!r}). "
                "Restore train()/eval() state or regenerate the artifact."
            )

    def _warn_on_version_skew(self, code: str) -> None:
        # Warn (but still run) when the artifact carries a version stamp that does
        # not match the current torch, so a committed artifact gone stale across a
        # torch upgrade is visible rather than silently running old logic. A missing
        # stamp (dropped by a hand-edit) is silent, so hill-climbing never warns.
        produced = self._read_stamp(code, _VERSION_TAG)
        if produced is None:
            # Artifacts written before the stamp became a repr() carry a bare version
            # string, which literal_eval rejects. Falling back to the raw text keeps the
            # skew warning working for every artifact already committed -- silently
            # losing it is exactly the case the stamp exists for.
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
        # precompile.load round-trip). A clobbered hand-edit (dropped forward / syntax
        # error) and an environment or version mismatch (an import that fails under the
        # current torch) surface as distinct, actionable PrecompileErrors rather than
        # one catch-all "delete to regenerate".
        from torch._precompile import _make_inlined_forward, PrecompileError

        try:
            if from_disk:
                log.warning(
                    "torch.compiler.export_python is about to EXEC the artifact at %s; "
                    "the file is trusted executable Python and may have been edited or "
                    "replaced since export. Only load paths whose contents you trust.",
                    self._path,
                )
            return _make_inlined_forward(code, warn=False, filename=self._path)
        except SyntaxError as e:
            # Kernels are hoisted to module level, so Python reports a typo in one
            # against this file at the right line. Say so: telling someone to delete an
            # artifact they are midway through tuning is the wrong advice.
            where = f" at line {e.lineno}" if e.lineno else ""
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} does not "
                f"parse{where}: {e.msg}. Fix it there, or delete the file to regenerate "
                "from the original function."
            ) from e
        except KeyError as e:
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
        code = self._load_from_disk() if os.path.exists(self._path) else None
        from_disk = code is not None
        if code is None:
            code, from_disk = self._precompile_and_save(args)
        if from_disk:
            self._warn_on_version_skew(code)
        self._module_training = self._read_stamp(code, _MODULE_TRAINING_TAG)
        self._input_overlaps = self._read_stamp(code, _INPUT_OVERLAP_TAG)
        self._input_duplicates = self._read_stamp(code, _INPUT_DUPLICATE_TAG)
        self._autocast = self._read_stamp(code, _AUTOCAST_TAG)
        self._global_state = self._read_stamp(code, _GLOBAL_STATE_TAG)
        self._cpu_isa = self._read_stamp(code, _CPU_ISA_TAG)
        self._cpu_isa_stamped = self._read_raw_stamp(code, _CPU_ISA_TAG) is not None
        self._code_devices = _code_devices(code)
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
                    self._loaded = self._materialize(args)
                finally:
                    self._materializing = None
            return self._loaded

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

    def _check_supported_args(self, args: tuple[Any, ...]) -> None:
        params = list(self._call_signature.parameters)
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
            name = params[pos] if pos < len(params) else f"argument {pos}"
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
