"""Locality-domain-aware tensor subclass and the public ``localize`` helper.

The public API surface is just :func:`localize`, which walks an
``nn.Module`` and replaces the parameters of every localizable layer --
an ``nn.Linear``'s weight and bias, by default -- with a tensor subclass
that splits along ``dim 0`` across the device's CUDA locality domains.
Forward calls on the layer then run the op as parallel branches on
green-context-backed streams that occupy disjoint SM partitions.

Which ops (and which layers) qualify is entirely a matter of what is in
the ``_LOCALIZABLE_OPS`` registry; see :func:`register_localizable_op`.

Typical use::

    model = MyModel().cuda()
    torch.cuda.localize(model)  # rewrites Linears in place
    # or:
    model = torch.cuda.localize(MyModel().cuda())

The tensor subclass (:class:`_LocalizedTensor`) and the per-device
handle (:class:`_LocalizationContext`) are internal -- callers should
not need them.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple, TYPE_CHECKING

import torch


# torch.fx.traceback, torch.utils._pytree, and torch.nn are imported lazily
# inside the functions that need them; importing at module top level
# triggers a circular import via torch.cuda's place in torch/__init__.py.
if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import torch.nn as nn

__all__ = ["localize"]


# ---------------------------------------------------------------------------
# Per-device handle for the shared green contexts / streams / mempools.
# ---------------------------------------------------------------------------


class _StreamSet:
    """One localized tensor's streams plus its fork/join sync events.

    Held directly on the ``_LocalizedTensor`` instance; every tensor (including
    one produced by ``detach``) gets its own fresh set, never shared.
    """

    def __init__(
        self,
        streams: Sequence[torch.cuda.Stream],
        default_event: torch.cuda.Event,
        branch_events: Sequence[torch.cuda.Event],
    ) -> None:
        self.streams = tuple(streams)
        self.default_event = default_event
        self.branch_events = tuple(branch_events)


class _LocalizationContext:
    """Per-device handle bundling the green contexts and locality-domain
    mempools that ``_LocalizedTensor`` instances on a device share.

    Heavyweight: each ``GreenContext`` claims an SM partition and each
    ``MemPool`` reserves caching-allocator state. Create
    one per device and reuse it across all LocalizedTensors on that device.
    ``_LocalizationContext.for_device(...)`` does this caching for you.

    This context should *not* be used to cache data as it is intended to have a
    longer lifetime.

    Streams and sync events are *per logical localized tensor*, never shared
    across tensors -- sharing them would serialize independent sites' work on a
    common FIFO and risk deadlock if the user relies on inter-stream sync to
    break a cross-site dependency cycle.  They live directly on each
    ``_LocalizedTensor`` -- every tensor gets its own fresh set (the context
    only mints them via :meth:`_make_stream_set`).
    """

    def __init__(
        self,
        green_contexts: Sequence[Any],
        mempools: list[Any],
        device: int,
    ) -> None:
        n = len(green_contexts)
        if len(mempools) != n:
            raise ValueError("partition counts must match across all lists")
        self.green_contexts = green_contexts
        self.mempools = mempools
        self.device = device

    @property
    def num_partitions(self) -> int:
        return len(self.green_contexts)

    def _make_stream_set(self) -> _StreamSet:
        """Create a fresh stream set: one stream per green context plus the
        fork/join sync events.  torch.cuda.Event is created lazily on first
        record, so recording a branch event inside ``torch.cuda.stream(green
        stream)`` associates it with the right context -- no context push
        needed (same pattern as ``execute_on_streams``)."""
        streams = tuple(gc.Stream() for gc in self.green_contexts)
        default_event = torch.cuda.Event()
        branch_events = [torch.cuda.Event() for _ in self.green_contexts]
        return _StreamSet(streams, default_event, branch_events)

    @staticmethod
    def for_device(device: int | None = None) -> _LocalizationContext:
        """Return the cached ``_LocalizationContext`` for ``device``,
        creating it lazily on first call.

        Specifying a default `None` device will create the context for the
        current device.

        Raises ``RuntimeError`` if green-context localization is
        unsupported on this device (requires CUDA 13.4+, and
        num_locality_domains >= 2).
        """
        if device is None:
            device = torch.cuda.current_device()
        cached = _LOCALIZATION_CONTEXT_CACHE.get(device)
        if cached is not None:
            return cached

        from torch.cuda.green_contexts import (
            get_num_locality_domains,
            GreenContext,
            is_localization_supported,
        )
        from torch.cuda.memory import LocalizedAllocator, MemPool

        if not is_localization_supported(device):
            raise RuntimeError(
                f"CUDA device {device} does not support green-context "
                "localization (requires CUDA 13.4+ and "
                "num_locality_domains >= 2)."
            )

        num_domains = get_num_locality_domains(device)
        gcs = GreenContext.split(
            locality_domain_ids=tuple(range(num_domains)), device_id=device
        )
        # Native pools belong to the current device, independently of allocator.device_id.
        with torch.cuda.device(device):
            pools = [
                MemPool(LocalizedAllocator(i, device=device).allocator())
                for i in range(num_domains)
            ]

        from torch._inductor.cudagraph_trees import register_external_mem_pool

        # The per-device context owns these pools for process lifetime. Keep them
        # registered across cudagraph-tree / Dynamo resets so later compilations of
        # localized models continue to capture with them.
        for pool in pools:
            register_external_mem_pool(pool, device=device)

        ctx = _LocalizationContext(gcs, pools, device)
        _LOCALIZATION_CONTEXT_CACHE[device] = ctx
        return ctx


_LOCALIZATION_CONTEXT_CACHE: dict[int, _LocalizationContext] = {}


# ---------------------------------------------------------------------------
# Localizable-op registry.
# ---------------------------------------------------------------------------
#
# Downstream wrappers (TRT-LLM, etc.) register their linear-shaped custom
# ops here so :class:`_LocalizedTensor`'s dispatch override knows how to
# split them across the locality-domain branches.  Three argument
# categories per op:
#
#   - ``localized_args``: arg positions that are pre-localized; each is
#     expected to be a :class:`_LocalizedTensor` (or ``None``) and the
#     dispatch reads its halves directly.  Typically the weight and an
#     optional bias.
#
#   - ``sliced_args``: arg positions that should be split per-branch as a
#     view but are *not* pool-localized.  Used for per-output-channel
#     side tensors like quantization scales -- they're small enough that
#     their default-pool storage doesn't meaningfully cost locality.
#     Values may be ``None``, an empty list, a Tensor, or a List[Tensor];
#     the dispatch handles each accordingly.
#
#   - everything else: passed through unchanged to each branch.
#
# Positions index the op's full signature, so a caller may pass any of them
# positionally or by keyword: ``arg_names`` maps a position to its keyword.
# A position the caller omitted entirely reads as ``None`` and stays omitted
# in the per-branch call, so the op's own default applies.


class _LocalizableOpMeta(NamedTuple):
    localized_args: tuple[int, ...]
    sliced_args: tuple[int, ...]
    # Signature order, for finding args the caller passed by keyword.  Empty
    # when the op has no schema to read them from and none were supplied, in
    # which case only positional args can be localized.
    arg_names: tuple[str, ...]
    # The nn.Module whose stock forward calls this op, when there is one.
    # localize() walks a module tree for these to find the layers to rewrite.
    module_type: type[nn.Module] | None


_LOCALIZABLE_OPS: dict[Any, _LocalizableOpMeta] = {}


def register_localizable_op(
    op: Any,
    *,
    localized_args: Sequence[int],
    sliced_args: Sequence[int] = (),
    arg_names: Sequence[str] | None = None,
    module_type: type[nn.Module] | None = None,
) -> None:
    """Register ``op`` as localizable.

    When :class:`_LocalizedTensor`'s dispatch fires on ``op``, the framework
    reads halves from each ``localized_args`` position, slices each
    ``sliced_args`` position along ``dim 0`` per branch, passes everything
    else through unchanged, runs ``op(*per_branch_args, **kwargs)`` on the
    branch's green-context-backed stream, and concatenates outputs.

    .. note:: The tensors are concatenated along dimension -1. If this is
    inappropriate for your network/use case, please file an issue and ask to
    specify the concatenation dimension as well.

    Args:
        op: the ``OpOverload`` to register, e.g.
            ``torch.ops.aten.linear.default``.
        localized_args: positions of args that are pre-localized into
            ``_LocalizedTensor`` halves (typically weight, bias).
        sliced_args: positions of args that should be sliced per-branch
            but not pool-localized (typically per-output-channel scales
            or zero-points for quantized linears).
        arg_names: the op's parameter names, in signature order, so args
            passed by keyword can be matched to the positions above.
            Defaults to the names in ``op``'s schema; pass explicitly for
            ops that have none (plain Python or builtin callables).
        module_type: the ``nn.Module`` whose stock ``forward`` calls
            ``op``.  When given, :func:`localize` rewrites the parameters
            named by ``localized_args`` on every such module it walks.

    Re-registering an op overwrites the previous metadata.
    """
    if op in _LOCALIZABLE_OPS:
        warnings.warn(f"Overwriting existing localization details on {str(op)}")
    if arg_names is None:
        schema = getattr(op, "_schema", None)
        arg_names = () if schema is None else tuple(a.name for a in schema.arguments)
    # localize() rewrites a module's parameters by name, so a module_type is
    # only usable if every localized position has one.
    if module_type is not None and any(i >= len(arg_names) for i in localized_args):
        raise ValueError(
            f"{op} registers module_type={module_type.__name__} but its "
            f"localized_args {tuple(localized_args)} are not all named by "
            f"arg_names={tuple(arg_names)}; pass arg_names explicitly."
        )
    _LOCALIZABLE_OPS[op] = _LocalizableOpMeta(
        localized_args=tuple(localized_args),
        sliced_args=tuple(sliced_args),
        arg_names=tuple(arg_names),
        module_type=module_type,
    )


class _LocalizedTensor(torch.Tensor):
    """Tensor subclass that splits a weight across CUDA locality domains.

    The halves are plain ``torch.Tensor`` objects allocated in different
    ``MemPool``s. ``__torch_dispatch__`` runs the ops registered
    in :data:`_LOCALIZABLE_OPS` as parallel branches on green-context-backed
    streams and concatenates the results. Concatenation always happens along
    dimension 0.
    """

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        halves: Sequence[torch.Tensor],
        ctx: _LocalizationContext,
    ) -> _LocalizedTensor:
        if len(halves) != ctx.num_partitions:
            raise ValueError(
                f"halves length ({len(halves)}) must match ctx.num_partitions "
                f"({ctx.num_partitions})"
            )
        if len(halves) != 2:
            raise NotImplementedError(
                f"_LocalizedTensor currently supports exactly 2 halves; got {len(halves)}"
            )
        h0, h1 = halves
        if h0.shape != h1.shape or h0.dtype != h1.dtype or h0.device != h1.device:
            raise ValueError(
                "all halves must have matching shape, dtype, and device; "
                f"got {[(tuple(h.shape), h.dtype, h.device) for h in halves]}"
            )
        # Compose the logical full shape from any-rank halves: halves are
        # stacked along dim 0 (out-features for a weight; bias is also
        # split along dim 0 since it's 1-D).
        full_shape = list(h0.shape)
        full_shape[0] = full_shape[0] * len(halves)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tuple(full_shape),
            dtype=h0.dtype,
            device=h0.device,
            layout=h0.layout,
            requires_grad=h0.requires_grad,
        )

    def __init__(
        self,
        halves: Sequence[torch.Tensor],
        ctx: _LocalizationContext,
    ) -> None:
        self._half0 = halves[0]
        self._half1 = halves[1]
        self._ctx = ctx

        # This tensor's own streams and fork/join sync events -- one fresh set
        # per tensor; independent sites never record/wait on the same events.
        self._sset = ctx._make_stream_set()

    @property
    def _halves(self) -> tuple[torch.Tensor, ...]:
        # The halves ordered for iteration.  A property (not a stored tuple)
        # so it tracks _half0 / _half1 across swap_tensors, which rewrites
        # only the instance __dict__ entries.
        return (self._half0, self._half1)

    @property
    def _streams(self) -> tuple[torch.cuda.Stream, ...]:
        return self._sset.streams

    @property
    def _default_event(self) -> torch.cuda.Event:
        return self._sset.default_event

    @property
    def _branch_events(self) -> tuple[torch.cuda.Event, ...]:
        return self._sset.branch_events

    # ``__torch_function__`` runs before aten decomposition, which is where
    # we want to intercept ``linear``: ``aten.linear`` is decomposed to
    # ``aten.t`` + ``aten.mm`` early in the dispatch chain, so by the time
    # ``__torch_dispatch__`` sees an op the 'linear' we want to override is long
    # gone.  Every localizable op -- the built-in ``F.linear`` as much as a
    # downstream custom op -- arrives here as ``func`` and is looked up in
    # ``_LOCALIZABLE_OPS``, which :func:`localize` populates before it can
    # hand out a _LocalizedTensor.
    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: Iterable[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}

        # Registered op: look up positions and dispatch per-branch.
        meta = _LOCALIZABLE_OPS.get(func)
        if meta is not None:
            return cls._localized_dispatch(func, args, kwargs, meta)

        # Anything else: drop the _LocalizedTensor torch_function hook and
        # let normal dispatch handle the call.  __torch_dispatch__ will
        # still fire on aten ops that touch a _LocalizedTensor.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    # ``__torch_dispatch__`` handles the lower-level ops that still touch a
    # _LocalizedTensor: ``detach`` (preserved subclass type) and the slow-path
    # fallback that reifies by concatenation.  Pushing pointwise tails onto
    # the per-branch streams (the "packing" optimization) is the
    # responsibility of a separate post-grad pass --
    # ``torch._inductor.fx_passes.pack_pointwise_after_cat`` -- not this
    # subclass.
    @classmethod
    # pyrefly: ignore [bad-override]
    def __torch_dispatch__(
        cls,
        op: Any,
        types: Iterable[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}

        # detach must return the same subclass type so nn.Parameter() can
        # wrap a _LocalizedTensor without triggering its "detach() returned
        # the wrong type" check.  Like any fresh localized tensor, the result
        # gets its own stream set.
        if op is torch.ops.aten.detach.default:
            (inp,) = args
            if isinstance(inp, _LocalizedTensor):
                halves = tuple(h.detach() for h in inp._halves)
                return _LocalizedTensor(halves, inp._ctx)

        # Slow path: reify each _LocalizedTensor by concatenating its halves and
        # re-dispatch on plain tensors.  We intentionally walk args manually
        # (not via tree_map) because pytree treats _LocalizedTensor as a node
        # and would re-wrap the result, causing infinite recursion through the
        # dispatch.
        def reify(t: Any) -> Any:
            if isinstance(t, _LocalizedTensor):
                return torch.cat(t._halves, dim=0)
            if isinstance(t, (list, tuple)):
                return type(t)(reify(x) for x in t)
            return t

        new_args = tuple(reify(a) for a in args)
        new_kwargs = {k: reify(v) for k, v in kwargs.items()}
        return op(*new_args, **new_kwargs)

    # -- per-op implementations ---------------------------------------------

    @classmethod
    def _localized_dispatch(
        cls,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict,
        meta: _LocalizableOpMeta,
    ) -> torch.Tensor:
        """Run ``func(*args, **kwargs)`` as parallel branch dispatches.

        Per-branch handling of each registered position, whether the caller
        passed it positionally or by keyword:
          - ``localized_args``: must be ``None`` or a ``_LocalizedTensor``;
            the matching half is substituted.
          - ``sliced_args``: ``None``, ``Tensor``, or ``List[Tensor]``;
            split along ``dim 0`` (or replicated if empty/None).
          - everything else: passed through unchanged.

        A registered position the caller omitted reads as ``None`` and is
        left out of the per-branch call, so the op's own default applies.

        Output halves are concatenated along ``dim -1``.
        """
        localized_args, sliced_args = meta.localized_args, meta.sliced_args

        # How the caller passed each registered position: by keyword (the
        # name), or positionally (None, meaning args[i]).  A position the
        # caller omitted is absent from the map.  Reads and the per-branch
        # write-back both go through this, so a value the caller passed by
        # keyword is handed back to the branch under that same keyword
        # rather than appended at a positional index someone else owns.
        keyword_of: dict[int, str | None] = {}
        for i in (*localized_args, *sliced_args):
            if i < len(args):
                keyword_of[i] = None
            elif i < len(meta.arg_names) and meta.arg_names[i] in kwargs:
                keyword_of[i] = meta.arg_names[i]

        def value_of(i: int) -> Any:
            if i not in keyword_of:
                return None
            keyword = keyword_of[i]
            return args[i] if keyword is None else kwargs[keyword]

        # Find the primary _LocalizedTensor (first non-None entry in
        # ``localized_args``) to pull context, streams, and events from.
        primary: _LocalizedTensor | None = None
        for i in localized_args:
            v = value_of(i)
            if isinstance(v, _LocalizedTensor):
                primary = v
                break
        if primary is None:
            raise AssertionError("Localized op did not have a localized tensor?")

        n_branches = len(primary._halves)

        # Pre-collect per-branch values for each localized position.
        localized_halves: dict[int, list[torch.Tensor | None]] = {}
        for i in localized_args:
            v = value_of(i)
            if v is None:
                localized_halves[i] = [None] * n_branches
                continue
            assert isinstance(v, _LocalizedTensor), (  # noqa: S101
                f"{func} arg {i} must be None or a _LocalizedTensor; got "
                f"{type(v).__name__}.  Use torch.cuda.localize(module) to "
                "rewrite paired tensors (weight and bias) together."
            )
            assert v._ctx is primary._ctx, (  # noqa: S101
                "all _LocalizedTensor args must share the same "
                "_LocalizationContext (same device, same green contexts)."
            )
            assert len(v._halves) == n_branches, (  # noqa: S101
                f"arg {i} has {len(v._halves)} halves; primary has "
                f"{n_branches}; they must match."
            )
            localized_halves[i] = list(v._halves)

        # Pre-split sliced_args along dim 0.
        sliced_per_branch: dict[int, list[Any]] = {}
        for i in sliced_args:
            v = value_of(i)
            if v is None:
                sliced_per_branch[i] = [None] * n_branches
            elif isinstance(v, torch.Tensor):
                sliced_per_branch[i] = list(torch.tensor_split(v, n_branches, dim=0))
            elif isinstance(v, (list, tuple)) and len(v) == 0:
                empty: Any = [] if isinstance(v, list) else ()
                sliced_per_branch[i] = [empty for _ in range(n_branches)]
            elif isinstance(v, (list, tuple)):
                # List[Tensor]: slice each element, transpose to per-branch.
                per_branch: list[list[Any]] = [[] for _ in range(n_branches)]
                for t in v:
                    splits = (
                        list(torch.tensor_split(t, n_branches, dim=0))
                        if isinstance(t, torch.Tensor)
                        else [t] * n_branches
                    )
                    for b in range(n_branches):
                        per_branch[b].append(splits[b])
                sliced_per_branch[i] = [type(v)(p) for p in per_branch]
            else:
                raise TypeError(
                    f"{func} arg {i} must be None, Tensor, or list/tuple of "
                    f"Tensors for sliced_args; got {type(v).__name__}"
                )

        # Every position we substitute per branch.  Localized wins if a
        # position was somehow registered as both.
        substitutions: dict[int, list[Any]] = {**sliced_per_branch, **localized_halves}

        # Mark non-localized inputs as ready on the calling stream so each
        # branch can wait on this event before reading them.
        default_stream = torch.cuda.current_stream(primary._ctx.device)
        primary._default_event.record(default_stream)

        outputs: list[torch.Tensor] = []
        for b in range(n_branches):
            stream = primary._streams[b]
            evt = primary._branch_events[b]

            # Substitute this branch's values back into whichever of args or
            # kwargs the caller passed them in.
            per_branch_args = list(args)
            per_branch_kwargs = dict(kwargs)
            for i, per_branch in substitutions.items():
                if i not in keyword_of:
                    continue
                keyword = keyword_of[i]
                if keyword is None:
                    per_branch_args[i] = per_branch[b]
                else:
                    per_branch_kwargs[keyword] = per_branch[b]

            # Stream contexts supply Dynamo's per-branch stream metadata to Inductor.
            with torch.cuda.stream(stream):
                stream.wait_event(primary._default_event)
                with torch.cuda.use_mem_pool(
                    primary._ctx.mempools[b], device=primary._ctx.device
                ):
                    h = func(*per_branch_args, **per_branch_kwargs)
                evt.record(stream)
            outputs.append(h)

        # Block default stream until all branches are done before cat
        # (eager-mode sync; see comment on the default-event record above).
        for evt in primary._branch_events:
            default_stream.wait_event(evt)
        return torch.cat(outputs, dim=-1)


def _localization_supported_or_warn(device: int) -> bool:
    """True if ``device`` supports green-context localization; otherwise
    warn (once per process per device) and return False so
    :func:`localize` can skip the parameter instead of raising."""
    from torch.cuda.green_contexts import is_localization_supported

    if is_localization_supported(device):
        return True
    # Lazy import: torch._dynamo isn't importable at torch.cuda
    # module-import time (circular via torch.cuda's place in torch/__init__),
    # but is always available by the time this runs.
    from torch._dynamo.utils import warn_once

    warn_once(
        f"CUDA device {device} does not support green-context "
        "localization (requires a CUDA 13.4+ driver and hardware with "
        ">= 2 locality domains); skipping localization.",
        stacklevel=5,
    )
    return False


def _localize_tensor(
    tensor: torch.Tensor,
    *,
    num_partitions: int | None = None,
    mark_static_halves: bool = True,
) -> _LocalizedTensor:
    """Internal: split a single tensor across locality domains.

    Used by :func:`localize` for each localizable layer's parameters.
    Returns a :class:`_LocalizedTensor` whose halves are allocated in
    per-locality-domain mempools.  Public callers should reach for
    :func:`localize` (which walks an ``nn.Module`` tree) instead of invoking
    this directly.

    If `num_partitions` is `None`, the tensor will be divided into however many
    partitions makes sense for the hardware.

    Everything is localized onto ``tensor``'s own device; splitting a tensor
    across some other device's locality domains would mean moving it there.

    When ``mark_static_halves`` is True (the default) each half is tagged with
    :func:`torch._dynamo.mark_static_address` before being wrapped.  The halves
    live in our LocalizationContext mempools, so their data pointers are stable
    across iterations by construction; marking them static lets cudagraph_trees
    skip the per-iter ``_foreach_copy_`` it would otherwise emit for non-static
    graph inputs.
    """
    if not tensor.is_cuda:
        raise ValueError(f"localize requires a CUDA tensor; got device={tensor.device}")

    device = tensor.device.index
    if num_partitions is None:
        num_partitions = torch.cuda.green_contexts.get_num_locality_domains(device)

    if tensor.shape[0] % num_partitions != 0:
        raise ValueError(
            f"tensor.shape[0] ({tensor.shape[0]}) must be divisible by "
            f"num_partitions ({num_partitions})"
        )

    ctx = _LocalizationContext.for_device(device)
    if num_partitions != ctx.num_partitions:
        raise NotImplementedError(
            f"only {ctx.num_partitions}-way splits supported on device {device}"
        )

    # Slice the source and copy each piece into the matching mempool so
    # the half lives in the locality-domain-local HBM the matmul will
    # read from.
    slices = list(torch.tensor_split(tensor, num_partitions, dim=0))
    halves: list[torch.Tensor] = []
    for slice_, pool in zip(slices, ctx.mempools):
        with torch.cuda.use_mem_pool(pool, device=device):
            half = torch.empty_like(slice_)
        # Fill under no_grad so the copy does not make the half a non-leaf
        # view of the source, then take the source's autograd flag: the
        # halves are what every localized op actually reads, so a parameter
        # that requires grad has to have halves that require it too.
        with torch.no_grad():
            half.copy_(slice_)
        half.requires_grad_(tensor.requires_grad)
        halves.append(half)

    if mark_static_halves:
        # Import lazily: torch._dynamo isn't necessarily importable at
        # torch.cuda module-import time, but is always available by the
        # time anyone calls localize().
        from torch._dynamo.decorators import mark_static_address

        for half in halves:
            mark_static_address(half)

    return _LocalizedTensor(halves, ctx)


def _register_default_localizable_ops() -> None:
    """Register the built-in localizable ops.

    ``F.linear`` is what ``nn.Linear.forward`` calls and is therefore the
    op that drives eager localization; ``aten.linear`` is its exported
    shape, so users who localize a gm with raw aten.linear nodes get it
    for free.  Both take ``(input, weight, bias)``.

    Called lazily rather than at import time because ``torch.ops`` is not
    yet populated when ``torch.cuda.__init__`` pulls this module in.
    Idempotent: skips ops already registered.
    """
    import torch.nn as nn

    # F.linear is a builtin with no introspectable signature, so its
    # parameter names have to be spelled out for keyword matching.
    if torch.nn.functional.linear not in _LOCALIZABLE_OPS:
        register_localizable_op(
            torch.nn.functional.linear,
            localized_args=(1, 2),
            arg_names=("input", "weight", "bias"),
            module_type=nn.Linear,
        )
    if torch.ops.aten.linear.default not in _LOCALIZABLE_OPS:
        register_localizable_op(torch.ops.aten.linear.default, localized_args=(1, 2))


def localize(
    module: nn.Module,
    *,
    num_partitions: int | None = None,
    mark_static_halves: bool = True,
) -> nn.Module:
    """Localize every localizable layer in ``module``'s tree (recursive, in place).

    A layer is localizable when its type was registered as some op's
    ``module_type`` via :func:`register_localizable_op`; out of the box that
    is :class:`torch.nn.Linear`, via ``F.linear``.  For each such layer
    reached from ``module``, the parameters named by the op's
    ``localized_args`` (``.weight`` and ``.bias`` for a linear) are replaced
    with a tensor subclass that splits along ``dim 0`` across the device's
    locality domains.  Forward calls on the layer then run the op as
    parallel branches on green-context-backed streams that occupy disjoint
    SM partitions, with the split parameters pre-placed into matching
    locality-domain mempools.

    The swap is done with :func:`torch.utils.swap_tensors`, which flips
    the parameter's ``__class__`` / ``__dict__`` / slots and the
    underlying ``at::Tensor`` pointer in place.  This means:

      * Python identity is preserved -- any reference captured before
        the call (e.g. ``w = model.fc.weight``) resolves to the
        localized parameter afterward.
      * Tied parameters are preserved automatically: a Parameter held
        by two modules (``head.weight = embed.weight``) is a single
        Python object, so the in-place swap is observed by both holders
        without an extra alias-walk.

    Unlocalizable holders of a tied weight (such as ``nn.Embedding``) will
    reify the full tensor on each forward via the slow concat path, but
    produce correct results.

    Unregistered module types, and registered ones that are already
    localized, are left alone.  The same ``module`` object is returned for
    chaining.

    Localization targets inference: any live ``.grad`` on a swapped
    parameter is cleared, and the fork/join is not set up to backprop
    through the split, so localized modules should not be trained.

    Args:
        module: the module (or model) to walk.  Mutated in place.
        num_partitions: forwarded to the per-tensor splitter.  Defaults
            to the device's locality-domain count (see
            :func:`torch.cuda.green_contexts.get_num_locality_domains`).
        mark_static_halves: when True (default), each per-domain
            half of every localized weight/bias is tagged with
            :func:`torch._dynamo.mark_static_address`.  The halves live
            in green-context-pinned mempools so their addresses are
            stable across iterations; marking them static lets
            cudagraph_trees skip the per-iter ``_foreach_copy_`` it
            would otherwise emit.  Disable for A/B benches that want
            to attribute the cudagraph win to this knob.

    If a parameter's device does not support green-context localization
    (hardware without locality domains, or a driver older than CUDA
    13.4), the parameter is left untouched and a warning is emitted
    once per process; the module keeps working unlocalized.

    Example::

        model = MyModel().cuda()
        torch.cuda.localize(model)  # in place
        # or
        model = torch.cuda.localize(MyModel().cuda())  # chained
    """
    import torch.nn as nn

    if not torch.is_inference_mode_enabled():
        raise RuntimeError(
            "torch.cuda.localize() requires inference mode. Localization is "
            "inference-only: it splits each parameter into per-locality-domain "
            "halves that the fork/join does not backprop through. Wrap the "
            "call, and the forward passes that use the localized module, in "
            "torch.inference_mode()."
        )

    _register_default_localizable_ops()

    if isinstance(module, torch.fx.GraphModule):
        return _localize_graph_module(
            module,
            num_partitions=num_partitions,
            mark_static_halves=mark_static_halves,
        )

    # Track the Python objects we've already visited, keyed by id().  Holds
    # both params we swapped and params we skipped (unsupported / not on
    # CUDA), so a tied parameter (one Parameter held by multiple modules)
    # is only processed once -- swap_tensors mutates the underlying Python
    # object in place, so every alias observes the result automatically.
    visited: set[int] = set()

    def _swap_to_localized(p: torch.Tensor) -> None:
        if id(p) in visited:
            return
        if isinstance(p, _LocalizedTensor) or not p.is_cuda:
            visited.add(id(p))
            return
        if not _localization_supported_or_warn(p.device.index):
            visited.add(id(p))
            return
        # We mark static halves *after* the Parameter wrap + swap below.
        # nn.Parameter(_LocalizedTensor) detach()s its input, which
        # rebuilds the wrapper via its __torch_dispatch__ detach handler:
        # the new halves share storage with the original ones but are fresh Python
        # objects, so any ``_dynamo_static_input_type`` attribute set on
        # the originals would be lost.  Pass ``mark_static_halves=False``
        # to ``_localize_tensor`` here and re-apply the mark on
        # ``p._halves`` after swap_tensors, which are the final ones.
        new = nn.Parameter(
            _localize_tensor(
                p,
                num_partitions=num_partitions,
                mark_static_halves=False,
            ),
            requires_grad=p.requires_grad,
        )
        # swap_tensors poisons the AccumulateGrad node attached to the
        # original parameter (see torch.utils.swap_tensors); clearing
        # ``p.grad`` first avoids that diagnostic.  Localization is
        # inference-only and does not backprop through the fork/join, so a
        # live gradient here is silently discarded -- warn before we do.
        if p.grad is not None:
            if p.requires_grad:
                from torch._dynamo.utils import warn_once

                warn_once(
                    "localize() is discarding a live .grad on a parameter with "
                    "requires_grad=True; localization is inference-only and does "
                    "not backprop through the fork/join split.",
                    stacklevel=4,
                )
            p.grad = None
        # swap_tensors inspects the parameter's autograd edge when anything
        # else still references it (use_count 2), and inference mode leaves
        # no grad_fn for it to find -- the lookup fails with an AttributeError
        # from autograd internals.  A caller localizing a model that was just
        # trained is exactly that case: a live graph still holds the
        # parameter.  Turn inference mode off for the swap itself; the
        # surrounding requirement is about how the localized module is *used*,
        # not about this bookkeeping.
        with torch.inference_mode(False):
            torch.utils.swap_tensors(p, new)
        # Same reason the static marks are re-applied here: the Parameter
        # wrap detached and rebuilt the halves, dropping the flag
        # _localize_tensor set on the originals.  These are the halves the
        # localized ops will read, so the flag has to land on them.
        for half in p._halves:  # type: ignore[attr-defined]
            half.requires_grad_(p.requires_grad)
        if mark_static_halves:
            from torch._dynamo.decorators import mark_static_address

            for half in p._halves:  # type: ignore[attr-defined]
                mark_static_address(half)
        visited.add(id(p))

    # Module types some registered op declared, and the parameters to rewrite
    # on each: the op's localized positions, named.  For F.linear those are
    # "weight" and "bias", which is exactly what nn.Linear calls them.
    localizable_modules = [
        (meta.module_type, [meta.arg_names[i] for i in meta.localized_args])
        for meta in _LOCALIZABLE_OPS.values()
        if meta.module_type is not None
    ]

    for child in module.modules():
        for module_type, param_names in localizable_modules:
            if not isinstance(child, module_type):
                continue
            # Only localize the stock forward: a subclass overriding it may
            # touch the parameters in ways the _LocalizedTensor swap silently
            # breaks -- in-place mutation (dropped by the reify path),
            # raw-pointer custom kernels (no contiguous storage), or weight
            # reparameterization -- so skip anything not inheriting it verbatim.
            if type(child).forward is not module_type.forward:
                continue
            for name in param_names:
                p = getattr(child, name, None)
                if p is not None:
                    _swap_to_localized(p)
            break
    return module


def _localize_graph_module(
    gm: torch.fx.GraphModule,
    *,
    num_partitions: int | None = None,
    mark_static_halves: bool = True,
) -> torch.fx.GraphModule:
    """:func:`localize` for an exported FX graph.

    Walks ``gm.graph`` for ``call_function`` nodes whose target is in
    :data:`_LOCALIZABLE_OPS`, resolves the ``get_attr`` parameter behind
    each localized-position arg, and replaces it in place with a
    :class:`_LocalizedTensor` Parameter.  The graph is not rewritten;
    the subclass takes over at runtime via ``__torch_function__``.

    Tied params (multiple ``get_attr`` nodes pointing at the same Python
    object, or aliases held by other submodules) are coalesced via the
    same id-keyed map used by the ``nn.Module`` path.
    """
    import torch.nn as nn

    # Snapshot all (submodule, attr_name) holding each parameter, so a
    # tied param replaced once propagates to every alias on gm.
    id_to_locations: dict[int, list[tuple[nn.Module, str]]] = {}
    for _, submod in gm.named_modules():
        for pname, p in submod._parameters.items():
            if p is None:
                continue
            id_to_locations.setdefault(id(p), []).append((submod, pname))

    seen: dict[int, nn.Parameter] = {}

    def _resolve_get_attr(node: torch.fx.Node) -> tuple[nn.Module, str] | None:
        if node.op != "get_attr":
            return None
        parts = str(node.target).split(".")
        owner: Any = gm
        for p in parts[:-1]:
            owner = getattr(owner, p)
        return owner, parts[-1]

    def _localize_attr(submod: nn.Module, attr_name: str) -> None:
        p = getattr(submod, attr_name, None)
        if p is None or isinstance(p, _LocalizedTensor) or not p.is_cuda:
            return
        if not _localization_supported_or_warn(p.device.index):
            return
        wid = id(p)
        new = seen.get(wid)
        if new is None:
            # See the note in localize()'s _swap_to_localized: the
            # Parameter wrap detaches and rebuilds the halves, so the
            # mark must be applied on the final wrapped halves, not on
            # the ones _localize_tensor produced internally.
            new = nn.Parameter(
                _localize_tensor(
                    p,
                    num_partitions=num_partitions,
                    mark_static_halves=False,
                ),
                requires_grad=p.requires_grad,
            )
            for half in new._halves:  # type: ignore[attr-defined]
                half.requires_grad_(p.requires_grad)
            if mark_static_halves:
                from torch._dynamo.decorators import mark_static_address

                for half in new._halves:  # type: ignore[attr-defined]
                    mark_static_address(half)
            seen[wid] = new
        for s, n in id_to_locations.get(wid, [(submod, attr_name)]):
            setattr(s, n, new)

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        meta = _LOCALIZABLE_OPS.get(node.target)
        if meta is None:
            continue
        for i in meta.localized_args:
            if i < len(node.args):
                arg = node.args[i]
            elif i < len(meta.arg_names):
                arg = node.kwargs.get(meta.arg_names[i])
            else:
                continue
            if not isinstance(arg, torch.fx.Node):
                continue
            resolved = _resolve_get_attr(arg)
            if resolved is None:
                continue
            _localize_attr(*resolved)
    return gm
