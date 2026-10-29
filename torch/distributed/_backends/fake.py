# mypy: allow-untyped-defs

import copy
import functools
import inspect

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    FakeStore,
    FakeWork,
    GatherOptions,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
)
from torch._prims_common import suggest_memory_format


# FakeStore is a no-op Key-Value store (implemented in C++) for initialization
# of the fake process group; one can either use FakeStore or HashStore.
__all__ = ["FakeProcessGroup", "FakeStore"]


def _invalid_argument(op: str, msg: str):
    return RuntimeError(f"FakeProcessGroup::{op}: {msg}")


def _check_split_sizes(split_sizes, tensor: torch.Tensor, group_size: int) -> None:
    if any(s < 0 for s in split_sizes):
        raise RuntimeError("Split sizes must be non-negative")
    if not split_sizes:
        if tensor.size(0) % group_size != 0:
            raise RuntimeError(
                "Tensor's dim 0 does not divide equally across group size"
            )
        return
    if len(split_sizes) != group_size:
        raise RuntimeError("Number of tensor splits not equal to group size")
    if sum(split_sizes) != tensor.size(0):
        raise RuntimeError("Split sizes doesn't match total dim 0 size")


def _flat_view(t: torch.Tensor) -> torch.Tensor:
    # A flat 1-D view over `t`'s storage, for element-wise segment copies.
    return t.as_strided((t.numel(),), (1,), t.storage_offset())


def _row_size(t: torch.Tensor) -> int:
    # Elements per row along dim 0. c10d split sizes count rows, not elements;
    # mirrors computeLengthsAndOffsets (Utils.hpp), including the empty dim 0.
    dim0 = t.size(0)
    return t.numel() // dim0 if dim0 else 1


def _split_segment(
    flat: torch.Tensor, split_sizes, index: int, world_size: int, row_size: int
) -> torch.Tensor:
    # The segment of `flat` belonging to peer `index`. An empty split list is
    # the c10d convention for an equal division.
    if not split_sizes:
        chunk = flat.numel() // world_size
        return flat.narrow(0, chunk * index, chunk)
    if len(split_sizes) < world_size:
        raise RuntimeError(
            f"FakeProcessGroup: split list has {len(split_sizes)} entries for "
            f"a world of {world_size}"
        )
    offset = sum(split_sizes[:index]) * row_size
    length = split_sizes[index] * row_size
    if offset + length > flat.numel():
        raise RuntimeError(
            f"FakeProcessGroup: split segment [{offset}, {offset + length}) "
            f"exceeds the {flat.numel()}-element buffer"
        )
    return flat.narrow(0, offset, length)


def _fill_by_tiling(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Fill `dst` by repeating `src`, or cut it short when `dst` is smaller.
    # Real backends write the whole slot, so never leave a tail uninitialized.
    dst_numel = dst.numel()
    src_numel = src.numel()
    if dst_numel == 0:
        return
    if src_numel == 0:
        # This peer contributes nothing, so the slot receives nothing.
        dst.zero_()
        return
    written = 0
    while written < dst_numel:
        n = min(src_numel, dst_numel - written)
        dst.narrow(0, written, n).copy_(src.narrow(0, 0, n))
        written += n


def _is_tensor_list(value, depth: int) -> bool:
    if depth == 0:
        return isinstance(value, torch.Tensor)
    return isinstance(value, (list, tuple)) and all(
        _is_tensor_list(v, depth - 1) for v in value
    )


def _keep_base_overloads(*depths: int):
    # Overriding a collective in Python shadows the convenience overloads
    # BackendBase binds under the same name (e.g. allreduce(tensor, op)).
    # Calls that do not match the canonical signature (tensor lists nested
    # `depths` deep, then opts) go to the binding, which raises pybind's
    # TypeError or re-enters here canonically; its Work is a plain Work.
    def decorator(method):
        base = getattr(dist.BackendBase, method.__name__)
        params = list(inspect.signature(method).parameters.values())[1:]
        names = [p.name for p in params[:-1]]
        default_opts = params[-1].default

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            bound = dict(zip((p.name for p in params), args), **kwargs)
            opts = bound.pop("opts", default_opts)
            if (
                len(args) <= len(params)
                and bound.keys() == set(names)
                and isinstance(opts, type(default_opts))
                and all(_is_tensor_list(bound[n], d) for n, d in zip(names, depths))
            ):
                return method(self, *args, **kwargs)
            return base(self, *args, **kwargs)

        return wrapper

    return decorator


_UNSET = object()

# Shared defaults, as in the C++ Backend bindings; collectives only read them.
_ALLTOALL_OPTS = AllToAllOptions()
_ALLGATHER_OPTS = AllgatherOptions()
_ALLREDUCE_COALESCED_OPTS = AllreduceCoalescedOptions()
_ALLREDUCE_OPTS = AllreduceOptions()
_BARRIER_OPTS = BarrierOptions()
_BROADCAST_OPTS = BroadcastOptions()
_GATHER_OPTS = GatherOptions()
_REDUCE_OPTS = ReduceOptions()
_REDUCE_SCATTER_OPTS = ReduceScatterOptions()
_SCATTER_OPTS = ScatterOptions()


class FakeProcessGroup(dist.BackendBase):
    """
    A backend that performs no communication and hallucinates collective
    results from local inputs, so a single process can simulate one rank.

    NOTE [FakeProcessGroup collective semantics]
    Collectives use deterministic single-process approximations. When output
    can be derived from local inputs, fake collectives copy those values into
    local outputs so tests do not consume uninitialized memory. For scatter on
    non-root ranks, the root's input list is unavailable, so the output tensor
    is left unchanged.

    NOTE [FakeProcessGroup output version counters]
    Real collective backends write outputs from kernels autograd never sees, so
    fake writes run below Autograd (which also allows copies into ``chunk()``
    views of tensors requiring grad). The version counter follows NCCL:
    ``_AutoDispatchBelowADInplaceOrView`` leaves it alone, as NCCL's kernels
    do; ``_AutoDispatchBelowAutograd`` bumps it where NCCL runs ``copy_`` (list
    allgather slots, the root's own scatter output and gather slot) and for
    ``allgather_coalesced``, which NCCL does not implement.

    NOTE [FakeProcessGroup uniform-rank simulation]
    With ``Options.simulate_uniform_ranks`` the group models a world in which
    every rank holds data identical to this one. That makes every collective
    well defined from local inputs alone, which the default approximations are
    not: they ignore the reduce op and fill all-to-all outputs without regard
    to the split structure. The cost is that ranks cannot diverge, so this
    cannot surface rank-divergence bugs. ``scatter`` on a non-root rank stays
    outside the contract, as does PRODUCT for sparse all-reduce.
    """

    class Options(dist.BackendBase.Options):
        """
        Options for FakeProcessGroup.

        ``fake_option`` is an arbitrary value for tests of options propagation.
        ``error_on_collective`` makes every collective raise.
        ``simulate_uniform_ranks``: see NOTE [FakeProcessGroup uniform-rank
        simulation].
        """

        # Reject misspelled fields like the C++ binding did.
        __slots__ = ("fake_option", "error_on_collective", "simulate_uniform_ranks")

        def __init__(self) -> None:
            super().__init__("fake")
            self.fake_option = 0
            self.error_on_collective = False
            self.simulate_uniform_ranks = False

    _options: "FakeProcessGroup.Options | None"

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "FakeProcessGroup cannot be constructed directly. Use "
            "torch.distributed.init_process_group(backend='fake') instead to "
            "ensure proper dispatch system integration."
        )

    @staticmethod
    def _create_internal(
        rank: int, world_size: int, options=_UNSET
    ) -> "FakeProcessGroup":
        # Omitted options get fresh defaults; an explicit None stays None.
        if options is _UNSET:
            options = FakeProcessGroup.Options()
        elif options is not None and not isinstance(options, FakeProcessGroup.Options):
            raise TypeError(
                f"options must be FakeProcessGroup.Options, got {type(options).__name__}"
            )
        if rank < 0 or rank >= world_size:
            raise RuntimeError(
                f"Cannot init process group where rank ({rank}) >= "
                f"world_size ({world_size})"
            )
        # Bypasses __init__, which rejects direct construction.
        pg = FakeProcessGroup.__new__(FakeProcessGroup)
        dist.BackendBase.__init__(pg, rank, world_size)
        pg._options = options
        return pg

    # None when the group was built without options, which callers (and
    # test_device_mesh) use to tell whether an override was supplied. C++
    # callers of getBackendOptions then get a base Backend.Options instead,
    # which split() converts.
    @property
    # pyrefly: ignore [bad-override]
    def options(self) -> "FakeProcessGroup.Options | None":
        return self._options

    @property
    def supports_splitting(self) -> bool:
        return True

    def getBackendName(self) -> str:
        return "fake"

    def set_timeout(self, timeout) -> None:
        # No real communication, so there is no timeout to configure.
        pass

    def split(self, store, ranks, opts) -> "FakeProcessGroup | None":
        # No split collective to join: non-members return None and members get
        # a fresh group ranked by their position in `ranks`.
        if self.rank() not in ranks:
            return None
        if type(opts) is dist.BackendBase.Options and opts.backend == "fake":
            # Base options (e.g. the getBackendOptions default for an
            # options-less group): inherit the fake fields from this group.
            fake_opts = (
                copy.copy(self._options)
                if self._options is not None
                else FakeProcessGroup.Options()
            )
            fake_opts._timeout = opts._timeout
            fake_opts.global_ranks_in_group = opts.global_ranks_in_group
            fake_opts.group_name = opts.group_name
            fake_opts.group_desc = opts.group_desc
            fake_opts.enable_reconfigure = opts.enable_reconfigure
            opts = fake_opts
        elif not isinstance(opts, FakeProcessGroup.Options):
            raise RuntimeError("opts not a FakeProcessGroup::Options.")
        return FakeProcessGroup._create_internal(
            ranks.index(self.rank()), len(ranks), opts
        )

    def _uniform_ranks(self) -> bool:
        return self._options is not None and self._options.simulate_uniform_ranks

    def _check_collective_error(self) -> None:
        if self._options is not None and self._options.error_on_collective:
            raise RuntimeError(
                "FakeProcessGroup collective operation error (error_on_collective=true)"
            )

    def _work(self, result) -> FakeWork:
        return FakeWork(result) if self._uniform_ranks() else FakeWork()

    def _apply_uniform_reduction(self, tensor: torch.Tensor, reduce_op) -> None:
        # Combine `size` identical contributions in place. Every reduce op has
        # a closed form once the contributions are known to be equal.
        size = self.size()
        # mul_/pow_ have no in-place form for bool; under c10d's
        # nonzero-is-true convention both reductions are the identity anyway.
        is_bool = tensor.dtype == torch.bool
        op = reduce_op.op
        if op == dist.ReduceOp.SUM:
            if not is_bool:
                tensor.mul_(size)
        elif op in (
            dist.ReduceOp.AVG,
            dist.ReduceOp.MIN,
            dist.ReduceOp.MAX,
            dist.ReduceOp.BAND,
            dist.ReduceOp.BOR,
        ):
            # Idempotent on equal operands.
            pass
        elif op == dist.ReduceOp.PRODUCT:
            # Overflow/saturation matches what a real backend would produce.
            if not is_bool:
                tensor.pow_(size)
        elif op == dist.ReduceOp.BXOR:
            # x ^ x == 0, so an even number of equal contributions cancels out.
            if size % 2 == 0:
                tensor.zero_()
        elif op == dist.ReduceOp.PREMUL_SUM:
            # Real NCCL PREMUL_SUM is float-only too.
            if not (tensor.is_floating_point() or tensor.is_complex()):
                raise RuntimeError(
                    "FakeProcessGroup: PREMUL_SUM requires a floating point or "
                    f"complex tensor, got {tensor.dtype}."
                )
            try:
                factor = reduce_op.factor
            except RuntimeError:
                raise RuntimeError(
                    "FakeProcessGroup: PREMUL_SUM was given without its scaling "
                    "factor. Build it with torch.distributed._make_nccl_premul_sum."
                ) from None
            if isinstance(factor, torch.Tensor):
                # Real backends co-locate the factor rather than failing.
                factor = factor.to(tensor.device)
            tensor.mul_(factor)
            tensor.mul_(size)
        else:
            raise RuntimeError(
                f"FakeProcessGroup: unrecognized reduce op {int(op)} under "
                "simulate_uniform_ranks."
            )

    def _uniform_reduce_all(self, tensors, reduce_op) -> FakeWork:
        if not self._uniform_ranks():
            return FakeWork()
        with torch._C._AutoDispatchBelowADInplaceOrView():
            for tensor in tensors:
                self._apply_uniform_reduction(tensor, reduce_op)
        return FakeWork(tensors)

    @_keep_base_overloads(1)
    def broadcast(self, tensors, opts=_BROADCAST_OPTS):
        self._check_collective_error()
        # Identity under either contract: every rank already holds the value.
        return self._work(tensors)

    @_keep_base_overloads(1)
    def allreduce(self, tensors, opts=_ALLREDUCE_OPTS):
        self._check_collective_error()
        return self._uniform_reduce_all(tensors, opts.reduceOp)

    def allreduce_sparse(self, tensors, opts=_ALLREDUCE_OPTS):
        self._check_collective_error()
        # pyrefly: ignore [missing-attribute]
        if self._uniform_ranks() and opts.reduceOp.op == dist.ReduceOp.PRODUCT:
            raise RuntimeError(
                "FakeProcessGroup: allreduce_sparse does not support PRODUCT "
                "under simulate_uniform_ranks."
            )
        return self._uniform_reduce_all(tensors, opts.reduceOp)

    def allreduce_coalesced(self, tensors, opts=_ALLREDUCE_COALESCED_OPTS):
        self._check_collective_error()
        return self._uniform_reduce_all(tensors, opts.reduceOp)

    @_keep_base_overloads(1)
    def reduce(self, tensors, opts=_REDUCE_OPTS):
        self._check_collective_error()
        # Real backends leave non-root tensors unspecified; mirror that.
        if self.rank() != opts.rootRank:
            return FakeWork()
        return self._uniform_reduce_all(tensors, opts.reduceOp)

    @_keep_base_overloads(2, 1)
    def allgather(self, output_tensors, input_tensors, opts=_ALLGATHER_OPTS):
        self._check_collective_error()
        with torch._C._AutoDispatchBelowAutograd():
            for tensor in output_tensors[0]:
                tensor.copy_(input_tensors[0])
        return self._work(output_tensors[0])

    def all_gather_single(self, output, input, opts=_ALLGATHER_OPTS):
        self._check_collective_error()
        with torch._C._AutoDispatchBelowADInplaceOrView():
            for chunk in output.chunk(self.size()):
                chunk.copy_(input)
        return self._work([output])

    def allgather_coalesced(self, output_lists, input_list, opts=_ALLGATHER_OPTS):
        self._check_collective_error()
        op = "allgather_coalesced"
        if not input_list:
            raise _invalid_argument(op, "requires non-empty input tensor list")
        if len(output_lists) != self.size():
            raise _invalid_argument(op, "output lists should be equal to world size")
        for output_list in output_lists:
            if len(output_list) != len(input_list):
                raise _invalid_argument(
                    op,
                    f"invalid output size: (expected length {len(input_list)}, "
                    f"got {len(output_list)})",
                )
        with torch._C._AutoDispatchBelowAutograd():
            for output_list in output_lists:
                for output, input in zip(output_list, input_list):
                    output.copy_(input)
        return self._work([t for output_list in output_lists for t in output_list])

    def all_gather_single_coalesced(self, outputs, inputs, opts=_ALLGATHER_OPTS):
        self._check_collective_error()
        if len(outputs) != len(inputs):
            raise _invalid_argument(
                "allgather_into_tensor_coalesced",
                "requires input/output tensor lists to have the same length",
            )
        with torch._C._AutoDispatchBelowADInplaceOrView():
            for output, input in zip(outputs, inputs):
                for chunk in output.chunk(self.size()):
                    chunk.copy_(input)
        return self._work(outputs)

    @_keep_base_overloads(2, 1)
    def gather(self, output_tensors, input_tensors, opts=_GATHER_OPTS):
        self._check_collective_error()
        op = "gather"
        size = self.size()
        if opts.rootRank < 0 or opts.rootRank >= size:
            raise _invalid_argument(op, f"invalid root rank: {opts.rootRank}")
        if len(input_tensors) != 1:
            raise _invalid_argument(op, "requires a single-element input tensor list")
        if self.rank() != opts.rootRank:
            if output_tensors:
                raise _invalid_argument(op, "requires empty output on non-root")
            return FakeWork()
        if len(output_tensors) != 1:
            raise _invalid_argument(
                op,
                "requires a single-element output list containing a list with "
                f"{size} tensors.",
            )
        if len(output_tensors[0]) != size:
            raise _invalid_argument(
                op,
                f"Incorrect output list size {len(output_tensors[0])}. Output "
                f"list size should be {size}, same as size of the process group.",
            )
        for r, tensor in enumerate(output_tensors[0]):
            # NCCL copy_s only the root's own slot; peers' arrive by ncclRecv.
            guard = (
                torch._C._AutoDispatchBelowAutograd()
                if r == self.rank()
                else torch._C._AutoDispatchBelowADInplaceOrView()
            )
            with guard:
                tensor.copy_(input_tensors[0])
        return self._work(output_tensors[0])

    @_keep_base_overloads(1, 2)
    def scatter(self, output_tensors, input_tensors, opts=_SCATTER_OPTS):
        self._check_collective_error()
        op = "scatter"
        size = self.size()
        if opts.rootRank < 0 or opts.rootRank >= size:
            raise _invalid_argument(op, f"invalid root rank: {opts.rootRank}")
        if len(output_tensors) != 1:
            raise _invalid_argument(op, "requires a single-element output tensor list")
        if self.rank() != opts.rootRank:
            if input_tensors:
                raise _invalid_argument(op, "requires empty input on non-root")
            return FakeWork()
        if len(input_tensors) != 1:
            raise _invalid_argument(
                op,
                "requires a single-element input list containing a list with "
                f"{size} tensors.",
            )
        if len(input_tensors[0]) != size:
            raise _invalid_argument(
                op,
                f"Incorrect input list size {len(input_tensors[0])}. Input "
                f"list size should be {size}, same as size of the process group.",
            )
        with torch._C._AutoDispatchBelowAutograd():
            output_tensors[0].copy_(input_tensors[0][self.rank()])
        return self._work(output_tensors)

    @_keep_base_overloads(1, 2)
    def reduce_scatter(self, output_tensors, input_tensors, opts=_REDUCE_SCATTER_OPTS):
        self._check_collective_error()
        op = "reduce_scatter"
        if len(output_tensors) != len(input_tensors):
            raise _invalid_argument(
                op, "requires input/output tensor lists to have the same length"
            )
        with torch._C._AutoDispatchBelowADInplaceOrView():
            for output, inputs in zip(output_tensors, input_tensors):
                if len(inputs) != self.size():
                    raise _invalid_argument(
                        op, "invalid input tensor list size, must be world size"
                    )
                output.copy_(inputs[self.rank()])
                if self._uniform_ranks():
                    self._apply_uniform_reduction(output, opts.reduceOp)
        return self._work(output_tensors)

    def _reduce_scatter_into(self, output, input, reduce_op) -> None:
        if input.numel() != output.numel() * self.size():
            raise RuntimeError(
                "input tensor must be the same size as output size times world size"
            )
        output.copy_(input.chunk(self.size())[self.rank()])
        if self._uniform_ranks():
            self._apply_uniform_reduction(output, reduce_op)

    def reduce_scatter_single(
        self, outputTensor, inputTensor, opts=_REDUCE_SCATTER_OPTS
    ):
        self._check_collective_error()
        with torch._C._AutoDispatchBelowADInplaceOrView():
            self._reduce_scatter_into(outputTensor, inputTensor, opts.reduceOp)
        return self._work([outputTensor])

    def reduce_scatter_single_coalesced(
        self, outputs, inputs, opts=_REDUCE_SCATTER_OPTS
    ):
        self._check_collective_error()
        if len(outputs) != len(inputs):
            raise _invalid_argument(
                "reduce_scatter_tensor_coalesced",
                "requires input/output tensor lists to have the same length",
            )
        with torch._C._AutoDispatchBelowADInplaceOrView():
            for output, input in zip(outputs, inputs):
                self._reduce_scatter_into(output, input, opts.reduceOp)
        return self._work(outputs)

    def all_to_all_single(
        self,
        output_tensor,
        input_tensor,
        output_split_sizes,
        input_split_sizes,
        opts=_ALLTOALL_OPTS,
    ):
        self._check_collective_error()
        size = self.size()
        _check_split_sizes(input_split_sizes, input_tensor, size)
        _check_split_sizes(output_split_sizes, output_tensor, size)
        if not input_tensor.is_contiguous(
            memory_format=suggest_memory_format(input_tensor)
        ):
            raise RuntimeError("Input tensor must be contiguous")
        if not output_tensor.is_contiguous(
            memory_format=suggest_memory_format(output_tensor)
        ):
            raise RuntimeError("Output tensor must be contiguous")
        with torch._C._AutoDispatchBelowADInplaceOrView():
            flat_input = _flat_view(input_tensor)
            flat_output = _flat_view(output_tensor)
            if self._uniform_ranks():
                # Every peer holds what we hold, so the segment peer j sends to
                # us is our own input split at index rank. Slots may differ in
                # size from it (asymmetric splits, receive-only ranks), so tile
                # or truncate to fill the shape the caller asked for.
                mine = _split_segment(
                    flat_input,
                    input_split_sizes,
                    self.rank(),
                    size,
                    _row_size(input_tensor),
                )
                output_row_size = _row_size(output_tensor)
                for j in range(size):
                    slot = _split_segment(
                        flat_output, output_split_sizes, j, size, output_row_size
                    )
                    _fill_by_tiling(slot, mine)
                return FakeWork([output_tensor])
            # Inputs from other ranks are unavailable, so copy as much of the
            # local input as fits and zero the remainder.
            flat_output.zero_()
            copy_size = min(flat_input.numel(), flat_output.numel())
            if copy_size > 0:
                flat_output.narrow(0, 0, copy_size).copy_(
                    flat_input.narrow(0, 0, copy_size)
                )
        return FakeWork()

    def alltoall(self, output_tensor, input_tensor, opts=_ALLTOALL_OPTS):
        self._check_collective_error()
        op = "alltoall"
        size = self.size()
        if len(input_tensor) != size:
            raise _invalid_argument(
                op,
                f"input tensor list size {len(input_tensor)} does not match "
                f"world size {size}",
            )
        if len(output_tensor) != size:
            raise _invalid_argument(
                op,
                f"output tensor list size {len(output_tensor)} does not match "
                f"world size {size}",
            )
        with torch._C._AutoDispatchBelowADInplaceOrView():
            if not self._uniform_ranks():
                for output, input in zip(output_tensor, input_tensor):
                    output.copy_(input)
                return FakeWork()
            # Every peer sends us its entry at our index, which equals our own
            # entry at index rank. _flat_view walks storage linearly, so reject
            # non-contiguous tensors rather than corrupt them silently.
            mine = input_tensor[self.rank()]
            if not mine.is_contiguous():
                raise _invalid_argument(
                    op,
                    "input tensor must be contiguous under simulate_uniform_ranks",
                )
            for output in output_tensor:
                if not output.is_contiguous():
                    raise _invalid_argument(
                        op,
                        "output tensor must be contiguous under simulate_uniform_ranks",
                    )
                # Shapes need not match, so tile.
                _fill_by_tiling(_flat_view(output), _flat_view(mine))
        return FakeWork(output_tensor)

    def send(self, tensors, dstRank, tag):
        return FakeWork()

    def recv(self, tensors, srcRank, tag):
        return FakeWork()

    def recv_anysource(self, tensors, tag):
        return FakeWork()

    def start_coalescing(self) -> None:
        pass

    def end_coalescing(self):
        self._check_collective_error()
        return FakeWork()

    def barrier(self, opts=_BARRIER_OPTS):
        self._check_collective_error()
        return FakeWork()


def _create_fake_pg(common_opts, backend_opts):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convenient tool when playing
    with distributed but don't care about the actual data.
    """
    # new_group() inherits the default backend name but not its options. Keep
    # an explicitly selected fake-world contract consistent across subgroups.
    if backend_opts is None and dist.is_initialized():
        default_pg = dist.distributed_c10d._get_default_group()
        for device in default_pg._device_types:
            default_backend = default_pg._get_backend(device)
            if not isinstance(default_backend, FakeProcessGroup):
                continue
            default_options = default_backend.options
            if default_options is not None and default_options.simulate_uniform_ranks:
                backend_opts = FakeProcessGroup.Options()
                backend_opts.simulate_uniform_ranks = True
                break

    return FakeProcessGroup._create_internal(
        common_opts.group_rank, common_opts.group_size, backend_opts
    )


def _register_fake_backend() -> None:
    dist.Backend.register_backend(
        dist.Backend.FAKE,
        _create_fake_pg,
        extended_api=True,
        devices=["cpu", "cuda", "hpu", "xpu"],
    )
