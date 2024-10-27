from __future__ import annotations

import copy
import dataclasses
import enum
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import DefaultDict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
from typing_extensions import Self

import torch.compiler.config
from torch._dynamo.utils import get_chromium_event_logger
from torch._logging._internal import trace_structured_artifact


if TYPE_CHECKING:
    import types

    from torch._dynamo.symbolic_convert import InstructionTranslator


log = logging.getLogger(__name__)

LOCK_TIMEOUT = 10

# How does in memory representation work?  Concretely, this module is
# responsible for holding GLOBAL state representing the state it holds, no
# other copies permitted.  So we retire frame_state entirely and store it
# here.  This should be reset when Dynamo is reset.  We never GC information
# (similar to how the filesystem doesn't get cleaned up except by tmp
# cleaner), so the expectation is the information is relatively cheap and we
# don't mind leaking it.


# How does the filesystem/remote cache work?  Here are the extra knobs:
#
# - WORKFLOW_ID: Do we have a unique identifier for the "training run"  (such that it
#   stays the same if we're running the same code, and changes if we're
#   running something different).
#
# - RANK_SHARING: Are we sharing the cache across ranks, or does each rank get
#   an individual cache?
#
# With no WORKFLOW_ID, we don't enable PGO cache by default.  This is to prevent
# situations where unrelated invocations of PyTorch unpredictably cause
# changes to each other's behavior.  With a WORKFLOW_ID, at least you know there
# is some "state" associated with it.  (State dict might be another way to
# tell if a run is related or not.)  You can opt-in to YOLO everything
# aliases everything by passing a shared WORKFLOW_ID for all your invocations.
#
# So cache is per WORKFLOW_ID.  With no RANK_SHARING, there is never contention
# between runs, so we can leisurely update a bundle with information we need.
# Because we are grouped by WORKFLOW_ID, we can have a single consolidated bundle
# for everything (or not; maybe worry about O(n^2) IO if we updated every
# compile--let's just instrument this.)  Can even take a filelock for extra
# safety (expect no contention); expect 50ns overhead from uncontended filelock.
#
# With RANK_SHARING, everyone is storming to modify the same cache files.
# We can do this by having folks atomic write to a CAS-store and then having
# readers do on-the-fly merging (this can be implemented in remote using
# prefix iteration).  As an optional optimization, one rank can be elected to
# handling bundling post facto (ideally, this is done async, after quiescence,
# without compiler collective need to wait for everyone to finish writing
# their bits.) Not sure how you can avoid a listdir because if some rank shows
# up with some new entries we need to pull them in ASAP (unless you want to
# delay bundling).
#
# With compiler collective, life is easier.  Compiler chat with each other so
# rank 0 has collected everything.  So elect rank 0 only to write the bundle.
# Don't even need CAS-store atomic write; just one rank writing an updating
# bundles.  So maybe don't bother with RANK_SHARING in that case.


@dataclasses.dataclass(frozen=True)
class CodeId:
    filename: str
    firstlineno: int
    name: str

    @staticmethod
    def make(code: types.CodeType) -> CodeId:
        return CodeId(code.co_filename, code.co_firstlineno, code.co_name)


@dataclasses.dataclass
class CodeState:
    automatic_dynamic: DefaultDict[str, FrameStateSizeEntry] = dataclasses.field(
        default_factory=lambda: defaultdict(FrameStateSizeEntry)
    )


_INIT_CODE_STATE: Optional[DefaultDict[CodeId, CodeState]] = None
_CODE_STATE: Optional[DefaultDict[CodeId, CodeState]] = None


@dataclasses.dataclass(frozen=True)
class InferStride:
    """
    Denotes the quantity stride[dim] * size[dim], which is what the stride would
    be for the next physical dimension that results in a contiguous layout.

    For example, given size = [2, 3], stride = [3, 1], we can replace this with
    stride = [InferStride(1), 1], because InferStride(1) = stride[1] * size[1] = 1 * 3 = 3

    Indirecting the representation in this way is important for the join operation
    on strides as if we join [2, 3][3, 1] and [2, 4][4, 1],
    we don't want [2, None][None, 1] which would get eventually symbolized into
    [2, s0][s1, 1] (notice that the relationship between s0 and s1 is broken).
    If we instead rewrite the expressions as InferStride so we have [2, 3][InferStride(1), 1]
    and [2, 4][InferStride(1), 1] we now join to [2, None][InferStride(1), 1] will
    result in [2, s0][s0, 1], as desired.
    """

    dim: int


_T = TypeVar("_T")


class AutoUnset(enum.Enum):
    """
    The identity element of our semilattice, a generic "don't know" element that
    is always subsumed when we get more information.
    """

    token = 0


auto_unset = AutoUnset.token


class AutoDynamic(enum.Enum):
    """
    The top element of our (bounded) semilattice, whenever you merge this with
    any other element you always get it again
    """

    token = 0


auto_dynamic = AutoDynamic.token


@dataclasses.dataclass
class FrameStateSizeEntry:
    scalar: Union[int, AutoDynamic, AutoUnset] = dataclasses.field(default=auto_unset)
    # NB: We don't have cases where we have a known dimensionality but
    # we know NOTHING about the individual sizes
    size: Union[
        AutoDynamic, AutoUnset, Tuple[Union[int, AutoDynamic], ...]
    ] = dataclasses.field(default=auto_unset)
    stride: Union[
        AutoDynamic, AutoUnset, Tuple[Union[int, AutoDynamic, InferStride], ...]
    ] = dataclasses.field(default=auto_unset)

    def is_size_dynamic(self, dim: int) -> bool:
        if self.size is auto_dynamic:
            return True
        if self.size is auto_unset:
            return False
        return self.size[dim] is auto_dynamic

    def is_stride_dynamic(self, dim: int) -> bool:
        # At the moment, dynamic strides is a bit buggy.  Good test case
        # here is `PYTORCH_TEST_WITH_DYNAMO=1 python test/test_autograd.py
        # TestAutograd.test_gradcheck_jacobian_mismatch`
        #
        # This if statement preserves historical behavior, which is that we
        # ONLY make strides dynamic if the size is exactly static everywhere.
        # We could potentially relax this but in general we should be very
        # careful about when to infer dynamic strides.
        #
        # Actually, the existing algorithm is already somewhat problematic.
        # Suppose a tensor that is sometimes:
        # f32[2, 3, 5][15, 5, 1] and other times
        # f32[2, 3, 5][5, 10, 1] (specifically, dim 0 and 1 are physically transposed).
        # If we infer strides should be (DYNAMIC, DYNAMIC, 1).  But this is
        # silly: we really should have just guarded on dim order.
        if not (
            isinstance(self.size, tuple) and all(type(s) is int for s in self.size)
        ):
            return False
        if self.stride is auto_dynamic:
            return True
        if self.stride is auto_unset:
            return False
        return self.stride[dim] is auto_dynamic

    @staticmethod
    def make_scalar(x: int) -> FrameStateSizeEntry:
        return FrameStateSizeEntry(scalar=x, size=auto_dynamic, stride=auto_dynamic)

    # NB: steals the inputs
    @staticmethod
    def make_tensor(
        size: Tuple[int, ...], stride: Tuple[int, ...]
    ) -> FrameStateSizeEntry:
        return FrameStateSizeEntry(scalar=auto_dynamic, size=size, stride=stride)

    @staticmethod
    def _merge_atom(x: _T, y: _T) -> Union[AutoDynamic, _T]:
        if x is auto_unset:
            return y
        if y is auto_unset:
            return x
        if x is auto_dynamic or y is auto_dynamic or x != y:
            return auto_dynamic
        return x

    @classmethod
    def _merge_atom_tup(
        cls,
        xs: Union[AutoDynamic, AutoUnset, Tuple[_T, ...]],
        ys: Union[AutoDynamic, AutoUnset, Tuple[_T, ...]],
    ) -> Union[AutoDynamic, AutoUnset, Tuple[Union[AutoDynamic, _T], ...]]:
        if xs is auto_unset:
            return ys
        if ys is auto_unset:
            return xs
        if xs is auto_dynamic or ys is auto_dynamic:
            return auto_dynamic
        if len(xs) != len(ys):
            return auto_dynamic
        return tuple(cls._merge_atom(x, y) for x, y in zip(xs, ys))

    def __ior__(self, other: Self) -> Self:
        self.scalar = self._merge_atom(self.scalar, other.scalar)
        self.size = self._merge_atom_tup(self.size, other.size)
        self.stride = self._merge_atom_tup(self.stride, other.stride)
        return self


def update_automatic_dynamic(
    tx: InstructionTranslator,
    name: str,
    entry: FrameStateSizeEntry,
    *,
    is_unspecialized_nn_module: bool = False,
) -> FrameStateSizeEntry:
    code_id = CodeId.make(tx.f_code)
    frame_state = get_code_state()[code_id]
    is_update = name in frame_state.automatic_dynamic
    mut_entry = frame_state.automatic_dynamic[name]
    old_entry = copy.copy(mut_entry)
    mut_entry |= entry

    # Do some logs (damn, I spend more code logging than I do actually doing
    # the updates lol)
    if is_update and old_entry.scalar != mut_entry.scalar:
        log.debug(
            "automatic dynamic int %s val %s != %s",
            name,
            entry.scalar,
            old_entry.scalar,
        )
        get_chromium_event_logger().log_instant_event(
            "automatic_dynamic",
            time.time_ns(),
            {
                "name": name,
                "dim_changed": "scalar",
                "reason": "scalar change",
                "cached": str(old_entry.scalar),
                "new": str(entry.scalar),
            },
        )
        if is_unspecialized_nn_module:
            log.info(
                "%s is converted to a symbolic integer. It is an attribute of a "
                "user defined nn module class. If you wish to keep it static, you can "
                "mark the nn module class as `torch._dynamo.mark_static`.",
                name,
            )

    def log_tup(
        tup_name: str, short_reason: str, long_reason: str, i: Optional[int] = None
    ) -> None:
        entry_tup = (
            getattr(entry, tup_name) if i is None else getattr(entry, tup_name)[i]
        )
        old_entry_tup = (
            getattr(old_entry, tup_name)
            if i is None
            else getattr(old_entry, tup_name)[i]
        )
        log.debug(
            "automatic dynamic %s %s %s %s != %s",
            tup_name,
            name,
            short_reason,
            # NB: We used to only report len(...) here for dim mismatch
            entry_tup,
            old_entry_tup,
        )
        get_chromium_event_logger().log_instant_event(
            "automatic_dynamic",
            time.time_ns(),
            {
                "name": name,
                "dim_changed": "all" if i is None else i,
                "reason": long_reason,
                "cached": str(old_entry_tup),
                "new": str(entry_tup),
            },
        )

    if is_update and old_entry.size != mut_entry.size:
        if isinstance(old_entry.size, tuple) and isinstance(entry.size, tuple):
            if len(old_entry.size) != len(entry.size):
                log_tup("size", "dim", "dimensionality change")
            else:
                for i in range(len(entry.size)):
                    if old_entry.size[i] != entry.size[i]:
                        log_tup("size", f"size({i})", "size change", i)
        else:
            log_tup("size", "other", "other")

    if is_update and old_entry.stride != mut_entry.stride:
        if isinstance(old_entry.stride, tuple) and isinstance(entry.stride, tuple):
            if len(old_entry.stride) != len(entry.stride):
                log_tup("stride", "dim", "dimensionality change")
            else:
                for i in range(len(entry.stride)):
                    if old_entry.stride[i] != entry.stride[i]:
                        log_tup("stride", f"stride({i})", "stride change", i)
        else:
            log_tup("stride", "other", "other")

    return mut_entry


def process_automatic_dynamic(
    tx: InstructionTranslator,
    name: str,
    entry: FrameStateSizeEntry,
    *,
    is_unspecialized_nn_module: bool = False,
) -> FrameStateSizeEntry:
    if (st := tx.distributed_state) is None:
        return update_automatic_dynamic(
            tx,
            name,
            entry,
            is_unspecialized_nn_module=is_unspecialized_nn_module,
        )
    elif st.all_states is None:
        # Preflight, always pretend as if it's static.  The point here
        # is we want to get through the preflight quickly, and static
        # will run faster.  The preexisting frame state will get
        # applied anyway after we do compiler collectives.
        # TODO: I'm not sure if we should just bong the entire pgo
        # state here, it kind of depends if we're going to have other
        # things that talk in compiler collective.  Also, the PGO
        # state, if we've already inferred something is automatic
        # dynamic, will have lost the actual input sizes, which might
        # be useful for debugging purposes (e.g., observing 0/1
        # specialization).  Bonging the entire PGO state here would
        # let us delete this logic here; the compiler collective
        # would just directly update_automatic_dynamic
        st.local_state.automatic_dynamic[name] = entry
        return entry
    else:
        # Apply the updates.  NB: all_states includes the local state
        # too.
        res = None
        for sub_state in st.all_states:
            if name in sub_state.automatic_dynamic:
                res = update_automatic_dynamic(
                    tx,
                    name,
                    sub_state.automatic_dynamic[name],
                    is_unspecialized_nn_module=is_unspecialized_nn_module,
                )
        assert res is not None
        return res


def code_state_path() -> Optional[str]:
    workflow_id = torch.compiler.config.workflow_id
    if workflow_id is None:
        return None

    from torch._inductor.runtime.runtime_utils import cache_dir

    return os.path.join(cache_dir(), "dynamo", f"code_state_{workflow_id}.pkl")


def get_code_state() -> DefaultDict[CodeId, CodeState]:
    global _CODE_STATE, _INIT_CODE_STATE
    if _CODE_STATE is None:
        path = code_state_path()

        _CODE_STATE = defaultdict(CodeState)

        if path is not None and os.path.exists(path):
            # Read lock not necessary as we always write atomically write to
            # the actual location
            with open(path, "rb") as f:
                try:
                    _CODE_STATE = pickle.load(f)
                except Exception:
                    log.warning("get_code_state failed while reading %s", path)
                else:
                    assert isinstance(_CODE_STATE, defaultdict)
                    log.info(
                        "get_code_state %s hit, %d entries", path, len(_CODE_STATE)
                    )
                    trace_structured_artifact(
                        "get_code_state",
                        "string",
                        lambda: repr(_CODE_STATE),
                    )
                    _INIT_CODE_STATE = copy.deepcopy(_CODE_STATE)
        else:
            log.info("get_code_state %s not found", path)

    assert _CODE_STATE is not None

    return _CODE_STATE


def put_code_state() -> None:
    path = code_state_path()
    if path is None:
        log.info("put_code_state: no workflow_id, disabled")
        return

    if _CODE_STATE is None:
        log.info("put_code_state: never initialized, will not write")
        return

    if _CODE_STATE == _INIT_CODE_STATE:
        log.info("put_code_state: no change, skipping")
        return

    tmp_path = path + ".tmp"
    lock_path = path + ".lock"
    # We /mostly/ don't need the lock but the tmp file could be clobbered
    # TODO: use a safe tempfile create to eliminate lock
    from filelock import FileLock

    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        with open(tmp_path, "wb") as f:
            pickle.dump(_CODE_STATE, f)
        os.rename(tmp_path, path)
        log.info("put_code_state: wrote %s, %d entries", path, len(_CODE_STATE))
        trace_structured_artifact(
            "put_code_state",
            "string",
            lambda: repr(_CODE_STATE),
        )


# NB: this does NOT reset the cached code state on disk
def reset_code_state() -> None:
    global _CODE_STATE, _INIT_CODE_STATE
    _CODE_STATE = None
    _INIT_CODE_STATE = None
