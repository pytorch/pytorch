from __future__ import annotations

import copy
import dataclasses
import enum
import logging
import time
from typing import Optional, Tuple, TYPE_CHECKING, TypeVar, Union
from typing_extensions import Self

from torch._dynamo.utils import get_chromium_event_logger


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


log = logging.getLogger(__name__)


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
    is_update = name in tx.output.frame_state
    mut_entry = tx.output.frame_state.setdefault(name, FrameStateSizeEntry())
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
