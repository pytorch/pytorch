# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import auto, Enum
from typing import (  # type: ignore[attr-defined]
    _eval_type,
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)


T = TypeVar("T", bound=NamedTuple)


class Ref(Generic[T]):
    pass


class TypeInfo(NamedTuple):
    name: str
    fields: List[Tuple[str, Type]]  # type: ignore[type-arg]

    @classmethod
    def from_type(cls, c: T) -> "TypeInfo":
        if hasattr(c, "__name__"):
            name = c.__name__
        else:
            name = str(c)
        return cls(
            name,
            [(f, _eval_type(c.__annotations__[f], globals(), {})) for f in c._fields],
        )


"""
Schema for flat DB

TODO schemas not yet implemented
# threads as recorded at termination of process
Threads
    id: int
    traceback_id: int
    process_id: int

Process:
    id: int # Same as world groups RANK
    pid: int
    hostname: str

NCCLOp:
    # nccl op implementation details (sends/recv)
    id: int
    nccl_call_id: int

"""


class Group(NamedTuple):
    id: int
    desc: str
    size: int


class Membership(NamedTuple):
    group_id: Ref[Group]
    global_rank: int


class Traceback(NamedTuple):
    id: int
    frames: str


class Collective(NamedTuple):
    id: int
    group_id: Ref[Group]


class NCCLCall(NamedTuple):
    id: int
    collective_id: Ref[Collective]
    group_id: Ref[Group]
    global_rank: int  # technically Ref[Process] once we have it
    traceback_id: Ref[Traceback]
    collective_type: str
    sizes: List[List[int]]


class Database(NamedTuple):
    groups: List[Group]
    memberships: List[Membership]
    tracebacks: List[Traceback]
    collectives: List[Collective]
    ncclcalls: List[NCCLCall]


# TODO: We need to add a schema for the following
types = [
    TypeInfo.from_type(t)  # type: ignore[type-var]
    for t in globals().values()
    if (
        isinstance(t, type)
        and issubclass(t, tuple)
        and hasattr(t, "_fields")
        and t is not TypeInfo
    )
]

"""
Stacktrace cache
TODO
"""


"""
Collective Matching logic

NOTE: For now, these collectives need to be supported by NCCL,
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html.
"""
COLLECTIVES = {
    "broadcast",
    "all_gather",
    "all_reduce",
    "_all_gather_base",
    "all_gather_into_tensor_coalesced",
    "reduce_scatter_tensor_coalesced",
    "_reduce_scatter_base",
    "gather",
    "scatter",
    "all_to_all",
}

P2P = {
    "send",
    "recv",
}


class MatchState(Enum):
    """
    Enum representing the possible states of matching for collective operations.

    - FULLY_MATCHED: Indicates that all aspects of the collective operations match.
    - COLLECTIVE_TYPE_MISMATCH: The types of the collective operations differ.
    - SIZE_OR_SYNTAX_MISMATCH: There is a mismatch in input/output sizes or violation of collective syntax.
    - COLLECTIVE_STATE_MISMATCH:
        The states of the collective not same, such as one finished while another just started or scheduled.
    - COLLECTIVE_DTYPE_MISMATCH: The data types of the collective input/output differ.
    - UNDECIDED:
        The match status is ambiguous or cannot be determined, e.g., we might need to check all ranks for alltoall_base.
    """

    FULLY_MATCHED = auto()
    COLLECTIVE_TYPE_MISMATCH = auto()
    SIZE_OR_SYNTAX_MISMATCH = auto()
    COLLECTIVE_STATE_MISMATCH = auto()
    COLLECTIVE_DTYPE_MISMATCH = auto()
    UNDECIDED = auto()

    def __call__(self, culprit: Optional[str] = None) -> "MatchState":
        # Make the enum instance callable to add culprit.
        self.culprit = culprit
        return self

    def __str__(self) -> str:
        return f"Error type: {self.name}, Detail finding {self.culprit if self.culprit else ''}"


class Op:
    """Parses relevant info about operation out of 'event' dict

    examples of supported `profiling_name`s:
        nccl:broadcast
        nccl:send 1->2
        nccl:recv 3<-0
    """

    def __init__(self, event: Dict[Any, Any], memberships: Dict[str, Set[Any]]):
        profiling_name = event["profiling_name"]
        nccl, name = profiling_name.split(":")
        assert nccl == "nccl", f"name formatting error? {nccl} != 'nccl'"
        parts = name.split(" ")
        type = parts[0]
        meta = parts[1] if len(parts) == 2 else None
        self.state = event["state"]
        self.pg_name, _ = event["process_group"]
        assert type in COLLECTIVES | P2P | {
            "coalesced"
        }, f"{type} is not a supported operation"
        self.type = type
        if type == "send":
            s, d = meta.split("->")
            self._src, self._dst = int(s), int(d)
        elif type == "recv":
            d, s = meta.split("<-")
            self._dst, self._src = int(d), int(s)
        else:
            self._src, self._dst = -1, -1
        pg_name, pg_desc = event["process_group"]
        self._init_global_src_dst(memberships[pg_name])
        self.pg_size = len(memberships[pg_name])
        if type in P2P | COLLECTIVES:
            self.input_sizes = event["input_sizes"]
            self.output_sizes = event["output_sizes"]
        else:
            self.input_sizes, self.output_sizes = None, None
        self.collective_seq_id = event["collective_seq_id"]
        self.p2p_seq_id = event["p2p_seq_id"]
        self.input_dtypes = event["input_dtypes"]
        self.output_dtypes = event["output_dtypes"]

    def _init_global_src_dst(self, pg_ranks: Set[Any]) -> None:
        pg_ranks = sorted(pg_ranks)
        self._src_g = pg_ranks[self._src] if self._src is not None else None
        self._dst_g = pg_ranks[self._dst] if self._dst is not None else None

    @property
    def src(self) -> int:
        assert self.type in P2P, "can't get src of non-p2p op"
        return self._src

    @property
    def dst(self) -> int:
        assert self.type in P2P, "can't get dst of non-p2p op"
        return self._dst

    def __repr__(self) -> str:
        if self.type in P2P:
            return (
                f"{self.type}(s={self._src_g} d={self._dst_g}, sz={self.input_sizes})"
            )
        return f"{self.type}(input_sizes={self.input_sizes}, {self.state})"

    def match(self, other: "Op") -> MatchState:
        # TODO: I think this can validly not match,
        # e.g. if one PG was used for p2p ops between only some of the peers?
        # if self.seq_id != other.seq_id:
        # return False

        if self.type == "send":
            # TODO: We need more states for p2p ops.
            return (
                MatchState.FULLY_MATCHED
                if (
                    other.type == "recv"
                    and self.src == other.src
                    and self.dst == other.dst
                    and self.input_sizes == other.output_sizes
                )
                else MatchState.SIZE_OR_SYNTAX_MISMATCH
            )
        elif self.type == "recv":
            return (
                MatchState.FULLY_MATCHED
                if (
                    other.type == "send"
                    and self.src == other.src
                    and self.dst == other.dst
                    and self.output_sizes == other.input_sizes
                )
                else MatchState.SIZE_OR_SYNTAX_MISMATCH
            )
        elif self.type in COLLECTIVES:
            if self.type != other.type:
                return MatchState.COLLECTIVE_TYPE_MISMATCH(
                    f"Type '{self.type}' and '{other.type}' do not match"
                )
            if self.state != other.state:
                # MatchState()
                return MatchState.COLLECTIVE_STATE_MISMATCH(
                    f"States '{self.state}' '{other.state}' do not match"
                )
            if (
                other.input_dtypes != other.output_dtypes
                or self.input_dtypes != other.input_dtypes
                or self.output_dtypes != other.output_dtypes
            ):
                return MatchState.COLLECTIVE_DTYPE_MISMATCH(
                    f"Dtypes '{self.input_dtypes}/{other.input_dtypes}' '{self.output_dtypes}/{other.output_dtypes}' do not match"
                )
            if self.type == "all_to_all":
                return MatchState.UNDECIDED
            if self.type != "scatter" and self.input_sizes != other.input_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Input sizes '{self.input_sizes}' '{other.input_sizes}' do not match"
                )
            if self.type != "gather" and self.output_sizes != other.output_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Output sizes '{self.output_sizes}' '{other.output_sizes}' do not match"
                )
            if self.type == "all_reduce" and self.input_sizes != other.output_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Input sizes '{self.input_sizes}' do not match output sizes '{other.output_sizes}'"
                )
            # TODO: need to consider uneven sharding for all-gather.
            # TODO: need to consider all_gather_into_tensor_coalesced (coalesced related)
            if self.type in [
                "all_gather",
                "all_gather_base",
            ] and not (
                math.prod(other.output_sizes[0])
                == math.prod(self.input_sizes[0]) * self.pg_size
            ):
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Input numel '{math.prod(other.input_sizes[0])} * pg size {self.pg_size}' "
                    f"do not match output numel '{math.prod(other.output_sizes[0])}'",
                )
            if self.type in [
                "reduce_scatter",
                "_reduce_scatter_base",
            ] and not (
                math.prod(other.input_sizes[0])
                == math.prod(self.output_sizes[0]) * self.pg_size
            ):
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Input numel '{math.prod(other.input_sizes[0])}' do not match output numel "
                    f"'{math.prod(other.output_sizes[0])} * pg size {self.pg_size}'",
                )
        elif self.type == "coalesced":
            return (
                MatchState.FULLY_MATCHED
                if (other.type == "coalesced")
                else MatchState.SIZE_OR_SYNTAX_MISMATCH
            )
        return MatchState.FULLY_MATCHED
