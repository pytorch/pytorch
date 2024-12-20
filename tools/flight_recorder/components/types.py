# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
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

from tools.flight_recorder.components.fr_logger import FlightRecorderLogger


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
        details = f", {self.culprit}" if getattr(self, "culprit", None) else ""
        return f"Error type: {self.name}{details}"


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
    id: str
    desc: str
    size: int


class Membership(NamedTuple):
    group_id: str
    global_rank: int


class Traceback(NamedTuple):
    id: int
    frames: str


class Collective(NamedTuple):
    id: int
    group_id: str
    pass_check: bool
    collective_seq_id: int
    p2p_seq_id: int
    record_id: int
    pg_desc: str
    collective_name: str
    input_sizes: List[List[int]]
    output_sizes: List[List[int]]
    expected_ranks: Set[int]
    collective_state: str
    collective_frames: List[Dict[str, str]]
    input_numel: Optional[int] = None
    output_numel: Optional[int] = None
    missing_ranks: Optional[Set[int]] = None
    mismatch_collectives: Optional[Dict[int, "Collective"]] = None
    type_of_mismatch: Optional[MatchState] = None


class NCCLCall(NamedTuple):
    id: int
    collective_id: Ref[Collective]
    group_id: str
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
    "all_reduce_barrier",
}

P2P = {
    "send",
    "recv",
}


class EntryState:
    """
    Util class to keep track of the state of an entry and standardize the way we
    log the error info during analysis.
    """

    def __init__(self, entry: Dict[str, Any], expected_ranks: Set[int]) -> None:
        self.pg_name = entry["process_group"][0]
        self.desc = entry["process_group"][1]
        self.pg_desc = (
            f"{self.pg_name}:{self.desc}" if self.desc != "undefined" else self.pg_name
        )
        self.profiling_name = entry["profiling_name"]
        self.collective_seq_id = entry["collective_seq_id"]
        self.p2p_seq_id = entry["p2p_seq_id"]
        self.record_id = entry["record_id"]
        self.input_sizes = entry["input_sizes"]
        self.output_sizes = entry["output_sizes"]
        self.collective_state = entry["state"]
        self.collective_frames = entry["frames"]
        self.expected_ranks = expected_ranks
        self.missing_ranks: Set[int]
        self.input_numel: int
        self.output_numel: int
        self.errors: Set[Tuple[int, MatchState]]

    def log(
        self,
        logger: FlightRecorderLogger,
        logger_msg: str,
        frame_formatter: Any,
        total_numel: Optional[Tuple[int, int]] = None,
        errors: Optional[Set[Tuple[int, MatchState]]] = None,
        missing_ranks: Optional[Set[int]] = None,
    ) -> None:
        logger.info(
            logger_msg,
            self.collective_seq_id,
        )
        logger.info("internal record id: %s", self.record_id)
        logger.info("group info: %s", self.pg_desc)
        logger.info("collective: %s", self.profiling_name)
        if missing_ranks:
            self.missing_ranks = missing_ranks
            logger.info("missing ranks: %s", missing_ranks)
        if total_numel:
            self.input_numel = total_numel[0]
            self.output_numel = total_numel[1]
            logger.info("total input numel: %d", total_numel[0])
            logger.info("total output numel: %d", total_numel[1])
        logger.info("input sizes: %s", self.input_sizes)
        logger.info("output sizes: %s", self.output_sizes)
        logger.info("world size: %d", len(self.expected_ranks))
        logger.info("expected ranks: %s", str(self.expected_ranks))
        logger.info("collective state: %s", self.collective_state)
        if errors:
            self.errors = errors
            error_msg = ", ".join(
                f"Culprit rank {error[0]}; {str(error[1])}" for error in errors
            )
            logger.info("error msg: %s", error_msg)
        logger.info(
            "collective stack trace: \n %s", frame_formatter(self.collective_frames)
        )

    def to_collective(
        self,
        id: int,
        errors: Optional[Set[Tuple[int, MatchState]]] = None,
        idx_map: Optional[Dict[int, int]] = None,
        all_entries: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> Collective:
        if not errors:
            return Collective(
                id=id,
                group_id=self.pg_name,
                record_id=self.record_id,
                pg_desc=self.pg_desc,
                pass_check=True,
                collective_seq_id=self.collective_seq_id,
                p2p_seq_id=self.p2p_seq_id,
                collective_name=self.profiling_name,
                input_sizes=self.input_sizes,
                output_sizes=self.output_sizes,
                expected_ranks=self.expected_ranks,
                collective_state=self.collective_state,
                collective_frames=self.collective_frames,
                missing_ranks=getattr(self, "missing_ranks", None),
            )
        else:
            assert idx_map is not None, "idx_map is None"
            assert all_entries is not None, "all_entries is None"
            mismatch_collectives = {}
            for rank, error in errors:
                idx = idx_map[rank]
                entry = all_entries[rank][idx]
                desc = entry["process_group"][1]
                pg_name = entry["process_group"][0]
                mismatch_collectives[rank] = Collective(
                    id=id,
                    group_id=entry["process_group"][0],
                    record_id=entry["record_id"],
                    pg_desc=f"{pg_name}:{desc}" if desc != "undefined" else pg_name,
                    pass_check=False,
                    collective_seq_id=entry["collective_seq_id"],
                    p2p_seq_id=entry["p2p_seq_id"],
                    collective_name=entry["profiling_name"],
                    input_sizes=entry["input_sizes"],
                    output_sizes=entry["output_sizes"],
                    expected_ranks=self.expected_ranks,
                    collective_state=entry["state"],
                    collective_frames=entry["frames"],
                    type_of_mismatch=error,
                )
            return Collective(
                id=id,
                group_id=self.pg_name,
                record_id=self.record_id,
                pg_desc=self.pg_desc,
                pass_check=False,
                collective_seq_id=self.collective_seq_id,
                p2p_seq_id=self.p2p_seq_id,
                collective_name=self.profiling_name,
                input_sizes=self.input_sizes,
                output_sizes=self.output_sizes,
                expected_ranks=self.expected_ranks,
                collective_state=self.collective_state,
                collective_frames=self.collective_frames,
                input_numel=self.input_numel if hasattr(self, "input_numel") else None,
                output_numel=self.output_numel
                if hasattr(self, "output_numel")
                else None,
                missing_ranks=self.missing_ranks
                if hasattr(self, "missing_ranks")
                else None,
                mismatch_collectives=mismatch_collectives,
            )

    def to_nccl_call(
        self,
        all_entries: Dict[int, List[Dict[str, Any]]],
        idx_map: Dict[int, int],
        nccl_call_id: int,
        collective_id: Any,
    ) -> List[NCCLCall]:
        result = []
        for i, k in idx_map.items():
            all_entries[i].pop(k)
            result.append(
                NCCLCall(
                    id=nccl_call_id,
                    collective_id=collective_id,
                    group_id=self.pg_name,  # type: ignore[arg-type]
                    global_rank=i,
                    traceback_id=0,  # type: ignore[arg-type]
                    collective_type=self.profiling_name,
                    sizes=self.input_sizes,
                )
            )
            nccl_call_id += 1
        return result


class Op:
    """Parses relevant info about operation out of 'event' dict

    examples of supported `profiling_name`s:
        nccl:broadcast
        nccl:send 1->2
        nccl:recv 3<-0
    """

    def __init__(
        self, event: Dict[Any, Any], memberships: Dict[str, Set[Any]], pg_name: str
    ):
        self.profiling_name = event["profiling_name"]
        nccl, name = self.profiling_name.split(":")
        assert nccl == "nccl", f"name formatting error? {nccl} != 'nccl'"
        parts = name.split(" ")
        type = parts[0]
        meta = parts[1] if len(parts) == 2 else None
        self.state = event["state"]
        self.pg_name, self.pg_desc = event["process_group"]
        assert type in COLLECTIVES | P2P | {
            "coalesced"
        }, f"{type} is not a supported operation"
        self.type = type
        if type == "send":
            assert isinstance(meta, str)
            s, d = meta.split("->")
            self._src, self._dst = int(s), int(d)
        elif type == "recv":
            assert isinstance(meta, str)
            d, s = meta.split("<-")
            self._dst, self._src = int(d), int(s)
        else:
            self._src, self._dst = -1, -1
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
        self.time_created_ns = event["time_created_ns"]
        self.collective_frames = event["frames"]
        self.is_verbose = os.getenv("FR_TRACE_VERBOSE_OUTPUT", "0") == "1"

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
        p2p_info = ""
        if self.type in P2P:
            p2p_info = f"s={self._src_g} d={self._dst_g}"
        if self.is_verbose:
            verbose_info = (
                f"timestamp_created={self.time_created_ns}",
                p2p_info,
                f"input_sizes={self.input_sizes}",
                f"output_sizes={self.output_sizes}",
                f"input_dtypes={self.input_dtypes}",
                f"output_dtypes={self.output_dtypes}",
                "collective_seq_id | p2p_seq_id="
                f"{self.p2p_seq_id if self.type in P2P else self.collective_seq_id}",
                f"pg_name={self.pg_name}",
                f"pg_description={self.pg_desc}",
                f"pg_size={self.pg_size}",
                f"state={self.state}",
            )
            return f"{self.type}(%s)" % ", ".join(s for s in verbose_info if s)
        return f"{self.type}(%sinput_sizes={self.input_sizes}, state={self.state})" % (
            f"{p2p_info}, " if p2p_info else ""
        )

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
                    f"Expected collective type: '{self.type}' does not match found collective type: '{other.type}'"
                )
            if self.state != other.state:
                # MatchState()
                return MatchState.COLLECTIVE_STATE_MISMATCH(
                    f"Expected state: '{self.state}' does not match found state: '{other.state}'"
                )
            if (
                set(self.input_dtypes) != set(self.output_dtypes)
                or set(self.input_dtypes) != set(other.input_dtypes)
                or set(self.input_dtypes) != set(other.output_dtypes)
            ):
                return MatchState.COLLECTIVE_DTYPE_MISMATCH(
                    f"Expected dtypes: '{set(self.input_dtypes)}' does not "
                    f"match found dtype: '{set(self.output_dtypes)}/"
                    f"{set(other.input_dtypes)}/{set(other.output_dtypes)}'",
                )
            if self.type == "all_to_all":
                return MatchState.UNDECIDED
            if self.type != "scatter" and self.input_sizes != other.input_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Expected input sizes: '{self.input_sizes}' does not match found input sizes: "
                    f"'{other.input_sizes}'",
                )
            if self.type != "gather" and self.output_sizes != other.output_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Expected output sizes: '{self.output_sizes}' does not match found output sizes: "
                    f"'{other.output_sizes}'"
                )
            if self.type == "all_reduce" and self.input_sizes != other.output_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Expected input sizes: '{self.input_sizes}' does not match found output sizes: '{other.output_sizes}'"
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
                    f"Found input numel '{math.prod(other.input_sizes[0])} * pg size {self.pg_size}' "
                    f"does not match output numel '{math.prod(other.output_sizes[0])}'",
                )
            if self.type in [
                "reduce_scatter",
                "_reduce_scatter_base",
            ] and not (
                math.prod(other.input_sizes[0])
                == math.prod(self.output_sizes[0]) * self.pg_size
            ):
                return MatchState.SIZE_OR_SYNTAX_MISMATCH(
                    f"Found input numel '{math.prod(other.input_sizes[0])}' does not match output numel "
                    f"'{math.prod(other.output_sizes[0])} * pg size {self.pg_size}'",
                )
        elif self.type == "coalesced":
            return (
                MatchState.FULLY_MATCHED
                if (other.type == "coalesced")
                else MatchState.SIZE_OR_SYNTAX_MISMATCH
            )
        return MatchState.FULLY_MATCHED
