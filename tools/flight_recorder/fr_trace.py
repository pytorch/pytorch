#!/usr/bin/env python3
"""Flight Recorder Trace Analyzer

This script primarily merges data from individual flight recorder buffers from individual ranks in a
PyTorch Distributed program into a flattened database format that can be used for further analysis.

However as part of the merging process, it is necessary to perform some analysis in order to match operators
on one rank with corresponding operators on other ranks and register them as one 'collective' entry.  During this
process, a significant amount of useful information can already be extracted such as where the first mismatch occurs
in cases of desync (when not all ranks issue a compatible collective in a particular process group).


Not Yet Implemented
- TODO- tracebacks aren't implemented

Known Issues
- Flight Recorder buffer sequence_id information is not sufficient to match collectives and coalseced collectives
  unless we have the trace data from the beginning of the program.  To enable confident analysis of trace buffers that
  do not start from zero (and to simplify the script's matching logic) we need to add more information to the recorder.
- Currently, the script omits checking the 'status' of collectives.  We can look for the first 'non completed'
  collective easily enough and report that.

Usage
python fr_trace.py -d <dump dir containing trace files> [-o <output file>]

- Omitting the optional output file will still yield analysis information to stdout
- the output file is a pickle of the flat DB, which may change in format in the future.
"""

import argparse
import ast
import math
import os
import pickle
import sys
from enum import Enum
from typing import (  # type: ignore[attr-defined]
    _eval_type,
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import tabulate  # type: ignore[import-untyped]


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
    "alltoall_base",
    "alltoall",
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
    - UNDECIDED:
        The match status is ambiguous or cannot be determined, e.g., we might need to check all ranks for alltoall_base.
    """

    FULLY_MATCHED = 1
    COLLECTIVE_TYPE_MISMATCH = 2
    SIZE_OR_SYNTAX_MISMATCH = 3
    COLLECTIVE_STATE_MISMATCH = 4
    UNDECIDED = 5


def check_size_even_expand(list1: List[Any], list2: List[Any], size: int) -> bool:
    ratio = None
    for a, b in zip(list1, list2):
        current_ratio = int(a) / int(b)
        if current_ratio == 1:
            continue
        if current_ratio != size:
            return False
        elif ratio is None:
            ratio = current_ratio
        else:
            return False
    return True


def check_size_alltoall(alltoall_cases: List[Dict[str, Any]]) -> Tuple[bool, int, int]:
    input_numel = 0
    output_numel = 0
    for e in alltoall_cases:
        input_numel += math.prod(e["input_sizes"][0])
        output_numel += math.prod(e["output_sizes"][0])
    return input_numel == output_numel, input_numel, output_numel


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
                return MatchState.COLLECTIVE_TYPE_MISMATCH
            if self.type in ["alltoall", "alltoall_base"]:
                return MatchState.UNDECIDED
            if self.type != "scatter" and self.input_sizes != other.input_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH
            if self.type != "gather" and self.output_sizes != other.output_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH
            if self.type == "all_reduce" and self.input_sizes != other.output_sizes:
                return MatchState.SIZE_OR_SYNTAX_MISMATCH
            # TODO: need to consider uneven sharding for all-gather.
            # TODO: need to consider all_gather_into_tensor_coalesced (coalesced related)
            if self.type in [
                "all_gather",
                "all_gather_base",
            ] and not check_size_even_expand(
                other.output_sizes, self.input_sizes, self.pg_size
            ):
                return MatchState.SIZE_OR_SYNTAX_MISMATCH
            if self.type in [
                "reduce_scatter",
                "_reduce_scatter_base",
            ] and not check_size_even_expand(
                other.input_sizes, self.output_sizes, self.pg_size
            ):
                return MatchState.SIZE_OR_SYNTAX_MISMATCH
            # TODO: need to add more checks for gather and scatter.
            if self.state != other.state:
                return MatchState.COLLECTIVE_STATE_MISMATCH
        elif self.type == "coalesced":
            return (
                MatchState.FULLY_MATCHED
                if (other.type == "coalesced")
                else MatchState.SIZE_OR_SYNTAX_MISMATCH
            )
        return MatchState.FULLY_MATCHED


def match_one_event(
    event_a: Dict[Any, Any],
    event_b: Dict[Any, Any],
    memberships: Dict[str, Set[Any]],
) -> MatchState:
    op_a = Op(event_a, memberships)
    op_b = Op(event_b, memberships)
    return op_a.match(op_b)


def match_coalesced_groups(
    all_rank_events: Dict[Any, Any],
    group_size: int,
    groups: Dict[str, Group],
    memberships: Dict[str, Set[Any]],
) -> bool:
    """
    all_rank_events: {
        rank: [
            (idx, event_dict)
        ]
    }

    Note: it is possible for event dicts in a coalesced group to be asymmetric.
        e.g. the following events lists form a valid coalescing group
             events0 [send:1]
             events1 [recv:0, send:2]
             events2 [recv:1]

    Rule 1: all ops should find a match
    Rule 2: relative ordering of sends and recvs in one event list can be arbitrary
        e.g.
        events1 [recv:0, send:2]  —> okay
        events1 [send:2, recv:0] —> also okay
    Rule 3: sends to the same dest or recvs from the src should be in a consistent order
        e.g.
        rank0 [send:1 (100B), send:1 (1000B)]
        rank1 [recv:0 (1000B), recv:0 (100B)]   —> not okay
    """
    all_ops = {
        rank: [Op(e, memberships) for i, e in all_rank_events[rank]]
        for rank in all_rank_events
    }

    def visualize_ops(match: bool) -> None:
        all_ops = {
            rank: [Op(e, memberships) for i, e in all_rank_events[rank]]
            for rank in all_rank_events
        }

        i = 0
        row = []
        progress = True
        table = []
        while progress:
            progress = False
            for r in all_ops:
                if len(all_ops[r]) > i:
                    _, event = all_rank_events[r][i]
                    row.append(Op(event, memberships))
                    progress = True
                else:
                    row.append(None)  # type: ignore[arg-type]
            table.append(row)
            row = []
            i += 1
        title = "Match" if match else "MISMATCH"
        print(f"{title}\n", tabulate(table))  # type: ignore[operator]

    # TODO can't verify seq_id bc there might have been valid seq deltas between ranks even within a pg.
    for op_list in all_ops.values():
        if not op_list:
            # print("TODO- not sure if its valid for only some ranks in a PG to participate in a coalesced op?")
            return False
        assert op_list[-1].type == "coalesced"
        op_list.pop(-1)

    while all_ops:
        first_rank = next(iter(all_ops))
        my_ops = all_ops[first_rank]

        if len(all_ops[first_rank]) == 0:
            all_ops.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        op = my_ops[0]
        match_idx = -1
        if op.type in P2P:
            dst_global_rank = sorted(memberships[op.pg_name])[op.dst]
            peer_ops = all_ops[dst_global_rank]
            for i, other in enumerate(peer_ops):
                if op.match(other) == MatchState.FULLY_MATCHED:
                    match_idx = i
                    break
                elif op.dst == other.src:
                    # Rule 3
                    break
                else:
                    # Rule 1
                    continue
        else:
            raise NotImplementedError("coalesced collective ops")
        if match_idx >= 0:
            my_ops.pop(0)
            peer_ops.pop(match_idx)
        else:
            visualize_ops(False)
            return False

    visualize_ops(True)
    return True


"""
Flat DB builder
"""


def build_groups_memberships(
    pg_config: Any,
) -> Tuple[List[Group], Dict[Any, Group], List[Membership], Dict[str, Set[Any]]]:
    """
    pg_config: {
        global_rank: {
            (pg_id, desc, ranks)
        }
    }

    `pg_id` is a system generated id, but depending on the mode of PG creation it could be a globally incrementing int
          or a hash of the ranks.  See `_process_group_name` in distributed_c10d.py.
    `desc` is provided by the user (optionally) and should be 'meaningful' (e.g. TP/PP/DP group)
    `ranks` is a list of the 'global ranks' that are members of the PG.

    (pg_id, desc, ranks) tuples are appended lazily to the flight buffer when `getNCCLComm` is called on a PG and
    the `enabled_` flag is true for that PG.
        - the order of calling (init_process_group, new_group, etc) does not affect the order of the tuples in the list

    Returns: a groups table and a membership table, where each row is a Group or Membership namedtuple
    """
    # flat lists for return
    groups = []
    memberships = []

    # dicts for faster cross-rank validation
    _groups = {}
    _memberships = {}
    for global_rank in pg_config:
        for pg_id in pg_config[global_rank]:
            desc = pg_config[global_rank][pg_id]["desc"]
            ranks = ast.literal_eval(pg_config[global_rank][pg_id]["ranks"])
            if isinstance(ranks, str):
                # TODO Bug in FR data format? ranks is '[0, 1,...]'
                ranks = eval(ranks)

            if pg_id not in _groups:
                groups.append(Group(id=pg_id, desc=desc, size=len(ranks)))
                for rank in ranks:
                    memberships.append(Membership(group_id=pg_id, global_rank=rank))
                _groups[pg_id] = groups[-1]
                # TODO: make ranks int no matter what input (because it can be json or pickled string)
                _memberships[pg_id] = set(ranks)
            else:
                # validation across ranks
                assert (
                    _groups[pg_id].desc == desc
                ), f"mismatch in desc {_groups[pg_id].desc} vs {desc}"
                assert _memberships[pg_id] == set(
                    ranks
                ), f"mismatch in membership {_memberships[pg_id]} vs {set(ranks)}"
    return groups, _groups, memberships, _memberships


def build_nccl_call(
    entry: Dict[Any, Any],
    id: int,
    collective_id: Any,
    group_id: int,
    global_rank: Any,
) -> NCCLCall:
    return NCCLCall(
        id=id,
        collective_id=collective_id,
        group_id=group_id,  # type: ignore[arg-type]
        global_rank=global_rank,
        traceback_id=0,  # type: ignore[arg-type]
        collective_type=entry["profiling_name"],
        sizes=entry["input_sizes"],
    )


def find_coalesced_group(
    pg_name: str, entries: List[Dict[str, Any]]
) -> List[Tuple[int, Dict[str, Any]]]:
    """Given a list of entries, if the collective_seq_id of the first entry matches that of subsequent ones,
    build an return a list of entries terminating in a 'coalesced' op entry all sharing a collective_seq_id
    TODO: handle p2p_seq_id v/s collective_seq_id separately here.
    """
    found = []
    collective_seq_id = None
    for i, e in enumerate(entries):
        if e["process_group"][0] != pg_name:
            continue
        elif collective_seq_id is None:
            collective_seq_id = e["collective_seq_id"]
            found.append((i, e))
        elif e["collective_seq_id"] == collective_seq_id:
            found.append((i, e))
        else:
            break

    if len(found) > 1:
        assert found[-1][1]["profiling_name"] == "nccl:coalesced"
        return found
    return []


def just_print_entries(
    all_entries: Dict[int, List[Dict[str, Any]]],
    _groups: Dict[str, Group],
    _memberships: Dict[str, Set[Any]],
) -> None:
    rows = []
    ranks = sorted(all_entries.keys())
    headers = [f"Rank {rank}" for rank in ranks]
    progress = True
    while progress:
        progress = False
        row = []
        for rank in ranks:
            if len(all_entries[rank]) == 0:
                row.append("")
            else:
                entry = all_entries[rank].pop(0)
                row.append(str(Op(entry, _memberships)))
                progress = True
        if progress:
            rows.append(row)

    print(tabulate(rows, headers=headers))  # type: ignore[operator]


def build_collectives(
    all_entries: Dict[str, List[Dict[str, Any]]],
    _groups: Dict[str, Group],
    _memberships: Dict[str, Set[Any]],
) -> Tuple[List[Traceback], List[Collective], List[NCCLCall]]:
    """
    groups, memberships are the non-flat dicts that are indexable
    all_entries is a raw dict from the original dumps:

    all_entries: {
        global_rank: [
            {
                record_id: ordered id of the event in the trace buffer
                pg_id: ProcessGroupNCCL::uid_
                    *note: `pg_id` corresponds to nothing in groups table
                process_group: (pg_name, desc)
                    *note: `pg_name`, `desc` corresponds to `pg_id`, `desc` in groups table
                collective_seq_id: ordered id for collective operations and coalesced group operations
                p2p_seq_id: ordered id for point-to-point operations
                op_id: ordered id including individual ops inside coalescing group
                profiling_name: descriptive name of the operation
                'time_created_ns',
                'input_sizes',
                'output_sizes',
                'state',
                'time_discovered_started_ns',
                'time_discovered_completed_ns',
                'retired',
                'frames',
            }
        ]
    }
    """
    tracebacks: List[Traceback] = []

    collectives: List[Collective] = []
    nccl_calls: List[NCCLCall] = []

    # once we find one mismatch, we stop pairing up collectives since the pairing is possibly incorrect
    # instead, just record the remaining ops as NCCLCalls
    mismatch = {_groups[g].id: 0 for g in _groups}
    MISMATCH_TAIL = 10
    """
    - it doesn't matter what order I put collectives/ncclops into their table. we can later on re-sort it by start time
    - there could be multiple options for the "first" collective to pair up (rank 0,1 might do a bcast while rank 2,3 do a bcast)
    - within a group, the first collective must be the same on all ranks in the group, then it can be marked as a
    collective and removed
    """
    while all_entries:
        # we greedily match collectives, starting arbitrarily with the trace from the first rank
        # later, if we exhaust the first rank, we continue with the next 'first rank'
        rank_iter = iter(all_entries)
        first_rank = next(rank_iter)
        other_ranks = list(rank_iter)

        if len(all_entries[first_rank]) == 0:
            all_entries.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        entries = all_entries[first_rank]
        pg_name, desc = entries[0]["process_group"]
        profiling_name = entries[0]["profiling_name"]
        collective_seq_id = entries[0]["collective_seq_id"]
        expected_ranks = set(_memberships[pg_name])
        candidate_ranks = {first_rank}
        candidate_idx = {}
        found_ranks = set()
        found_idx = {}

        if find_coalesced_group(pg_name, entries):
            expected_ranks.add(first_rank)
            done_ranks = set()
            all_coalesced_entries = {}
            while expected_ranks:
                curr = expected_ranks.pop()
                done_ranks.add(curr)
                grp = (
                    find_coalesced_group(pg_name, all_entries[curr])  # type: ignore[index]
                    if curr in all_entries  # type: ignore[comparison-overlap]
                    else []
                )
                all_coalesced_entries[curr] = grp
                for index, entry in grp:
                    op = Op(entry, _memberships)
                    peer = None
                    if op.type == "send":
                        assert op._src_g == curr, (op._src_g, curr)
                        peer = op._dst_g
                    elif op.type == "recv":
                        assert op._dst_g == curr, (op._dst_g, curr)
                        peer = op._src_g
                    if peer and peer not in done_ranks:
                        expected_ranks.add(peer)

            match = match_coalesced_groups(
                all_coalesced_entries,
                group_size=_groups[pg_name].size,
                groups=_groups,
                memberships=_memberships,
            )

            if match and mismatch[pg_name] == 0:
                collectives.append(Collective(id=len(collectives), group_id=pg_name))
            else:
                mismatch[pg_name] += 1

            for r in all_coalesced_entries:
                reversed_calls = []
                for i, _ in reversed(all_coalesced_entries[r]):
                    reversed_calls.append(
                        build_nccl_call(
                            all_entries[r].pop(i),  # type: ignore[index]
                            id=len(nccl_calls),
                            collective_id=collectives[-1].id if match else None,
                            group_id=pg_name,
                            global_rank=r,
                        )
                    )
                nccl_calls.extend(reversed(reversed_calls))
        else:
            has_undecided_case = False
            errors = Set()
            for o in expected_ranks.intersection(set(other_ranks)):
                for i, e in enumerate(all_entries[o]):  # type: ignore[index]
                    # step over ops from other PGs
                    # only check match state when seq_id matches
                    if (
                        e["process_group"] == (pg_name, desc)
                        and e["collective_seq_id"] == collective_seq_id
                    ):
                        match_state = match_one_event(entries[0], e, _memberships)
                        if (
                            match_state
                            in [MatchState.FULLY_MATCHED, MatchState.UNDECIDED]
                            and mismatch[pg_name] == 0
                        ):
                            found_ranks.add(o)
                            found_idx[o] = i
                            has_undecided_case = match_state == MatchState.UNDECIDED
                        else:
                            candidate_ranks.add(o)
                            candidate_idx[o] = i
                            errors.add(match_state)
                        break

            # case one: not every rank join the collective or in the flight recorder.
            if (candidate_ranks | found_ranks) != expected_ranks:
                mismatch[pg_name] += 1
                print(
                    f"Not all ranks joining collective for group {pg_name}:{desc} collective {profiling_name}",
                    f"Missing ranks are {expected_ranks - (candidate_ranks | found_ranks)}",
                )
            elif len(candidate_ranks) == 1:
                # case two: alltoall or alltoall_base case.
                if has_undecided_case:
                    alltoall_cases = [entries[0]] + [
                        all_entries[o][found_idx[o]] for o in found_ranks
                    ]
                    check_result, input_numel, output_numel = check_size_alltoall(
                        alltoall_cases
                    )
                    if not check_result:
                        mismatch[pg_name] += 1
                        print(
                            f"Input/output mismatch in the collective for group {pg_name}:{desc} collective {profiling_name}",
                            f"input_numel {input_numel} output_numel{output_numel}",
                        )
                        candidate_ranks.update(found_ranks)
                        candidate_idx.update(found_idx)
                        found_idx.clear()
                        found_ranks.clear()
                    else:
                        found_ranks.update(candidate_ranks)
                        found_idx.update(candidate_idx)
                        candidate_idx.clear()
                        candidate_ranks.clear()
                # case three: all joined and everything matches on all ranks.
                else:
                    found_ranks.update(candidate_ranks)
                    found_idx.update(candidate_idx)
                    candidate_idx.clear()
                    candidate_ranks.clear()
            # case four: mismatch cases due to not same type, size mismatch or state mismatch.
            else:
                error_msg = ", ".join(error.name for error in errors)
                print(
                    f"Collective errors for group {pg_name}:{desc} collective {profiling_name}",
                    f"Found errors: {error_msg}",
                )
                candidate_ranks.update(found_ranks)
                candidate_idx.update(found_idx)
                found_idx.clear()
                found_ranks.clear()

            # at this point there are 3 possibilities
            # 1. we found a match on all the ranks that are members of the group
            #  -> we create a Collective and remove the individual entries from their original lists
            if found_ranks == expected_ranks and mismatch[pg_name] == 0:
                collectives.append(Collective(id=len(collectives), group_id=pg_name))
                for r in found_ranks:
                    i = found_idx[r] if r != first_rank else 0
                    nccl_calls.append(
                        build_nccl_call(
                            all_entries[r].pop(i),  # type: ignore[index]
                            id=len(nccl_calls),
                            collective_id=collectives[-1].id,
                            group_id=pg_name,
                            global_rank=r,
                        )
                    )

            # 2. we found a partial match but some ranks are missing
            # 3. we found no match
            #  -> since its not a complete collective, no entry goes into collectives but we still record a nccl call
            #     TODO should there be a way to mark 'mismatches'?
            else:
                print("appending a non-matching collective")
                # TODO: figure out a better for mismatch.
                # Also, shall we add seq Id as well?
                for r in candidate_ranks:
                    i = candidate_idx[r] if r != first_rank else 0
                    nccl_calls.append(
                        build_nccl_call(
                            all_entries[r].pop(i),  # type: ignore[index]
                            id=len(nccl_calls),
                            collective_id=None,
                            group_id=pg_name,
                            global_rank=r,
                        )
                    )

        if mismatch[pg_name] > MISMATCH_TAIL:
            print(f"Too many mismatches for process_group {pg_name}:{desc}, aborting")
            sys.exit(-1)

    return tracebacks, collectives, nccl_calls


def check_no_missing_dump_files(
    entries: Dict[str, Any], memberships: List[Membership]
) -> None:
    all_ranks = set()
    for membership in memberships:
        all_ranks.add(str(membership.global_rank))
    dumps_ranks = set(entries.keys())
    assert (
        dumps_ranks == all_ranks
    ), f"Missing dump files from ranks {all_ranks - dumps_ranks}"


def check_version(versions: Dict[str, Any]) -> None:
    for rank, version in versions.items():  # noqa: PERF102
        major, minor = map(int, version.split("."))
        # assert major == 2, f"Rank {rank} unsupported version {version}"
        # assert minor >= 0, f"Rank {rank} unsupported version {version}"


def check_trace_from_beginning(entries: Dict[str, Any]) -> bool:
    for rank in entries:
        first_record_id = entries[rank][0]["record_id"]
        # TODO add more sequence information such that analysis can proceed even without complete buffer

        # assert first_record_id == 0, f"Rank {rank} trace does not start at time 0 (first record is {first_record_id}."
        if first_record_id != 0:
            print(
                f"Rank {rank} trace does not start at time 0 (first record is {first_record_id}."
            )
            return False
    return True


def build_db(details: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> Database:
    # temporary state used for building database
    entries = {}
    pg_config = {}
    version = {}
    for dump in details.values():
        rank = dump["rank"]
        entries[rank] = dump["entries"]
        version[rank] = dump["version"]
        pg_config[rank] = dump["pg_config"]

    check_version(version)
    check_trace_from_beginning(entries)

    # flattened database
    groups, _groups, memberships, _memberships = build_groups_memberships(pg_config)
    print("built groups, memberships")

    check_no_missing_dump_files(entries, memberships)

    if args.just_print_entries:
        just_print_entries(entries, _groups, _memberships)
        sys.exit(0)

    tracebacks, collectives, nccl_calls = build_collectives(
        entries, _groups, _memberships
    )
    print("built collectives, nccl_calls")
    if args.verbose:
        print("Groups\n", tabulate(groups, headers=Group._fields))  # type: ignore[operator]
        print("Memberships\n", tabulate(memberships, headers=Membership._fields))  # type: ignore[operator]
        print("Collectives\n", tabulate(collectives, headers=Collective._fields))  # type: ignore[operator]
        print("NCCLCalls\n", tabulate(nccl_calls, headers=NCCLCall._fields))  # type: ignore[operator]
    db = Database(
        tracebacks=tracebacks,
        collectives=collectives,
        ncclcalls=nccl_calls,
        groups=groups,
        memberships=memberships,
    )
    return db


def read_dump(prefix: str, filename: str) -> Dict[str, Union[str, int, List[Any]]]:
    basename = os.path.basename(filename)
    assert (
        basename.find(prefix) == 0
    ), f"args.prefix ({prefix}) must match the beginning of each filename ({basename})"
    rank = int(basename[len(prefix) :])
    host_name = f"host_rank{rank}"

    with open(filename, "rb") as infile:
        dump = pickle.load(infile)

    entries = dump["entries"]
    version = dump["version"]
    pg_config = dump["pg_config"]

    return {
        "host_name": host_name,
        "rank": rank,
        "entries": entries,
        "version": version,
        "pg_config": pg_config,
    }


def read_dir(prefix: str, folder: str) -> Dict[Any, Any]:  # TODO; fix types
    import gc
    import time

    gc.disable()
    details = {}
    t0 = time.time()
    for root, _, files in os.walk(folder):
        for f in files:
            ta = time.time()
            details[f] = read_dump(prefix, os.path.join(root, f))
            tb = time.time()
            # print(f"read file {f} in {tb - ta}s")
    print(f"loaded {len(files)} files in {tb - t0}s")
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dir", help="Directory with flight recorder dumps")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument(
        "-p",
        "--prefix",
        help="prefix to strip such that rank can be extracted",
        default="rank_",
    )
    parser.add_argument("-j", "--just_print_entries", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    details = read_dir(args.prefix, args.dir)
    db = build_db(details, args)
    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump((types, db), f)


if __name__ == "__main__":
    main()
