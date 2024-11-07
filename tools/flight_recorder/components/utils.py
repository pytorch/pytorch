# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from typing import Any, Dict, List, Set, Tuple

from tools.flight_recorder.components.fr_logger import FlightRecorderLogger
from tools.flight_recorder.components.types import (
    Group,
    MatchState,
    Membership,
    Op,
    P2P,
)


logger: FlightRecorderLogger = FlightRecorderLogger()


try:
    from tabulate import tabulate
except ModuleNotFoundError:
    logger.debug("tabulate is not installed. Proceeding without it.")


def format_frame(frame: Dict[str, str]) -> str:
    name = frame["name"]
    filename = frame["filename"]
    line = frame["line"]
    return f"{name} at {filename}:{line}"


def format_frames(frames: List[Dict[str, str]]) -> str:
    formatted_frames = []
    for frame in frames:
        formatted_frames.append(format_frame(frame))
    return "\n".join(formatted_frames)


def match_one_event(
    event_a: Dict[Any, Any],
    event_b: Dict[Any, Any],
    memberships: Dict[str, Set[Any]],
    pg_name: str,
) -> MatchState:
    op_a = Op(event_a, memberships, pg_name)
    op_b = Op(event_b, memberships, pg_name)
    return op_a.match(op_b)


def match_coalesced_groups(
    all_rank_events: Dict[Any, Any],
    group_size: int,
    groups: Dict[str, Group],
    memberships: Dict[str, Set[Any]],
    _pg_guids: Dict[Tuple[str, int], str],
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
        rank: [
            Op(e, memberships, _pg_guids[(e["process_group"][0], rank)])
            for i, e in all_rank_events[rank]
        ]
        for rank in all_rank_events
    }

    def visualize_ops(
        match: bool,
        _pg_guids: Dict[Tuple[str, int], str],
    ) -> None:
        all_ops = {
            rank: [
                Op(e, memberships, _pg_guids[(e["process_group"][0], rank)])
                for i, e in all_rank_events[rank]
            ]
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
                    rank, event = all_rank_events[r][i]
                    row.append(
                        Op(
                            event,
                            memberships,
                            _pg_guids[(event["process_group"][0], rank)],
                        )
                    )
                    progress = True
                else:
                    row.append(None)  # type: ignore[arg-type]
            table.append(row)
            row = []
            i += 1
        title = "Match" if match else "MISMATCH"
        logger.info("%s \n", title)
        logger.info("%s", tabulate(table))  # type: ignore[operator]

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
            visualize_ops(False, _pg_guids)
            return False

    visualize_ops(True, _pg_guids)
    return True


def check_size_alltoall(alltoall_cases: List[Dict[str, Any]]) -> Tuple[bool, int, int]:
    input_numel = 0
    output_numel = 0
    for e in alltoall_cases:
        input_numel += math.prod(e["input_sizes"][0])
        output_numel += math.prod(e["output_sizes"][0])
    return input_numel != output_numel, input_numel, output_numel


def find_coalesced_group(
    pg_name: str,
    entries: List[Dict[str, Any]],
    _pg_guids: Dict[Tuple[str, int], str],
    rank: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Given a list of entries, if the collective_seq_id of the first entry matches that of subsequent ones,
    build an return a list of entries terminating in a 'coalesced' op entry all sharing a collective_seq_id
    """
    found = []
    collective_seq_id = None
    for i, e in enumerate(entries):
        if _pg_guids[(e["process_group"][0], rank)] != pg_name:
            continue
        elif collective_seq_id is None:
            collective_seq_id = (
                e["p2p_seq_id"] if e["is_p2p"] else e["collective_seq_id"]
            )
            found.append((i, e))
        elif not e["is_p2p"] and e["collective_seq_id"] == collective_seq_id:
            found.append((i, e))
        elif e["is_p2p"] and e["p2p_seq_id"] == collective_seq_id:
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
    _pg_guids: Dict[Tuple[str, int], str],
    args: argparse.Namespace,
) -> None:
    rows = []
    ranks = sorted(all_entries.keys())
    headers = [
        f"Rank {rank}"
        for rank in ranks
        if args.selected_ranks is None or rank in args.selected_ranks
    ]
    progress = True
    while progress:
        progress = False
        row = []
        for rank in ranks:
            if args.selected_ranks is not None and rank not in args.selected_ranks:
                continue
            if len(all_entries[rank]) == 0:
                row.append("")
            else:
                entry = all_entries[rank].pop(0)
                pg_name = _pg_guids[(entry["process_group"][0], rank)]
                if (
                    args.pg_filters is None
                    or entry["process_group"][1] in args.pg_filters
                    or entry["process_group"][0] in args.pg_filters
                ):
                    row.append(str(Op(entry, _memberships, pg_name)))
                else:
                    row.append("")
                progress = True
        if progress:
            rows.append(row)

    logger.info(tabulate(rows, headers=headers))


def check_no_missing_dump_files(
    entries: Dict[int, Any], memberships: List[Membership]
) -> None:
    all_ranks = set()
    for membership in memberships:
        all_ranks.add(int(membership.global_rank))
    dumps_ranks = {int(key) for key in entries.keys()}
    assert (
        dumps_ranks == all_ranks
    ), f"Missing dump files from ranks {all_ranks - dumps_ranks}"


def check_version(version_by_ranks: Dict[str, str], version: str) -> None:
    for rank, v in version_by_ranks.items():
        assert (
            v == version
        ), f"Rank {rank} has different version {v} from the given version {version}"


def get_version_detail(version: str) -> Tuple[int, int]:
    version = version.split(".")
    assert len(version) == 2, f"Invalid version {version}"
    major, minor = map(int, version)
    return major, minor


def align_trace_from_beginning(
    entries: Dict[int, List[Dict[str, Any]]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Align the trace entries by record ID for entries.
    This function takes a dictionary of rank names to lists of trace entries as input.
    Each trace entry is a dictionary containing information about a collective operation,
    including its unique identifier (`record_id` is monotonically increasing as we write into the ring buffer).
    The function finds the largest starting point across all ranks by taking the maximum
    `record_id` value of the first entry in each rank. Finally, it filters out any
    entries with `record_id` values less than the maximum starting point.
    The function returns the updated dictionary of sorted and filtered trace entries.

    Args:
        entries (Dict[str, List[Dict[str, Any]]]): A dictionary of rank names to lists of trace entries.

    Returns:
        entries (Dict[str, List[Dict[str, Any]]]): Entries sorted by record ID and filtered by the maximum starting point.
    """

    maximum_starting_record_id = 0
    for rank in entries:
        # Although this is a ring buffer, we already sort the entries by `record_id` when dumping, we just
        # need to find the largest starting point. For example, if the buffer has the following entries:
        # Rank 0: [0, 1, 2, 3, 4, 5, 6]
        # Rank 1: [1, 2, 3, 4, 5, 6, 7]
        # Rank 2: [2, 3, 4, 5, 6, 7, 8]
        # Rank 3: [0, 1, 2, 3, 4, 5, None]
        # Then we should start from collective 2 not 0 because any collective before,
        # we don't have complete records from all ranks so we need to ignore them.
        first_record_id = entries[rank][0]["record_id"]
        maximum_starting_record_id = max(maximum_starting_record_id, first_record_id)

    for rank in entries:
        entries[rank] = [
            entry
            for entry in entries[rank]
            if entry["record_id"] >= maximum_starting_record_id
        ]

    return entries
