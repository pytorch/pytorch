# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from typing import Any

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger
from torch.distributed.flight_recorder.components.types import (
    Collective,
    EntryState,
    Group,
    MatchInfo,
    MatchState,
    MatchStateRecord,
    Membership,
    Op,
    P2P,
)


__all__ = [
    "add_stack_id_in_entries",
    "align_trace_from_beginning",
    "check_current_entry_match",
    "check_no_missing_dump_files",
    "check_version",
    "error_analysis",
    "find_coalesced_group",
    "find_coalesced_group_with_non_p2p",
    "get_version_detail",
    "just_print_entries",
    "match_coalesced_groups_with_non_p2p",
    "match_coalesced_groups",
    "format_frame",
    "format_frames",
    "match_one_event",
    "check_size_alltoall",
]

logger: FlightRecorderLogger = FlightRecorderLogger()


try:
    from tabulate import tabulate
except ModuleNotFoundError:
    logger.debug("tabulate is not installed. Proceeding without it.")


def format_frame(frame: dict[str, str]) -> str:
    name = frame["name"]
    filename = frame["filename"]
    line = frame["line"]
    return f"{name} at {filename}:{line}"


def format_frames(frames: list[dict[str, str]]) -> str:
    formatted_frames = []
    for frame in frames:
        # pyrefly: ignore [bad-argument-type]
        formatted_frames.append(format_frame(frame))
    return "\n".join(formatted_frames)


def match_one_event(
    event_a: dict[Any, Any],
    event_b: dict[Any, Any],
    memberships: dict[str, set[Any]],
    pg_name: str,
) -> MatchInfo:
    op_a = Op(event_a, memberships, pg_name)
    op_b = Op(event_b, memberships, pg_name)
    return op_a.match(op_b)


def match_coalesced_groups(
    all_rank_events: dict[Any, Any],
    group_size: int,
    groups: dict[str, Group],
    memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
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
        _pg_guids: dict[tuple[str, int], str],
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
                    # Check if the pg_guid exists for this rank and process group
                    pg_key = (event["process_group"][0], rank)
                    if pg_key in _pg_guids:
                        row.append(
                            Op(
                                event,
                                memberships,
                                _pg_guids[pg_key],
                            )
                        )
                    else:
                        # Skip this entry if pg_guid mapping doesn't exist
                        row.append(None)  # type: ignore[arg-type]
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
                if op.match(other).state == MatchState.FULLY_MATCHED:
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


# We enabled the creating FR entry for non-P2P slow path collective ops in v2.7.
def match_coalesced_groups_with_non_p2p(
    all_rank_events: dict[Any, Any],
    pg_info: tuple[str, str],
    memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
    mismatch: dict[str, int],
    dumps_ranks: set[int],
    version: str,
    collectives: list[Collective],
    match_record: MatchStateRecord,
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
            for _, e in all_rank_events[rank]
        ]
        for rank in all_rank_events
    }
    is_p2p = any(op.type in P2P for ops in all_ops.values() for op in ops)
    pg_name = pg_info[0]

    def visualize_ops(
        match: bool,
        _pg_guids: dict[tuple[str, int], str],
    ) -> None:
        all_ops = {
            rank: [
                Op(e, memberships, _pg_guids[(e["process_group"][0], rank)])
                for _, e in all_rank_events[rank]
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
                    # Check if the pg_guid exists for this rank and process group
                    pg_key = (event["process_group"][0], rank)
                    if pg_key in _pg_guids:
                        row.append(
                            Op(
                                event,
                                memberships,
                                _pg_guids[pg_key],
                            )
                        )
                    else:
                        # Skip this entry if pg_guid mapping doesn't exist
                        row.append(None)  # type: ignore[arg-type]
                    progress = True
                else:
                    row.append(None)  # type: ignore[arg-type]
            table.append(row)
            row = []
            i += 1
        title = "Match" if match else "MISMATCH"
        logger.info("%s \n", title)
        logger.info("%s", tabulate(table))  # type: ignore[operator]

    # TODO Need to verify no seq_id deltas for P2P ops.
    for rank, op_list in all_ops.items():
        if not op_list:
            logger.error("Rank %s has an empty op list.", rank)
            continue
        if op_list[-1].type == "coalesced" and is_p2p:
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
        if is_p2p:
            dst_global_rank = sorted(memberships[op.pg_name])[op.dst]
            peer_ops = all_ops[dst_global_rank]
            for i, other in enumerate(peer_ops):
                if op.match(other).state == MatchState.FULLY_MATCHED:
                    match_idx = i
                    break
                elif op.dst == other.src:
                    # Rule 3
                    break
                else:
                    # Rule 1
                    continue
            if match_idx >= 0:
                my_ops.pop(0)
                peer_ops.pop(match_idx)
            else:
                visualize_ops(False, _pg_guids)
                return False
        else:
            all_coalesced_entries = {
                rank: [e for _, e in all_rank_events[rank]] for rank in all_rank_events
            }
            current_entry = all_coalesced_entries[first_rank][0]
            my_ops.pop(0)

            match_record.reset_for_coalesced(
                EntryState(current_entry, match_record.expected_ranks),
                {first_rank},
            )

            # Iterate through all the ranks and check if there is a mismatch for the current entry.
            check_current_entry_match(
                all_coalesced_entries,
                _pg_guids,
                pg_info,
                current_entry,
                memberships,
                mismatch,
                match_record,
            )

            # Use heuristics to decide what type of errors and error messages we should print.
            error_analysis(
                all_coalesced_entries,
                match_record,
                dumps_ranks,
                first_rank,
                current_entry,
                mismatch,
                get_version_detail(version),
                pg_info[0],
            )

            # TODO: For now, we only check the correctness of individual collective within a coalesced one in
            # this script. We need to merge  (e.g, input/output sizes) together
            # for downstream consumer.

            # at this point there are 3 possibilities
            # 1. we found a match on all the ranks that are members of the group
            #  -> we create a Collective and remove the individual entries from their original lists
            if (
                match_record.found_ranks == match_record.expected_ranks
                and mismatch[pg_name] == 0
            ):
                # Just pop out this collective.
                idx_map = {
                    r: match_record.found_idx[r] if r != first_rank else 0
                    for r in match_record.found_ranks
                }
                for i, k in idx_map.items():
                    all_rank_events[i].pop(k)
                for r in match_record.found_ranks:
                    if r != first_rank:
                        all_ops[r].pop(0)

            # 2. we found a partial match but some ranks are missing
            # 3. we found no match
            #  -> since its not a complete collective, no entry goes into collectives but we still record a nccl call
            else:
                logger.debug("Non-matching collective inside coalesced group")
                idx_map = {
                    r: match_record.candidate_idx[r] if r != first_rank else 0
                    for r in match_record.candidate_ranks
                }
                collectives.append(
                    match_record.entry_state.to_collective(
                        len(collectives),
                        errors=match_record.errors,
                        idx_map=idx_map,
                        all_entries=all_coalesced_entries,
                    )
                )
                return False

    if is_p2p:
        visualize_ops(True, _pg_guids)
    return True


def check_size_alltoall(alltoall_cases: list[dict[str, Any]]) -> tuple[bool, int, int]:
    input_numel = 0
    output_numel = 0
    for e in alltoall_cases:
        input_numel += math.prod(e["input_sizes"][0])
        output_numel += math.prod(e["output_sizes"][0])
    return input_numel != output_numel, input_numel, output_numel


def check_current_entry_match(
    all_entries: dict[int, list[dict[str, Any]]],
    _pg_guids: dict[tuple[str, int], str],
    pg_info: tuple[str, str],
    current_entry: dict[str, Any],
    _memberships: dict[str, set[Any]],
    mismatch: dict[str, int],
    match_record: MatchStateRecord,
) -> None:
    pg_name, desc = pg_info[0], pg_info[1]
    for o in match_record.expected_ranks.intersection(set(match_record.other_ranks)):
        for i, e in enumerate(all_entries[o]):  # type: ignore[index]
            # step over ops from other PGs
            # only check match state when seq_id matches
            if (
                _pg_guids[(e["process_group"][0], o)] == pg_name
                and e["process_group"][1] == desc
                and e["collective_seq_id"] == match_record.entry_state.collective_seq_id
            ):
                match_info = match_one_event(current_entry, e, _memberships, pg_name)
                if (
                    match_info.state in [MatchState.FULLY_MATCHED, MatchState.UNDECIDED]
                    and mismatch[pg_name] == 0
                ):
                    match_record.found_ranks.add(o)
                    match_record.found_idx[o] = i
                    match_record.has_undecided_case = (
                        match_info.state == MatchState.UNDECIDED
                    )
                else:
                    match_record.candidate_ranks.add(o)
                    match_record.candidate_idx[o] = i
                    if match_info.state not in [
                        MatchState.FULLY_MATCHED,
                        MatchState.UNDECIDED,
                    ]:
                        # Here we assume the current rank is not the source of the error.
                        # But it's possible that the current rank is the culprit, then users will
                        # see lots of normal ranks reported as culprit.
                        # TODO: we need to figure out a better way to handle the case mentioned above.
                        match_record.errors.add((o, match_info))
                break


def error_analysis(
    all_entries: dict[int, list[dict[str, Any]]],
    match_record: MatchStateRecord,
    dumps_ranks: set[int],
    first_rank: int,
    current_entry: dict[str, Any],
    mismatch: dict[str, int],
    version: tuple[int, int],
    pg_name: str,
) -> None:
    major_v, minor_v = version[0], version[1]
    # case one: not every rank join the collective or in the flight recorder.
    if (
        match_record.candidate_ranks | match_record.found_ranks
    ) != match_record.expected_ranks and match_record.expected_ranks - (
        match_record.candidate_ranks | match_record.found_ranks
    ) <= dumps_ranks:
        mismatch[pg_name] += 1
        logger_msg = "Not all ranks joining collective, sequence number: %s"
        missing_ranks = match_record.expected_ranks - (
            match_record.candidate_ranks | match_record.found_ranks
        )
        match_record.entry_state.log(
            logger, logger_msg, format_frames, missing_ranks=missing_ranks
        )
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
    # We didn't see any mismatch and all expected ranks are in the dump.
    elif len(
        match_record.candidate_ranks
    ) == 1 and match_record.expected_ranks.issubset(dumps_ranks):
        # case two: alltoall or alltoall_base case.
        if match_record.has_undecided_case:
            alltoall_cases = [current_entry] + [
                all_entries[o][match_record.found_idx[o]]
                for o in match_record.found_ranks
            ]
            fail_check, total_input_numel, total_output_numel = check_size_alltoall(
                alltoall_cases
            )
            if major_v <= 2 and minor_v <= 3:
                # We don't log the input/output sizes for alltoall before v2.4,
                # so we don't consider the size mismatch as an error for now.
                fail_check = False
            if fail_check:
                # When we see errors in all_to_all, it's hard to tell which rank is the source of the error.
                mismatch[pg_name] += 1
                logger_msg = (
                    "Input/output mismatch in the collective sequence number: %s"
                )
                match_record.entry_state.log(
                    logger,
                    logger_msg,
                    format_frames,
                    total_numel=(total_input_numel, total_output_numel),
                )
                match_record.candidate_ranks.update(match_record.found_ranks)
                match_record.candidate_idx.update(match_record.found_idx)
                match_record.found_idx.clear()
                match_record.found_ranks.clear()
                match_record.errors.add(
                    (first_rank, MatchInfo(MatchState.SIZE_OR_SYNTAX_MISMATCH))
                )
            else:
                match_record.found_ranks.update(match_record.candidate_ranks)
                match_record.found_idx.update(match_record.candidate_idx)
                match_record.candidate_idx.clear()
                match_record.candidate_ranks.clear()
        # case three: all joined and everything matches on all ranks.
        else:
            match_record.found_ranks.update(match_record.candidate_ranks)
            match_record.found_idx.update(match_record.candidate_idx)
            match_record.candidate_idx.clear()
            match_record.candidate_ranks.clear()
    # case four: mismatch cases due to not same type, size mismatch or state mismatch.
    elif len(match_record.errors) > 0:
        mismatch[pg_name] += 1
        logger_msg = "Collective sequence number: %s has errors"
        match_record.entry_state.log(
            logger, logger_msg, format_frames, errors=match_record.errors
        )
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
    # partial analysis case when we cannot decide what's wrong with this collective entry.
    else:
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
        # if any element in expected_ranks not in dumps_ranks.
        if match_record.expected_ranks - dumps_ranks:
            mismatch[pg_name] += 1
            logger.info(
                "We cannot decide what's wrong with this collective entry "
                "because we missed FR dumps from ranks (%s) so we don't have enough "
                "information. If you want to debug further use -j to dump all raw trace",
                str(match_record.expected_ranks - dumps_ranks),
            )
        else:
            logger.info(
                "No errors found for this collective entry, There could be some "
                "other reasons why we see collective timeout."
            )


def find_coalesced_group(
    pg_name: str,
    entries: list[dict[str, Any]],
    _pg_guids: dict[tuple[str, int], str],
    rank: int,
) -> list[tuple[int, dict[str, Any]]]:
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


# We enabled the creating FR entry for non-P2P slow path collective ops in v2.7.
def find_coalesced_group_with_non_p2p(
    pg_name: str,
    entries: list[dict[str, Any]],
    _pg_guids: dict[tuple[str, int], str],
    rank: int,
) -> list[tuple[int, dict[str, Any]]]:
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
        name = found[-1][1]["profiling_name"]
        if name.startswith("nccl:") and not name.endswith("_coalesced"):
            logger.error("Rank %s does not have a coalesced end.", rank)
        return found
    return []


def just_print_entries(
    all_entries: dict[int, list[dict[str, Any]]],
    _groups: dict[str, Group],
    _memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
    args: argparse.Namespace,
    stack_id_trace_map: dict[str, int],
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

    if stack_id_trace_map and args.print_stack_trace:
        headers = ["stack_id", "frame_stack"]
        rows = []

        for frame, stack_id in sorted(
            stack_id_trace_map.items(), key=lambda item: item[1]
        ):
            rows.append([str(stack_id), frame])

        logger.info(tabulate(rows, headers=headers))


def check_no_missing_dump_files(
    entries: dict[int, Any], memberships: list[Membership]
) -> None:
    all_ranks = {int(membership.global_rank) for membership in memberships}
    dumps_ranks = {int(key) for key in entries}
    missing = all_ranks - dumps_ranks
    assert len(missing) == 0, f"Missing dump files from ranks {missing}"


def check_version(version_by_ranks: dict[str, str], version: str) -> None:
    for rank, v in version_by_ranks.items():
        assert v == version, (
            f"Rank {rank} has different version {v} from the given version {version}"
        )


def get_version_detail(version: str) -> tuple[int, int]:
    # pyrefly: ignore [bad-assignment]
    version = version.split(".")
    assert len(version) == 2, f"Invalid version {version}"
    major, minor = map(int, version)
    return major, minor


def add_stack_id_in_entries(
    entries: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    stack_id = 0
    stack_id_trace_map = {}
    for rank in entries:
        for dump in entries[rank]:
            if dump.get("frames", []):
                frames = str(dump["frames"])
                if frames not in stack_id_trace_map:
                    stack_id_trace_map[frames] = stack_id
                    dump["stack_id"] = stack_id
                    stack_id += 1
                else:
                    dump["stack_id"] = stack_id_trace_map[frames]
            else:
                dump["stack_id"] = -1

    return entries, stack_id_trace_map


def align_trace_from_beginning(
    entries: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
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
        # If we don't have any trace from some ranks, ignore them
        # as well.
        if len(entries[rank]) == 0:
            continue
        first_record_id = entries[rank][0]["record_id"]
        maximum_starting_record_id = max(maximum_starting_record_id, first_record_id)

    for rank in entries:
        entries[rank] = [
            entry
            for entry in entries[rank]
            if entry["record_id"] >= maximum_starting_record_id
        ]

    return entries
