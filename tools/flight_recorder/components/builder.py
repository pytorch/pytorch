# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import sys
from typing import Any, Dict, List, Set, Tuple  # type: ignore[attr-defined]

from tools.flight_recorder.components.fr_logger import FlightRecorderLogger
from tools.flight_recorder.components.types import (
    Collective,
    Database,
    Group,
    MatchState,
    Membership,
    NCCLCall,
    Op,
    Traceback,
)
from tools.flight_recorder.components.utils import (
    align_trace_from_beginning,
    check_no_missing_dump_files,
    check_size_alltoall,
    check_version,
    find_coalesced_group,
    format_frames,
    get_version_detail,
    just_print_entries,
    match_coalesced_groups,
    match_one_event,
)


# Set up logging
logger: FlightRecorderLogger = FlightRecorderLogger()


try:
    from tabulate import tabulate
except ModuleNotFoundError:
    logger.warning("tabulate is not installed. Proceeding without it.")

    # Define a no-op tabulate function
    def tabulate(data: Any, headers: Any = None) -> Any:  # type: ignore[misc]
        return data


"""
Flat DB builder
"""


def build_groups_memberships(
    pg_config: Any,
) -> Tuple[
    List[Group],
    Dict[Any, Group],
    List[Membership],
    Dict[str, Set[Any]],
    Dict[Tuple[str, int], str],
]:
    """
    pg_config: {
        global_rank: {
            (pg_guid, desc, ranks)
        }
    }

    `pg_guid` is a system generated id, but depending on the mode of PG creation it could be a globally incrementing int
          or a hash of the ranks.  See `_process_group_name` in distributed_c10d.py.
    `desc` is provided by the user (optionally) and should be 'meaningful' (e.g. TP/PP/DP group)
    `ranks` is a list of the 'global ranks' that are members of the PG.

    (pg_guid, desc, ranks) tuples are appended lazily to the flight buffer when `getNCCLComm` is called on a PG and
    the `enabled_` flag is true for that PG.
        - the order of calling (init_process_group, new_group, etc) does not affect the order of the tuples in the list

    Returns:
        `groups`: a groups table where each row is a Group namedtuple.
        `_groups`: a dict that is indexed by pg_guid with Group namedtuple as value.
        `memberships`: a membership table where each row is a Membership namedtuple.
        `_memberships`: a dict that is indexed by pg_guid with set of ranks (int) as value.
        `_pg_guids`: a dict that is indexed by (pg_uid, global_rank) with pg_guid as value.
    """
    # flat lists for return
    groups = []
    memberships = []

    # dicts for faster cross-rank validation
    _groups = {}
    _memberships = {}
    _pg_guids = {}
    for global_rank in pg_config:
        for pg_uid in pg_config[global_rank]:
            desc = pg_config[global_rank][pg_uid]["desc"]
            ranks = ast.literal_eval(pg_config[global_rank][pg_uid]["ranks"])
            # With the adoption of the split_group API, we can have multiple PGs with the same pg_guid (PG Name)
            # So we need to add the hash of all its ranks within the PG as well.
            # Also guid must be a string because `_process_group_name` returns a string.
            pg_guid = pg_uid + str(hash(frozenset(ranks)))
            _pg_guids[(pg_uid, global_rank)] = pg_guid
            if isinstance(ranks, str):
                # TODO Bug in FR data format? ranks is '[0, 1,...]'
                ranks = eval(ranks)

            if pg_guid not in _groups:
                groups.append(Group(id=pg_guid, desc=desc, size=len(ranks)))
                for rank in ranks:
                    memberships.append(Membership(group_id=pg_guid, global_rank=rank))
                _groups[pg_guid] = groups[-1]
                _memberships[pg_guid] = set(ranks)
            else:
                # validation across ranks
                assert (
                    _groups[pg_guid].desc == desc
                ), f"mismatch in desc {_groups[pg_guid].desc} vs {desc} for group {pg_guid}"
                assert (
                    _memberships[pg_guid] == set(ranks)
                ), f"mismatch in membership for group {pg_guid} {_memberships[pg_guid]} vs {set(ranks)}"
    return groups, _groups, memberships, _memberships, _pg_guids


def build_nccl_call(
    entry: Dict[Any, Any],
    id: int,
    collective_id: Any,
    group_id: str,
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


def build_collectives(
    all_entries: Dict[int, List[Dict[str, Any]]],
    _groups: Dict[str, Group],
    _memberships: Dict[str, Set[Any]],
    _pg_guids: Dict[Tuple[str, int], str],
    version: str,
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
    major_v, minor_v = get_version_detail(version)
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
        original_pg_name = pg_name  # For db build and logs printing, we want to use the original pg_name, not the hash one.
        pg_desc = (
            f"{original_pg_name}:{desc}" if desc != "undefined" else original_pg_name
        )
        pg_name = _pg_guids[(original_pg_name, first_rank)]
        collective_seq_id = entries[0]["collective_seq_id"]
        record_id = entries[0]["record_id"]
        input_sizes = entries[0]["input_sizes"]
        output_sizes = entries[0]["output_sizes"]
        collective_state = entries[0]["state"]
        collective_frames = format_frames(entries[0]["frames"])
        expected_ranks = set(_memberships[pg_name])
        candidate_ranks = {first_rank}
        candidate_idx = {}
        found_ranks = set()
        found_idx = {}

        if find_coalesced_group(pg_name, entries, _pg_guids, first_rank):
            expected_ranks.add(first_rank)
            done_ranks = set()
            all_coalesced_entries = {}
            while expected_ranks:
                curr = expected_ranks.pop()
                done_ranks.add(curr)
                grp = (
                    find_coalesced_group(pg_name, all_entries[curr], _pg_guids, curr)  # type: ignore[index]
                    if curr in all_entries  # type: ignore[comparison-overlap]
                    else []
                )
                all_coalesced_entries[curr] = grp
                for index, entry in grp:
                    op = Op(entry, _memberships, pg_name)
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
                _pg_guids=_pg_guids,
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
                            group_id=original_pg_name,
                            global_rank=r,
                        )
                    )
                nccl_calls.extend(reversed(reversed_calls))
        else:
            has_undecided_case = False
            errors = set()
            for o in expected_ranks.intersection(set(other_ranks)):
                for i, e in enumerate(all_entries[o]):  # type: ignore[index]
                    # step over ops from other PGs
                    # only check match state when seq_id matches
                    if (
                        _pg_guids[(e["process_group"][0], o)] == pg_name
                        and e["process_group"][1] == desc
                        and e["collective_seq_id"] == collective_seq_id
                    ):
                        match_state = match_one_event(
                            entries[0], e, _memberships, pg_name
                        )
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
                            if match_state not in [
                                MatchState.FULLY_MATCHED,
                                MatchState.UNDECIDED,
                            ]:
                                # Here we assume the current rank is not the source of the error.
                                # But it's possible that the current rank is the culprit, then users will
                                # see lots of normal ranks reported as culprit.
                                # TODO: we need to figure out a better way to handle the case mentioned above.
                                errors.add((o, match_state))
                        break

            # case one: not every rank join the collective or in the flight recorder.
            if (candidate_ranks | found_ranks) != expected_ranks:
                mismatch[pg_name] += 1
                logger.info(
                    "Not all ranks joining collective %s at entry %s",
                    collective_seq_id,
                    record_id,
                )
                logger.info("group info: %s", pg_desc)
                logger.info("collective: %s", profiling_name)
                missing_ranks = expected_ranks - (candidate_ranks | found_ranks)
                logger.info("missing ranks: %s", missing_ranks)
                logger.info("input sizes: %s", input_sizes)
                logger.info("output sizes: %s", output_sizes)
                logger.info("expected ranks: %d", len(expected_ranks))
                logger.info("collective state: %s", collective_state)
                logger.info("collective stack trace: \n %s", collective_frames)
                candidate_ranks.update(found_ranks)
                candidate_idx.update(found_idx)
                found_idx.clear()
                found_ranks.clear()
            elif len(candidate_ranks) == 1:
                # case two: alltoall or alltoall_base case.
                if has_undecided_case:
                    alltoall_cases = [entries[0]] + [
                        all_entries[o][found_idx[o]] for o in found_ranks
                    ]
                    fail_check, input_numel, output_numel = check_size_alltoall(
                        alltoall_cases
                    )
                    if major_v <= 2 and minor_v <= 3:
                        # We don't log the input/output sizes for alltoall before v2.4,
                        # so we don't consider the size mismatch as an error for now.
                        fail_check = False
                    if fail_check:
                        # When we see errors in all_to_all, it's hard to tell which rank is the source of the error.
                        mismatch[pg_name] += 1
                        logger.info(
                            "Input/output mismatch in the collective %s at entry %s",
                            collective_seq_id,
                            record_id,
                        )
                        logger.info("group info: %s", pg_desc)
                        logger.info("collective: %s", profiling_name)
                        logger.info("input sizes: %s", input_sizes)
                        logger.info("output sizes: %s", output_sizes)
                        logger.info("expected ranks: %d", len(expected_ranks))
                        logger.info("collective state: %s", collective_state)
                        logger.info("collective stack trace: \n%s", collective_frames)
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
            elif len(errors) > 0:
                mismatch[pg_name] += 1
                error_msg = ", ".join(
                    f"Culprit rank {error[0]}; {str(error[1])}" for error in errors
                )
                logger.info(
                    "Collective %s at entry %s errors", collective_seq_id, record_id
                )
                logger.info("group info: %s", pg_desc)
                logger.info("collective: %s", profiling_name)
                logger.info("input sizes: %s", input_sizes)
                logger.info("output sizes: %s", output_sizes)
                logger.info("expected ranks: %d", len(expected_ranks))
                logger.info("collective state: %s", collective_state)
                logger.info("error message: %s", error_msg)
                logger.info("collective stack trace: \n%s", collective_frames)
                candidate_ranks.update(found_ranks)
                candidate_idx.update(found_idx)
                found_idx.clear()
                found_ranks.clear()

            # at this point there are 3 possibilities
            # 1. we found a match on all the ranks that are members of the group
            #  -> we create a Collective and remove the individual entries from their original lists
            if found_ranks == expected_ranks and mismatch[pg_name] == 0:
                collectives.append(
                    Collective(id=len(collectives), group_id=original_pg_name)
                )
                for r in found_ranks:
                    i = found_idx[r] if r != first_rank else 0
                    nccl_calls.append(
                        build_nccl_call(
                            all_entries[r].pop(i),  # type: ignore[index]
                            id=len(nccl_calls),
                            collective_id=collectives[-1].id,
                            group_id=original_pg_name,
                            global_rank=r,
                        )
                    )

            # 2. we found a partial match but some ranks are missing
            # 3. we found no match
            #  -> since its not a complete collective, no entry goes into collectives but we still record a nccl call
            #     TODO should there be a way to mark 'mismatches'?
            else:
                logger.debug("appending a non-matching collective")
                # TODO: figure out a better for mismatch.
                # Also, shall we add seq Id as well?
                for r in candidate_ranks:
                    i = candidate_idx[r] if r != first_rank else 0
                    nccl_calls.append(
                        build_nccl_call(
                            all_entries[r].pop(i),  # type: ignore[index]
                            id=len(nccl_calls),
                            collective_id=None,
                            group_id=original_pg_name,
                            global_rank=r,
                        )
                    )

        if mismatch[pg_name] > MISMATCH_TAIL:
            logger.error(
                "Too many mismatches for process_group %s: %s aborting", pg_name, desc
            )
            sys.exit(-1)

    return tracebacks, collectives, nccl_calls


def build_db(
    details: Dict[str, Dict[str, Any]], args: argparse.Namespace, version: str
) -> Database:
    # temporary state used for building database
    entries = {}
    pg_config = {}
    version_by_ranks = {}
    for dump in details.values():
        rank = dump["rank"]
        entries[rank] = dump["entries"]
        version_by_ranks[rank] = dump["version"]
        pg_config[rank] = dump["pg_config"]

    # Ensure version is consistent across all ranks.
    check_version(version_by_ranks, version)
    entries = align_trace_from_beginning(entries)

    # flattened database
    groups, _groups, memberships, _memberships, _pg_guids = build_groups_memberships(
        pg_config
    )
    logger.debug("built groups, memberships")

    if args.just_print_entries:
        just_print_entries(entries, _groups, _memberships, _pg_guids, args)
        sys.exit(0)

    check_no_missing_dump_files(entries, memberships)

    tracebacks, collectives, nccl_calls = build_collectives(
        entries, _groups, _memberships, _pg_guids, version
    )
    logger.debug("built collectives, nccl_calls")
    if args.verbose:
        logger.debug("Groups")
        logger.debug(tabulate(groups, headers=Group._fields))
        logger.debug("Memberships")
        logger.debug(tabulate(memberships, headers=Membership._fields))
        logger.debug("Collectives")
        logger.debug(tabulate(collectives, headers=Collective._fields))
        logger.debug("NCCLCalls")
        logger.debug(tabulate(nccl_calls, headers=NCCLCall._fields))
    db = Database(
        tracebacks=tracebacks,
        collectives=collectives,
        ncclcalls=nccl_calls,
        groups=groups,
        memberships=memberships,
    )
    return db
