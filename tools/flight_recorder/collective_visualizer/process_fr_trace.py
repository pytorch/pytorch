#!/usr/bin/env python3

# pyre-strict
import asyncio
import json
import os
import pickle
import re
from argparse import ArgumentParser, Namespace
from glob import glob
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

STACK_FRAME_FORMAT_STR: str = "{filename}:{line}, in `{name}`"


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Python script that dijests Flight Recorder traces and produces an UI to visualize the NCCL collectives around the time when NCCL watchdog timeout happens. We assumed all ranks have its Flight Recorder trace dumped."
    )
    parser.add_argument(
        "--dump-dir",
        "-d",
        type=str,
        help=r"Dump directory to the Flight Recorder traces to analyze. We assume the Flight Recorder traces are stored in the format of nccl_trace_rank_<rank>.",
    )
    parser.add_argument(
        "--fr-trace-filename-format",
        "-f",
        type=str,
        default=r"nccl_trace_rank_<rank>",
        help=r"The format that the Flight Recorder traces filenames follow, should include a placeholder <rank> to represent which rank the FR trace belongs to.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=os.path.join(".", "visualize", "public"),
        help="Output directory that the script will dump the intermediate trace.json file into. Default to the react visualize folder (./visualize/public)",
    )
    parser.add_argument(
        "--num-collectives-before-first-mismatch",
        "-n",
        type=int,
        default=20,
        help="Number of extra collectives that will be visualized before the first mismatching collective. Default to 20.",
    )
    return parser.parse_args()


def read_dump(trace_file_name: str) -> List[Any]:
    with open(trace_file_name, "rb") as f:
        dump = pickle.load(f)

    return dump["entries"]


def _get_pg_str(pg_name: str, pg_desc: str) -> str:
    return pg_name if pg_desc == "undefined" else f"{pg_name}:{pg_desc}"


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


def process_fr_traces(
    fr_traces: Dict[int, List[Any]], num_collectives_before_first_mismatch: int
) -> Tuple[Dict[int, List[Tuple[str, List[str]]]], int]:
    """Process Flight Recorder trace entries to the all collectives after the first mismatch + k collectives before the first collective."""
    min_rows = min(len(ops) for ops in fr_traces.values())
    last_row_all_completed = 0
    # find the last row that all ops are completed
    for row_idx in range(min_rows - 1, -1, -1):
        if all(
            fr_traces[rank_id][row_idx]["state"] == "completed"
            for rank_id in fr_traces.keys()
        ):
            last_row_all_completed = row_idx
            break

    return (
        {
            rank: [
                (
                    _get_pg_str(*entry["process_group"]),
                    [
                        STACK_FRAME_FORMAT_STR.format(**frame)
                        for frame in reversed(entry.get("frames", []))
                    ]
                    + [
                        f"{entry['collective_seq_id']}(seq_id):{entry['profiling_name'].split(':')[-1]}:{entry['state']}"
                    ],
                )
                for entry in entries[
                    max(
                        0,
                        last_row_all_completed - num_collectives_before_first_mismatch,
                    ) :
                ]
            ]
            for rank, entries in fr_traces.items()
        },
        (
            num_collectives_before_first_mismatch
            if last_row_all_completed > num_collectives_before_first_mismatch
            else num_collectives_before_first_mismatch - last_row_all_completed
        ),
    )


def dump_traces_into_json(
    traces: Dict[int, List[Tuple[str, List[str]]]],
    first_mismatch_record_id: int,
    output_dir: str,
) -> str:
    if not os.path.exists(output_dir):
        raise FileExistsError(f"Directory {output_dir} does not exist.")
    print(f"Storing intermediate json file into directory {output_dir}")
    output_file_name = os.path.join(output_dir, "trace.json")
    with open(output_file_name, "w") as f:
        json.dump(
            {"first_mismatch_record_id": first_mismatch_record_id, "traces": traces}, f
        )
    return output_file_name


async def main() -> None:
    args = get_args()
    fr_trace_filename_format = args.fr_trace_filename_format.replace(
        r"<rank>", r"(?P<rank>\d+)"
    )
    fr_trace_filenames = {
        int(result.group("rank")): f
        for f in glob(os.path.join(args.dump_dir, "*"))
        if (result := re.search(fr_trace_filename_format, f)) is not None
    }
    if not fr_trace_filenames:
        raise RuntimeError(
            f"No Flight Recorder traces found under directory {args.dump_dir} with the file format {args.fr_trace_filename_format}"
        )
    fr_traces = {
        rank: read_dump(filename) for rank, filename in fr_trace_filenames.items()
    }
    fr_traces = align_trace_from_beginning(fr_traces)
    processed_entries, first_mismatch_record_id = process_fr_traces(
        fr_traces=fr_traces,
        num_collectives_before_first_mismatch=args.num_collectives_before_first_mismatch,
    )
    output_filename = dump_traces_into_json(
        processed_entries, first_mismatch_record_id, args.output_dir
    )
    print(f"Dump traces into {output_filename} succeeds.")


def invoke_main() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    invoke_main()
