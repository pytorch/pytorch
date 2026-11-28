"""Pack the collected tests into shards that take equally long, from the times main
recorded. A shard runs a few test files at a time, each in a lane of its own, and
the tests marked serial with none beside them."""

from __future__ import annotations

import gzip
import heapq
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tests", required=True, help="the list that collect wrote")
    parser.add_argument("--output", required=True, help="JSON plan to write")
    times = "seconds per test on main where the shards run: a file of tools/tci/times"
    parser.add_argument("--times", required=True, help=times)
    parser.add_argument("--target", type=float, default=1200, help="seconds per shard")
    # 3 as on main, whose times were recorded with that many (NUM_PROCS).
    at_once = "test files that a shard runs at a time"
    parser.add_argument("--at-once", type=int, default=3, help=at_once)
    parser.add_argument("--max-shards", type=int, default=256, help="GitHub's limit")
    matrix = (
        "append to FILE the jobs of a GitHub matrix that run the shards: `NAME=<a JSON "
        "object for each>`. FIELDS is a JSON object, what a workflow has for these tests: "
        "a job gets them and its `shard`. A field that the reports of the tests have too "
        "(os, cpu_architecture, accelerator, ...) must say what collect found"
    )
    names = ("FILE", "NAME", "FIELDS")
    parser.add_argument("--matrix", nargs=3, metavar=names, help=matrix)


def minutes(seconds: float) -> str:
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def mean_seconds(classes: list[dict[str, Any]], default: float) -> float:
    """Mean time of the tests of some classes of the times file."""
    count = sum(c["others"] + len(c["tests"]) for c in classes)
    total = sum(c["others_seconds"] + sum(c["tests"].values()) for c in classes)
    return total / count if count else default


def seconds_per_test(
    classes: dict[str, Any], default: float, tests: list[str]
) -> dict[str, float]:
    """The time main recorded for each test of a file. A test without one gets the
    mean of the short tests of its class, else of the file, else `default`."""
    file_mean = mean_seconds(list(classes.values()), default)
    seconds = {}
    for test in tests:
        # Class::test_name[parameters]; main files a test under its innermost class.
        head, bracket, parameters = test.partition("[")
        *outer, name = head.split("::")
        name += bracket + parameters
        cls = classes.get(outer[-1] if outer else "")
        if cls is None:
            seconds[test] = file_mean
        elif name in cls["tests"]:
            seconds[test] = cls["tests"][name]
        elif cls["others"]:
            seconds[test] = cls["others_seconds"] / cls["others"]
        else:
            seconds[test] = file_mean
    return seconds


def pack(
    beside: dict[str, tuple[float, dict[str, float]]],
    alone: dict[str, tuple[float, dict[str, float]]],
    target: float,
    at_once: int,
    max_shards: int,
) -> list[dict[str, Any]]:
    """`beside` maps a test file to the seconds it takes to load and the seconds of
    each of its tests that run beside those of other files; `alone` has the same for
    its tests that run with none beside them. A shard runs the latter one after the
    other, then `at_once` lanes of the former side by side. Returns the shards:
    `seconds`, `alone` and `lanes`, where a file has the tests that the shard runs of
    it, or None for all of them. One piece of a file that is in pieces has them as
    {"and_unnamed": tests}: it also runs the tests of the file that no piece names."""
    # What runs alone takes all the lanes of its shard.
    groups = [(file, *beside[file], 1) for file in beside]
    groups += [(file, *alone[file], at_once) for file in alone]
    work = [sum(tests.values()) for _, _, tests, _ in groups]
    budget = target
    for _ in range(10):
        # Tests that do not fit in a lane are cut, and every piece loads the file
        # again. That moves the total, and with it the size of a shard.
        rooms = [max(budget - load, budget / 2) for _, load, _, _ in groups]
        fit = [math.ceil(seconds / room) for seconds, room in zip(work, rooms)]
        pieces = [max(1, min(len(g[2]), n)) for g, n in zip(groups, fit)]
        takes = [(w + n * g[1]) * g[3] for g, w, n in zip(groups, work, pieces)]
        shards = min(max_shards, max(1, math.ceil(sum(takes) / (target * at_once))))
        if abs(sum(takes) / (shards * at_once) - budget) < 1:
            break
        budget = sum(takes) / (shards * at_once)

    # (seconds, file, its tests or None for all of the file's, whether they run alone)
    units: list[tuple[float, str, list[str] | None, bool]] = []
    for (file, load, tests, width), seconds, n in zip(groups, work, pieces):
        if n == 1:
            every = (file in beside) != (file in alone)
            units.append(
                (load + seconds, file, None if every else list(tests), width > 1)
            )
            continue
        # Longest test first, each into the piece with the least so far.
        parts: list[tuple[float, int, list[str]]]
        parts = [(0.0, i, []) for i in range(n)]
        for test in sorted(tests, key=lambda test: -tests[test]):
            so_far, i, names = heapq.heappop(parts)
            names.append(test)
            heapq.heappush(parts, (so_far + tests[test], i, names))
        units += [(load + s, file, names, width > 1) for s, _, names in parts if names]
    units.sort(key=lambda unit: -unit[0])

    plan: list[dict[str, Any]]
    plan = [{"alone": {}, "lanes": [{} for _ in range(at_once)]} for _ in range(shards)]

    def put(into: dict[str, list[str] | None], unit: Any) -> float:
        seconds, file, tests_of_piece, _ = unit
        earlier_piece = into.get(file)
        if earlier_piece is None or tests_of_piece is None:
            into[file] = tests_of_piece
            return seconds
        # Two pieces of one file that run one after the other load the file once.
        earlier_piece += tests_of_piece
        return seconds - (beside.get(file) or alone[file])[0]

    # Longest first. What runs alone fills one shard after the other, so that the
    # lanes of the other shards keep their whole length for the largest pieces.
    starts = [0.0] * shards
    for unit in units:
        if unit[3]:
            fits = [i for i in range(shards) if starts[i] + unit[0] <= budget]
            i = fits[0] if fits else starts.index(min(starts))
            starts[i] += put(plan[i]["alone"], unit)
    # The others each go to the lane with the least so far. A lane starts when what
    # runs alone in its shard ends.
    lanes = [(start, i, k) for i, start in enumerate(starts) for k in range(at_once)]
    heapq.heapify(lanes)
    for unit in units:
        if not unit[3]:
            so_far, i, k = heapq.heappop(lanes)
            heapq.heappush(lanes, (so_far + put(plan[i]["lanes"][k], unit), i, k))
    for i, shard in enumerate(plan):
        shard["seconds"] = round(max(end for end, j, _ in lanes if j == i))
    # A test can exist where the tests run and not where they were listed: a GPU has
    # tests that an older one lacks. So one piece of each file that is in pieces
    # runs what no piece names too: one in a lane, if the file has a piece there.
    places = [lane for shard in plan for lane in shard["lanes"]]
    places += [shard["alone"] for shard in plan]
    unnamed: set[str] = set()
    for place in places:
        for file, tests in place.items():
            if tests is not None and file not in unnamed:
                place[file] = {"and_unnamed": tests}
                unnamed.add(file)
    return plan


def run(args: argparse.Namespace) -> None:
    with gzip.open(args.tests, "rt") as f:
        collected = json.load(f)
    times = json.loads(Path(args.times).read_text())
    source, on_main = times["source"], times["files"]
    classes = [cls for file in on_main.values() for cls in file.values()]
    default = mean_seconds(classes, 0)
    # The first configuration is the one of the jobs that run the shards.
    first = source["configurations"][0]["environment"]
    # The tests of a device exist only where there is one.
    if first["device_count"] and not collected["environment"]["device_count"]:
        sys.exit(f"the times are for {first['device_name']}: collect found no device")
    # What the workflow says of the environment is what its jobs are named by.
    fields = json.loads(args.matrix[2]) if args.matrix else {}
    found = collected["environment"]
    wrong = {f: found[f] for f in fields.keys() & found.keys() if fields[f] != found[f]}
    if wrong:
        sys.exit(f"the fields of the workflow say otherwise: collect found {wrong}")

    results = {file: r for file, r in collected["files"].items() if r["tests"]}
    # A file's seconds are those of its listing, by a fork that found torch imported.
    # A run starts a process of its own for it: collect timed how long one takes.
    start = collected["start"]
    # The tests marked serial run alone, the others beside those of other files.
    beside: dict[str, tuple[float, dict[str, float]]] = {}
    alone: dict[str, tuple[float, dict[str, float]]] = {}
    for file, r in results.items():
        recorded = on_main.get(f"test/{file}", {})
        seconds = seconds_per_test(recorded, default, r["tests"])
        marked = set(r.get("serial", ()))
        # In the order of the list, so that the same list gives the same plan.
        for tests, serial in ((beside, False), (alone, True)):
            kept = {t: seconds[t] for t in r["tests"] if (t in marked) == serial}
            if kept:
                tests[file] = (r["seconds"] + start, kept)
    shards = pack(beside, alone, args.target, args.at_once, args.max_shards)
    Path(args.output).write_text(json.dumps({"times": source, "shards": shards}))
    if args.matrix:
        jobs = [json.dumps({**fields, "shard": n}) for n in range(len(shards))]
        # No brackets: a workflow joins the jobs of its environments into one list.
        with open(args.matrix[0], "a") as f:
            f.write(f"{args.matrix[1]}={','.join(jobs)}\n")

    print(
        f"times: {source['table']} on {source['ran_on']}, last {source['window']}, "
        f"fetched {source['fetched_at']}"
    )
    # A test that the first configuration does not run on main has the time of a
    # later one.
    if collected["environment"] != first:
        print(f"  they are for {first}")
        print(f"  the tests were collected on {collected['environment']}")
    unknown = [file for file in results if f"test/{file}" not in on_main]
    print(
        f"{sum(len(r['tests']) for r in results.values())} tests in {len(results)} files; "
        f"no time on main for the {sum(len(results[file]['tests']) for file in unknown)} "
        f"tests of {len(unknown)} files"
    )
    folders = Counter(file.split("/")[0] if "/" in file else "." for file in unknown)
    if folders:
        print("  by folder: " + ", ".join(f"{d} {n}" for d, n in folders.most_common()))
    parts = [part for shard in shards for part in (shard["alone"], *shard["lanes"])]
    cut = Counter(file for part in parts for file, tests in part.items() if tests)
    seconds = [shard["seconds"] for shard in shards]
    print(
        f"{len(shards)} shards of {minutes(min(seconds))} to {minutes(max(seconds))}, "
        f"{args.at_once} files at a time; "
        f"{len(cut)} files cut into {sum(cut.values())} pieces"
    )
    marked_tests = sum(len(tests) for _, tests in alone.values())
    by_itself = sum(load + sum(tests.values()) for load, tests in alone.values())
    print(
        f"{marked_tests} tests of {len(alone)} files are marked serial and run "
        f"alone: {minutes(by_itself)}"
    )
