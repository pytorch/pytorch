import argparse
import enum
import json
import os
from collections import defaultdict


class Role(enum.Enum):
    CONFIGURE = 0
    GENERATE = 1
    CMAKEBUILD = 2
    CUSTOM = 3
    COMPILE = 4
    LINK = 5
    INSTALL = 6


def format_millis(millis):
    seconds, millis = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:>5}m {seconds:02}.{millis:03}s"


# print the top 10 items from a list of (name, duration) pairs
def log_top10(items, title):
    items.sort(key=lambda x: x[1], reverse=True)
    print(title)
    for (name, millis), _ in zip(items, range(10)):
        print(format_millis(millis), "|", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", help="<build-dir>/.cmake/instrumentation/v1/data/index/index.json"
    )
    parser.add_argument("--root", help="Root of the repository")
    cmdargs = parser.parse_args()

    with open(cmdargs.index) as index_file:
        index = json.load(index_file)
        trace_path = os.path.join(index["dataDir"], index["trace"])
    with open(trace_path) as trace_file:
        trace = json.load(trace_file)

    custom_command_times = []
    source_compile_times = []
    target_compile_times = defaultdict(lambda: 0)
    target_link_times = defaultdict(lambda: 0)
    stage_times = defaultdict(lambda: 0)

    print("*** Build Metrics Summary ***")
    for item in trace:
        args = item["args"]
        role = Role[args["role"].upper()]
        duration_ms = args["duration"]
        match role:
            case Role.CUSTOM:
                custom_command_times.append((args["command"], duration_ms))
            case Role.LINK:
                target_link_times[args["target"]] += duration_ms
            case Role.COMPILE:
                target_compile_times[args["target"]] += duration_ms
                source_compile_times.append(
                    (os.path.relpath(args["source"], cmdargs.root), duration_ms)
                )
            case _ as role:
                stage_times[role] += duration_ms

    print("\nStage durations")
    for stage in [Role.CONFIGURE, Role.GENERATE, Role.CMAKEBUILD, Role.INSTALL]:
        if stage in stage_times:
            print(f"{stage.name:>12}: {format_millis(stage_times[stage])}")
    log_top10(source_compile_times, "\nSource compile CPU time")
    log_top10(list(target_compile_times.items()), "\nTarget compile CPU time")
    log_top10(list(target_link_times.items()), "\nTarget link CPU time")
    log_top10(custom_command_times, "\nCustom command CPU time")
    print(f"\nLoad {trace_path} into ui.perfetto.dev for the complete trace")


if __name__ == "__main__":
    main()
