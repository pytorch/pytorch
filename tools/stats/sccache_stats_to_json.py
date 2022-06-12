import json
import sys
import os
from typing import Any

GITHUB_JOB_ID = os.environ["OUR_GITHUB_JOB_ID"]


def parse_value(value: str) -> Any:
    # Take the value from a line of `sccache --show-stats` and try to parse
    # out a value
    try:
        return int(value)
    except ValueError:
        # sccache reports times as 0.000 s, so detect that here and strip
        # off the non-numeric parts
        if value.endswith(" s"):
            return float(value[: -len(" s")])

    return value


def get_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").lower()


STAT_NAMES = {
    "compile_requests",
    "compile_requests_executed",
    "cache_hits",
    "cache_misses",
    "cache_timeouts",
    "cache_read_errors",
    "forced_recaches",
    "cache_write_errors",
    "compilation_failures",
    "cache_errors",
    "non_cacheable_compilations",
    "non_cacheable_calls",
    "non_compilation_calls",
    "unsupported_compiler_calls",
    "average_cache_write",
    "average_cache_read_miss",
    "average_cache_read_hit",
    "failed_distributed_compilations",
}


if __name__ == "__main__":
    data = {"job_id": int(GITHUB_JOB_ID)}
    for line in sys.stdin:
        line = line.strip()
        values = [x.strip() for x in line.split("  ")]
        values = [x for x in values if x != ""]
        if len(values) == 2:
            name = get_name(values[0])
            if name in STAT_NAMES:
                data[name] = parse_value(values[1])

    print(json.dumps(data, indent=2))
