import fileinput
import sys
import os
from typing import Any

from tools.stats.scribe import (
    schema_from_sample,
    rds_write,
    register_rds_schema,
)


def sprint(*args: Any) -> None:
    print("[sccache_stats]", *args, file=sys.stderr)


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
    if os.getenv("IS_GHA", "0") == "1":
        data = {}
        if len(sys.argv) == 2:
            with open(sys.argv[1]) as f:
                lines = f.readlines()
        else:
            lines = list(fileinput.input())
        for line in lines:
            line = line.strip()
            values = [x.strip() for x in line.split("  ")]
            values = [x for x in values if x != ""]
            if len(values) == 2:
                name = get_name(values[0])
                if name in STAT_NAMES:
                    data[name] = parse_value(values[1])

        # The data from sccache is always the same so this should be fine, if it
        # ever changes we will probably need to break this out so the fields
        # we want are hardcoded
        register_rds_schema("sccache_stats", schema_from_sample(data))

        rds_write("sccache_stats", [data])
        sprint("Wrote sccache stats to DB")
    else:
        sprint("Not in GitHub Actions, skipping")
