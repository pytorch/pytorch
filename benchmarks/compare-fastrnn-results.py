import argparse
import json
from collections import namedtuple


Result = namedtuple("Result", ["name", "base_time", "diff_time"])


def construct_name(fwd_bwd, test_name):
    bwd = "backward" in fwd_bwd
    suite_name = fwd_bwd.replace("-backward", "")
    return f"{suite_name}[{test_name}]:{'bwd' if bwd else 'fwd'}"


def get_times(json_data):
    r = {}
    for fwd_bwd in json_data:
        for test_name in json_data[fwd_bwd]:
            name = construct_name(fwd_bwd, test_name)
            r[name] = json_data[fwd_bwd][test_name]
    return r


parser = argparse.ArgumentParser("compare two pytest jsons")
parser.add_argument("base", help="base json file")
parser.add_argument("diff", help="diff json file")
parser.add_argument(
    "--format", default="md", type=str, help="output format (csv, md, json, table)"
)
args = parser.parse_args()

with open(args.base) as base:
    base_times = get_times(json.load(base))
with open(args.diff) as diff:
    diff_times = get_times(json.load(diff))

all_keys = set(base_times.keys()).union(diff_times.keys())
results = [
    Result(name, base_times.get(name, float("nan")), diff_times.get(name, float("nan")))
    for name in sorted(all_keys)
]

header_fmt = {
    "table": "{:48s} {:>13s} {:>15s} {:>10s}",
    "md": "| {:48s} | {:>13s} | {:>15s} | {:>10s} |",
    "csv": "{:s}, {:s}, {:s}, {:s}",
}
data_fmt = {
    "table": "{:48s} {:13.6f} {:15.6f} {:9.1f}%",
    "md": "| {:48s} | {:13.6f} | {:15.6f} | {:9.1f}% |",
    "csv": "{:s}, {:.6f}, {:.6f}, {:.2f}%",
}

if args.format in ["table", "md", "csv"]:
    header_fmt_str = header_fmt[args.format]
    data_fmt_str = data_fmt[args.format]
    print(header_fmt_str.format("name", "base time (s)", "diff time (s)", "% change"))
    if args.format == "md":
        print(header_fmt_str.format(":---", "---:", "---:", "---:"))
    for r in results:
        print(
            data_fmt_str.format(
                r.name,
                r.base_time,
                r.diff_time,
                (r.diff_time / r.base_time - 1.0) * 100.0,
            )
        )
elif args.format == "json":
    print(json.dumps(results))
else:
    raise ValueError("Unknown output format: " + args.format)
