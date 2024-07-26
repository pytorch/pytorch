import csv
import os
import re
import sys


# This script takes the logs produced by the benchmark scripts (e.g.,
# torchbench.py) and parses it into a CSV file that summarizes what
# is failing and why.  It is kept separate from the benchmark script
# emitting a more structured output as it is often more convenient
# to iterate quickly on log files offline instead of having to make
# a change to the benchmark script and then do a full sweep to see
# the updates.
#
# This script is not very well written, feel free to rewrite it as necessary

assert len(sys.argv) == 2

full_log = open(sys.argv[1]).read()

# If the log contains a gist URL, extract it so we can include it in the CSV
gist_url = ""
m = re.search(r"https://gist.github.com/[a-f0-9]+", full_log)
if m is not None:
    gist_url = m.group(0)

# Split the log into an entry per benchmark
entries = re.split(
    r"(?:cuda (?:train|eval) +([^ ]+)|WARNING:root:([^ ]+) failed to load)", full_log
)[1:]
# Entries schema example:
# `['hf_Bert', None, '
#  PASS\nTIMING: entire_frame_compile:1.80925 backend_compile:6e-05\nDynamo produced 1 graph(s) covering 367 ops\n']`


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


c = 0
i = 0

out = csv.DictWriter(
    sys.stdout,
    [
        "bench",
        "name",
        "result",
        "component",
        "context",
        "explain",
        "frame_time",
        "backend_time",
        "graph_count",
        "op_count",
        "graph_breaks",
        "unique_graph_breaks",
    ],
    dialect="excel",
)
out.writeheader()
out.writerow({"explain": gist_url})


# Sometimes backtraces will be in third party code, which results
# in very long file names.  Delete the absolute path in this case.
def normalize_file(f):
    if "site-packages/" in f:
        return f.split("site-packages/", 2)[1]
    else:
        return os.path.relpath(f)


# Assume we run torchbench, huggingface, timm_models in that order
# (as output doesn't say which suite the benchmark is part of)
# TODO: make this more robust

bench = "torchbench"

# 3 = 1 + number of matches in the entries split regex
for name, name2, log in chunker(entries, 3):
    if name is None:
        name = name2
    if name.startswith("Albert"):
        bench = "huggingface"
    elif name.startswith("adv_inc"):
        bench = "timm_models"

    # Payload that will go into the csv
    r = "UNKNOWN"
    explain = ""
    component = ""
    context = ""

    if "PASS" in log:
        r = "PASS"
    if "TIMEOUT" in log:
        r = "FAIL TIMEOUT"
    if "Accuracy failed" in log:
        r = "FAIL ACCURACY"

    # Attempt to extract out useful information from the traceback

    log = log.split(
        "The above exception was the direct cause of the following exception"
    )[0]
    split = log.split("Traceback (most recent call last)", maxsplit=1)
    if len(split) == 2:
        log = split[1]
    log = log.split("Original traceback:")[0]
    m = re.search(
        r'File "([^"]+)", line ([0-9]+), in .+\n +(.+)\n([A-Za-z]+(?:Error|Exception|NotImplementedError): ?.*)',
        log,
    )

    if m is not None:
        r = "FAIL"
        component = f"{normalize_file(m.group(1))}:{m.group(2)}"
        context = m.group(3)
        explain = f"{m.group(4)}"
    else:
        m = re.search(
            r'File "([^"]+)", line ([0-9]+), in .+\n +(.+)\nAssertionError', log
        )
        if m is not None:
            r = "FAIL"
            component = f"{normalize_file(m.group(1))}:{m.group(2)}"
            context = m.group(3)
            explain = "AssertionError"

    # Sometimes, the benchmark will say FAIL without any useful info
    # See https://github.com/pytorch/torchdynamo/issues/1910
    if "FAIL" in log:
        r = "FAIL"

    if r == "UNKNOWN":
        c += 1

    backend_time = None
    frame_time = None
    if "TIMING:" in log:
        result = re.search("TIMING:(.*)\n", log).group(1)
        split_str = result.split("backend_compile:")
        if len(split_str) == 2:
            backend_time = float(split_str[1])
            frame_time = float(split_str[0].split("entire_frame_compile:")[1])

    if "STATS:" in log:
        result = re.search("STATS:(.*)\n", log).group(1)
        # call_* op count: 970 | FakeTensor.__torch_dispatch__:35285 | ProxyTorchDispatchMode.__torch_dispatch__:13339
        split_all = result.split("|")
        # TODO: rewrite this to work with arbitrarily many stats

    graph_count = None
    op_count = None
    graph_breaks = None
    unique_graph_breaks = None
    if m := re.search(
        r"Dynamo produced (\d+) graphs covering (\d+) ops with (\d+) graph breaks \((\d+) unique\)",
        log,
    ):
        graph_count = m.group(1)
        op_count = m.group(2)
        graph_breaks = m.group(3)
        unique_graph_breaks = m.group(4)

    # If the context string is too long, don't put it in the CSV.
    # This is a hack to try to make it more likely that Google Sheets will
    # offer to split columns
    if len(context) > 78:
        context = ""

    # Temporary file names are meaningless, report it's generated code in this
    # case
    if "/tmp/" in component:
        component = "generated code"
        context = ""

    out.writerow(
        {
            "bench": bench,
            "name": name,
            "result": r,
            "component": component,
            "context": context,
            "explain": explain,
            "frame_time": frame_time,
            "backend_time": backend_time,
            "graph_count": graph_count,
            "op_count": op_count,
            "graph_breaks": graph_breaks,
            "unique_graph_breaks": unique_graph_breaks,
        }
    )
    i += 1

if c:
    print(f"failed to classify {c} entries", file=sys.stderr)
