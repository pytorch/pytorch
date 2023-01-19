import csv
import os
import re
import subprocess
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

full_log = open(sys.argv[1], "r").read()

# If the log contains a gist URL, extract it so we can include it in the CSV
gist_url = ""
m = re.search(r"https://gist.github.com/[a-f0-9]+", full_log)
if m is not None:
    gist_url = m.group(0)

# Record the current commit hash for ease of reproducibility
hash = subprocess.check_output(
    "git rev-parse HEAD".split(" "), encoding="utf-8"
).rstrip()

# Split the log into an entry per benchmark
entries = re.split(
    r"(?:cuda (?:train|eval) +([^ ]+)|WARNING:root:([^ ]+) failed to load)", full_log
)[1:]


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


c = 0
i = 0

out = csv.writer(sys.stdout, dialect="excel")
out.writerow(["", hash, "", "", "", "", gist_url])

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

    out.writerow([bench, name, "", r, component, context, explain])
    i += 1

if c:
    print(f"failed to classify {c} entries", file=sys.stderr)
