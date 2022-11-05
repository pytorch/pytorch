import re
import os
import csv
import sys
import subprocess

full_log = open(sys.argv[1], 'r').read()
gist_url = open("url.log", "r").read().rstrip()
hash = subprocess.check_output('git rev-parse HEAD'.split(' '), encoding='utf-8').rstrip()

entries = re.split(r"Running (torchbench|huggingface|timm_models)\.py ([^.]+)\.\.\.", full_log)[1:]

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

c = 0
i = 0

out = csv.writer(sys.stdout, dialect='excel')
out.writerow(["", hash, "", "", "", gist_url])

def normalize_file(f):
    if "site-packages/" in f:
        return f.split("site-packages/", 2)[1]
    else:
        return os.path.relpath(f)

for bench, name, log in chunker(entries, 3):
    r = "UNKNOWN"
    explain = ""
    component = ""
    if "PASS" in log:
        r = "PASS"
    if "TIMEOUT" in log:
        r = "FAIL TIMEOUT"
    if "Accuracy failed" in log:
        r = "FAIL ACCURACY"
    log = log.split("The above exception was the direct cause of the following exception")[0]
    split = log.split("Traceback (most recent call last)", maxsplit=1)
    if len(split) == 2:
        log = split[1]
    log = log.split("Original traceback:")[0]
    errors = re.findall(r'[A-Za-z]+(?:Error|Exception): .+', log)
    if errors:
        r = "FAIL"
        explain = errors[-1]
        files = re.findall('File "([^"]+)"', log)
        if files:
            for f in reversed(files):
                if f != "<string>":
                    component = normalize_file(f)
                    break
            else:
                if "ERROR RUNNING GUARDS" in log or "NULL ERROR" in log:
                    component = "<dynamo guards>"
    else:
        m = re.search(r'File "([^"]+)", line ([0-9]+), in .+\n +(.+)\nAssertionError', log)
        if m is not None:
            r = "FAIL"
            component = normalize_file(m.group(1))
            explain = f"AssertionError on line {m.group(2)}: {m.group(3)}"
    if r == "UNKNOWN":
        c += 1
    out.writerow([bench, name, "", r, component, explain])
    i += 1

if c:
    print(f"failed to classify {c} entries", file=sys.stderr)
