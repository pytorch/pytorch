import re
import os
import csv
import sys

full_log = open(sys.argv[1], 'r').read()

entries = re.split(r"Running ([^.]+)\.\.\.", full_log)[1:]
print(entries[:4])

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

c = 0

out = csv.writer(sys.stdout, dialect='excel')

for name, log in chunker(entries, 2):
    r = "UNKNOWN"
    explain = ""
    component = ""
    if "PASS" in log:
        r = "PASS"
    if "TIMEOUT" in log:
        r = "FAIL TIMEOUT"
    if "Accuracy failed" in log:
        r = "FAIL ACCURACY"
    errors = re.findall(r'[A-Za-z]+Error: .+', log)
    if errors:
        r = "FAIL"
        explain = errors[-1]
        files = re.findall('File "([^"]+)"', log)
        if files:
            for f in reversed(files):
                if f != "<string>":
                    if "site-packages/" in f:
                        component = f.split("site-packages/", 2)[1]
                    else:
                        component = os.path.relpath(f)
                    break
            else:
                if "ERROR RUNNING GUARDS" in log or "NULL ERROR" in log:
                    component = "<dynamo guards>"
    if r == "UNKNOWN":
        c += 1
    if r == "FAIL":
        out.writerow([name, "", r, component, explain])

if c:
    print(f"failed to classify {c} entries", file=sys.stderr)
