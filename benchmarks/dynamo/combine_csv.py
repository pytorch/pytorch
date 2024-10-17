# This script takes csvs produced by parse_logs.py and combines them
# into a single CSV file

import ast
import csv
import sys
from collections import defaultdict


assert len(sys.argv) == 3

RESULTS = defaultdict(dict)

for side, f in zip(["static", "dynamic"], sys.argv[1:]):
    with open(f) as f:
        reader = csv.DictReader(f)
        for row in reader:
            RESULTS[(row["bench"], row["name"])][side] = row

fields = ["frame_time", "graph_breaks"]

out = csv.DictWriter(
    sys.stdout,
    ["bench", "name"] + [f"delta_{n}" for n in fields] + ["static_url", "dynamic_url"],
    dialect="excel",
)
out.writeheader()

for (bench, name), sides in RESULTS.items():
    if "static" not in sides:
        continue
    if "dynamic" not in sides:
        continue
    if not name:
        out.writerow(
            {
                "static_url": sides["static"]["explain"],
                "dynamic_url": sides["dynamic"]["explain"],
            }
        )
        continue
    row = {"bench": bench, "name": name}
    for f in fields:
        try:
            static = ast.literal_eval(sides["static"][f])
            dynamic = ast.literal_eval(sides["dynamic"][f])
        except SyntaxError:
            continue
        row[f"delta_{f}"] = dynamic - static
    out.writerow(row)
