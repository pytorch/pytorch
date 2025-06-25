import sys
import argparse
import re
from collections import Counter

parser = argparse.ArgumentParser(description="tlparse: summarize graph breaks in Dynamo logs")
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Show each graph break individually instead of a summary"
)
args = parser.parse_args()

break_lines = []
capture_next = False

for raw_line in sys.stdin:
    # Strip leading log info, e.g. "W0626 ...] [0/0]"
    line = raw_line.split("]")[-1].strip() if "]" in raw_line else raw_line.strip()

    if "Graph break: from user code at" in line:
        capture_next = True
        continue

    if capture_next and line.startswith("File"):
        break_lines.append(line)
        capture_next = False

# Output result
if args.verbose:
    for line in break_lines:
        print(line)
else:
    counts = Counter(break_lines)
    for location, count in counts.items():
        print(f"{location} ({count} occurrence{'s' if count > 1 else ''})")
