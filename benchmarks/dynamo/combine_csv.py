import ast
import csv
import sys
from collections import defaultdict
from typing import Dict, Tuple


def combine_csvs(static_file: str, dynamic_file: str) -> None:
    """Combine CSV files produced by parse_logs.py into a single CSV output."""

    # Maps (bench, name) to their static and dynamic results
    results: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)

    # Read and store data from static and dynamic files
    for side, filename in [("static", static_file), ("dynamic", dynamic_file)]:
        with open(filename) as file:
            reader = csv.DictReader(file)
            for row in reader:
                results[(row["bench"], row["name"])][side] = row

    fields = ["frame_time", "graph_breaks"]

    # Prepare the CSV writer
    out = csv.DictWriter(
        sys.stdout,
        ["bench", "name"]
        + [f"delta_{n}" for n in fields]
        + ["static_url", "dynamic_url"],
        dialect="excel",
    )
    out.writeheader()

    # Process and write combined results
    for (bench, name), sides in results.items():
        if "static" not in sides or "dynamic" not in sides:
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
        for field in fields:
            try:
                static_value = ast.literal_eval(sides["static"][field])
                dynamic_value = ast.literal_eval(sides["dynamic"][field])
            except SyntaxError:
                continue
            row[f"delta_{field}"] = dynamic_value - static_value
        out.writerow(row)


def main():
    if len(sys.argv) != 3:
        raise ValueError("Expected two arguments: static_file and dynamic_file")

    static_file, dynamic_file = sys.argv[1:3]
    combine_csvs(static_file, dynamic_file)


if __name__ == "__main__":
    main()
