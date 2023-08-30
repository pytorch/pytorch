import csv
import os
import re
import sys


def extract_gist_url(log_content):
    """Extract a gist URL from the log content."""
    m = re.search(r"https://gist.github.com/[a-f0-9]+", log_content)
    return m.group(0) if m else ""


def chunker(seq, size):
    """Break the sequence into chunks of a given size."""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def normalize_file(file_path):
    """Shorten the file path if it's too long."""
    if "site-packages/" in file_path:
        return file_path.split("site-packages/", 2)[1]
    return os.path.relpath(file_path)


def determine_benchmark(name):
    """Determine the type of benchmark based on the name of the first model in the list"""
    if name.startswith("Albert"):
        return "huggingface"
    elif name.startswith("adv_inc"):
        return "timm_models"
    return "torchbench"


def parse_log_entry(name, log):
    """Parse individual log entries for errors, timings, etc."""
    # Initialize default values
    result_data = {
        "result": "UNKNOWN",
        "component": "",
        "context": "",
        "explain": "",
        "frame_time": None,
        "backend_time": None,
        "graph_count": None,
        "op_count": None,
        "graph_breaks": None,
        "unique_graph_breaks": None,
    }

    # Parse for specific logs
    # TODO: Extend the log parsing as needed
    if "PASS" in log:
        result_data["result"] = "PASS"
    if "TIMEOUT" in log:
        result_data["result"] = "FAIL TIMEOUT"
    if "Accuracy failed" in log:
        result_data["result"] = "FAIL ACCURACY"
    if "FAIL" in log:
        result_data["result"] = "FAIL"

    # Extract errors and details
    error_pattern = r'File "([^"]+)", line ([0-9]+), in .+\n +(.+)\n([A-Za-z]+(?:Error|Exception|NotImplementedError): ?.*)'
    m = re.search(error_pattern, log)
    if m:
        result_data["component"] = f"{normalize_file(m.group(1))}:{m.group(2)}"
        result_data["context"] = m.group(3)
        result_data["explain"] = m.group(4)

    # Parse for timings
    timing_match = re.search("TIMING:(.*)\n", log)
    if timing_match:
        result = timing_match.group(1)
        frame, backend = result.split("backend_compile:")
        result_data["frame_time"] = float(frame.split("entire_frame_compile:")[1])
        result_data["backend_time"] = float(backend)

    # Parse for grahh break stats
    stats_match = re.search(
        r"Dynamo produced (\d+) graphs covering (\d+) ops with (\d+) graph breaks \((\d+) unique\)",
        log,
    )
    if stats_match:
        result_data["graph_count"] = stats_match.group(1)
        result_data["op_count"] = stats_match.group(2)
        result_data["graph_breaks"] = stats_match.group(3)
        result_data["unique_graph_breaks"] = stats_match.group(4)

    return result_data


def main():
    assert len(sys.argv) == 2

    with open(sys.argv[1]) as file:
        full_log = file.read()

    gist_url = extract_gist_url(full_log)

    entries = re.split(
        r"(?:cuda (?:train|eval) +([^ ]+)|WARNING:root:([^ ]+) failed to load)",
        full_log,
    )[1:]

    # Initialize CSV writer
    headers = [
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
    ]
    out = csv.DictWriter(sys.stdout, headers, dialect="excel")
    out.writeheader()
    out.writerow({"explain": gist_url})

    unclassified_count = 0

    for name, name2, log in chunker(entries, 3):
        if name is None:
            name = name2

        bench = determine_benchmark(name)
        parsed_data = parse_log_entry(name, log)

        # Construct CSV row
        csv_data = {
            "bench": bench,
            "name": name,
        }
        csv_data.update(parsed_data)
        out.writerow(csv_data)

        if parsed_data["result"] == "UNKNOWN":
            unclassified_count += 1

    if unclassified_count:
        print(f"Failed to classify {unclassified_count} entries", file=sys.stderr)


if __name__ == "__main__":
    main()
