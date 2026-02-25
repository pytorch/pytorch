#!/usr/bin/env python3
"""Remove cherry-picked commits from a commitlist CSV.

Usage:
    python scripts/release_notes/remove_cherry_picks.py \\
        --commitlist scripts/release_notes/results/commitlist.csv \\
        --cherry-picks scripts/release_notes/results/cherry_picks_170119.csv

Reads the cherry picks CSV (output of parse_cherry_picks.py) and removes
matching rows from the commitlist, writing the result to a new file.
The original commitlist is never modified.
"""

import argparse
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Remove cherry-picked commits from a commitlist CSV"
    )
    parser.add_argument(
        "--commitlist",
        required=True,
        help="Path to the input commitlist.csv",
    )
    parser.add_argument(
        "--cherry-picks",
        required=True,
        help="Path to the cherry picks CSV (output of parse_cherry_picks.py)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output commitlist path (default: commitlist_no_cherry_picks.csv in same dir as commitlist)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Path for the removal log file (default: results/cherry_pick_removals.log relative to script)",
    )
    args = parser.parse_args()

    commitlist_path = Path(args.commitlist)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = commitlist_path.parent / "commitlist_no_cherry_picks.csv"

    script_dir = Path(__file__).resolve().parent
    if args.log:
        log_path = Path(args.log)
    else:
        results_dir = script_dir / "results"
        results_dir.mkdir(exist_ok=True)
        log_path = results_dir / "cherry_pick_removals.log"

    # Load cherry picks
    cherry_picks = []
    with open(args.cherry_picks, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cherry_picks.append(row)
    logger.info(f"Loaded {len(cherry_picks)} cherry pick entries")

    # Read commitlist as raw lines to preserve exact formatting
    with open(commitlist_path) as f:
        header_line = f.readline()
        commitlist_lines = f.readlines()
    logger.info(f"Loaded {len(commitlist_lines)} rows from {commitlist_path}")

    # Build a map from abbreviated hash to line for fast lookup
    hash_to_line = {}
    for line in commitlist_lines:
        abbrev_hash = line.split(",", 1)[0]
        hash_to_line[abbrev_hash] = line

    # Process cherry picks: find and remove matching rows
    removed = 0
    not_found = 0
    skipped = 0
    log_entries = []

    for cp in cherry_picks:
        sha = cp.get("commit_sha", "")
        pr_number = cp.get("pr_number", "")
        pr_title = cp.get("pr_title", "")
        label = f"PR #{pr_number} ({pr_title})" if pr_number else "N/A"

        if not sha:
            skipped += 1
            log_entries.append(f"SKIPPED (no hash): {label}")
            continue

        # Find matching abbreviated hash via prefix match
        matched_hash = None
        for abbrev_hash in hash_to_line:
            if sha.startswith(abbrev_hash) or abbrev_hash.startswith(sha):
                matched_hash = abbrev_hash
                break

        if matched_hash:
            removed += 1
            log_entries.append(f"REMOVED: {matched_hash} -> {label}")
            del hash_to_line[matched_hash]
        else:
            not_found += 1
            log_entries.append(f"NOT FOUND: {sha[:11]} -> {label}")
            logger.warning(f"Commit {sha[:11]} ({label}) not found in commitlist")

    # Write new commitlist
    with open(output_path, "w") as f:
        f.write(header_line)
        for line in commitlist_lines:
            abbrev_hash = line.split(",", 1)[0]
            if abbrev_hash in hash_to_line:
                f.write(line)

    logger.info(
        f"Wrote {len(hash_to_line)} rows to {output_path} "
        f"(removed {removed}, not found {not_found}, skipped {skipped})"
    )

    # Write log file
    with open(log_path, "w") as f:
        for entry in log_entries:
            f.write(entry + "\n")
    logger.info(f"Wrote removal log to {log_path}")


if __name__ == "__main__":
    main()
