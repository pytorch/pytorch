"""
This script takes a pytest CSV file produced by pytest --csv foo.csv
and summarizes it into a more minimal CSV that is good for uploading
to Google Sheets.  We have been using this with dynamic shapes to
understand how many tests fail when we turn on dynamic shapes.  If
you have a test suite with a lot of skips or xfails, if force the
tests to run anyway, this can help you understand what the actual
errors things are failing with are.

The resulting csv is written to stdout.  An easy way to get the csv
onto your local file system is to send it to GitHub Gist:

    $ python scripts/analysis/format_test_csv.py foo.csv | gh gist create -

See also scripts/analysis/run_test_csv.sh
"""

import argparse
import csv
import subprocess
import sys

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--log-url", type=str, default="", help="URL of raw logs")
parser.add_argument("file", help="pytest CSV file to format")
args = parser.parse_args()

out = csv.writer(sys.stdout, dialect="excel")
hash = subprocess.check_output(
    "git rev-parse HEAD".split(" "), encoding="utf-8"
).rstrip()

out.writerow([hash, args.log_url, ""])

with open(args.file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] not in {"failed", "error"}:
            continue
        msg = row["message"].split("\n")[0]
        msg.replace(
            " - erroring out! It's likely that this is caused by data-dependent control flow or similar.",
            "",
        )
        msg.replace("\t", " ")
        # Feel free to edit this; the idea is to remove prefixes that are
        # just gooping up the resulting spreadsheet outpu
        name = row["name"].replace("test_make_fx_symbolic_exhaustive_", "")
        out.writerow([name, msg, ""])
