import yaml
import glob
import re
import csv
import os
import argparse
import subprocess
import sys
import shutil


EXPORT_FIXES_PREFIX = "ctidy-fixes"
FILE_NAME_PATTERN = re.compile(f".*{EXPORT_FIXES_PREFIX}-(.*)")
FILE_NAME_GLOB = f"*{EXPORT_FIXES_PREFIX}*.yml"
MAX_TOKENS_REGEX_PATTERN = r"Number of tokens \((\d+)\).*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--output-directory",
        default="token_counts",
        help="Output directory",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="token_counts.csv",
        help="Outfile csv file",
        type=str,
    )

    return parser.parse_args()


def generate_export_fixes(outdir: str) -> None:
    config_file = os.path.join(
        "tools", "linter", "clang_tidy", "extra_configs", ".clang-tidy-max-tokens"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.linter.clang_tidy",
            "-e",
            ".clang-tidy-bin/clang-tidy",
            "--parallel",
            "--export-fixes",
            "--config-file",
            config_file,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    out_files = glob.glob(FILE_NAME_GLOB)
    for file in out_files:
        shutil.move(file, os.path.join(outdir, file))


def generate_csv(inputdir: str, outfile: str) -> None:
    counts = {}
    files = glob.glob(f"{inputdir}/{FILE_NAME_GLOB}")
    for file in files:
        with open(file) as f:
            data = yaml.safe_load(f)

            if data["Diagnostics"][0]["DiagnosticName"] != "misc-max-tokens":
                continue

            msg = data["Diagnostics"][0]["DiagnosticMessage"]["Message"]
            match = re.match(MAX_TOKENS_REGEX_PATTERN, msg)
            assert (
                match is not None
            ), f"Couldn't match regex {MAX_TOKENS_REGEX_PATTERN} in {msg}"
            counts[file] = int(match.group(1))

    with open(outfile, "w") as f:
        headers = ["filename", "token counts"]
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        for key, value in counts.items():
            filename = key
            match = re.match(FILE_NAME_PATTERN, filename)
            assert (
                match is not None
            ), f"Couldn't match regex {FILE_NAME_PATTERN} in {filename}"
            filename = os.path.splitext(match.group(1))[0]
            writer.writerow({"filename": filename, "token counts": value})


def main() -> None:
    options = parse_args()

    try:
        generate_export_fixes(outdir=options.output_directory)
        print(f"Generated export fixes files at {options.output_directory}")

        generate_csv(inputdir=options.output_directory, outfile=options.output_file)
        print(f"Generated csv file at {options.output_file}")
    except Exception as e:
        print("Failed to export token counts!")
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
