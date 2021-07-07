import argparse
import re
from typing import List


DEFAULT_MAX_TOKEN_COUNT = 1
MAX_TOKENS_CHECK_DIAG_NAME = "misc-max-tokens"
MAX_TOKENS_PRAGMA_PATERN = r"#pragma\s+clang\s+max_tokens_total\s+(\d+)"


def add_max_tokens_pragma(code: str, num_max_tokens: int) -> str:
    lines = code.split("\n")

    found_pragma = False
    pragma = f"#pragma clang max_tokens_total {num_max_tokens}"

    for idx, line in enumerate(lines):
        match = re.match(MAX_TOKENS_PRAGMA_PATERN, line.strip())
        if match:
            found_pragma = True
            token_count = match.group(1)
            if int(token_count) != num_max_tokens:
                lines[idx] = pragma

    if not found_pragma:
        lines = [pragma] + lines

    return "\n".join(lines)


def strip_max_tokens_pragmas(code: str) -> str:
    lines = code.split("\n")
    lines = [
        line
        for line in lines
        if re.match(MAX_TOKENS_PRAGMA_PATERN, line.strip()) is None
    ]
    return "\n".join(lines)


def add_max_tokens_pragma_to_files(files: List[str], num_max_tokens: int) -> None:
    for filename in files:
        with open(filename, "r+") as f:
            data = f.read()
            data = add_max_tokens_pragma(data, num_max_tokens)

            f.seek(0)
            f.write(data)
            f.truncate()


def strip_max_tokens_pragma_from_files(files: List[str]) -> None:
    for filename in files:
        with open(filename, "r+") as f:
            data = f.read()
            data = strip_max_tokens_pragmas(data)

            f.seek(0)
            f.write(data)
            f.truncate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add max_tokens_total pragmas to C/C++ source files"
    )
    parser.add_argument(
        "-n",
        "--num-max-tokens",
        default=DEFAULT_MAX_TOKEN_COUNT,
        help="Set the token count to this value",
        type=int,
    )
    parser.add_argument(
        "files", nargs="+", help="Add max_tokens_total pragmas to the specified files"
    )
    parser.add_argument(
        "-i", "--ignores", nargs="+", default=[], help="Ignore the specified files"
    )
    parser.add_argument(
        "-s",
        "--strip",
        action="store_true",
        help="Remove max_tokens_total pragmas from the input files",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()

    ignores = set(options.ignores)
    files = [filename for filename in options.files if filename not in ignores]
    if options.strip:
        strip_max_tokens_pragma_from_files(files)
    else:
        add_max_tokens_pragma_to_files(files, options.num_max_tokens)


if __name__ == "__main__":
    main()
