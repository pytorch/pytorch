import argparse
import re
from typing import List


DEFAULT_MAX_TOKEN_COUNT = 1
MAX_TOKENS_CHECK_DIAG_NAME = "misc-max-tokens"
MAX_TOKENS_PRAGMA_PATERN = r"#pragma\s+clang\s+max_tokens_total\s+(\d+)"


def add_max_tokens_pragma(files: List[str], num_max_tokens: int) -> None:
    for filename in files:
        with open(filename, "r+") as f:
            lines = f.readlines()

            f.seek(0)

            found_pragma = False
            pragma = f"#pragma clang max_tokens_total {num_max_tokens}\n"

            for idx, line in enumerate(lines):
                match = re.match(MAX_TOKENS_PRAGMA_PATERN, line)
                if match:
                    found_pragma = True
                    token_count = match.group(1)
                    if token_count != num_max_tokens:
                        lines[idx] = pragma

            if not found_pragma:
                f.write(pragma)

            for line in lines:
                f.write(line)

            f.truncate()


def strip_max_tokens_pragma(files: List[str]) -> None:
    for filename in files:
        with open(filename, "r+") as f:
            lines = f.readlines()
            lines = [
                line
                for line in lines
                if re.match(MAX_TOKENS_PRAGMA_PATERN, line) is None
            ]
            f.seek(0)
            for line in lines:
                f.write(line)
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
    parser.add_argument("-i", "--ignores", nargs="+", help="Ignore the specified files")
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
        strip_max_tokens_pragma(files)
    else:
        add_max_tokens_pragma(files, options.num_max_tokens)


if __name__ == "__main__":
    main()
