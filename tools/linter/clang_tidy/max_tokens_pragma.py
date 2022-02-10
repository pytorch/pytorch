import argparse
import re
from typing import List


# > Why is DEFAULT_MAX_TOKEN_COUNT set to 1?
#
# clang-tidy doesn't have a direct way to query for token counts in the
# codebase. The workaround is to set the max token count to 1. This will cause
# clang-tidy to output a warning with the actual token count of the file.
#
# A non-destructive way to set the max token count to 1 would be to pass it
# through the -fmax-tokens option. However, this flag will be overridden if here
# exists a #pragma max_tokens_total statement in the file. This necessitates a
# destructive way to set the max token count to 1.
DEFAULT_MAX_TOKEN_COUNT = 1
MAX_TOKENS_CHECK_DIAG_NAME = "misc-max-tokens"
MAX_TOKENS_PRAGMA_PATTERN = r"^#pragma\s+clang\s+max_tokens_total\s+(\d+)$"


def add_max_tokens_pragma(code: str, num_max_tokens: int) -> str:
    lines = code.splitlines()

    found_pragma = False
    pragma = f"#pragma clang max_tokens_total {num_max_tokens}"

    for idx, line in enumerate(lines):
        match = re.match(MAX_TOKENS_PRAGMA_PATTERN, line.strip())
        if match:
            found_pragma = True
            token_count = match.group(1)
            if int(token_count) != num_max_tokens:
                lines[idx] = pragma

    if not found_pragma:
        lines = [pragma] + lines

    return "\n".join(lines)


def strip_max_tokens_pragmas(code: str) -> str:
    lines = code.splitlines()
    lines = [
        line
        for line in lines
        if re.match(MAX_TOKENS_PRAGMA_PATTERN, line.strip()) is None
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
        "-i", "--ignore", nargs="+", default=[], help="Ignore the specified files"
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

    ignored = set(options.ignore)
    files = [filename for filename in options.files if filename not in ignored]
    if options.strip:
        strip_max_tokens_pragma_from_files(files)
    else:
        add_max_tokens_pragma_to_files(files, options.num_max_tokens)


if __name__ == "__main__":
    main()
