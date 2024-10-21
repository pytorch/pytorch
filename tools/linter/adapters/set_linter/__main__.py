import sys

from .configs import get_args
from .files import resolve_python_files
from .lint_file import lint_file


CONFIG_FILE = "setlint.json"


def main() -> None:
    args = get_args(CONFIG_FILE)
    if not (python_files := resolve_python_files(args.files, args.exclude)):
        sys.exit("No files selected")

    for f in python_files:
        lint_file(f, args)


if __name__ == "__main__":
    main()
