# mypy: allow-untyped-defs
import argparse
import glob
import os
from pathlib import Path
from zipfile import ZipFile


# Exclude some standard library modules to:
# 1. Slim down the final zipped file size
# 2. Remove functionality we don't want to support.
DENY_LIST = [
    # Interface to unix databases
    "dbm",
    # ncurses bindings (terminal interfaces)
    "curses",
    # Tcl/Tk GUI
    "tkinter",
    "tkinter",
    # Tests for the standard library
    "test",
    "tests",
    "idle_test",
    "__phello__.foo.py",
    # importlib frozen modules. These are already baked into CPython.
    "_bootstrap.py",
    "_bootstrap_external.py",
]

strip_file_dir = ""


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def write_to_zip(file_path, strip_file_path, zf, prepend_str=""):
    stripped_file_path = prepend_str + remove_prefix(file_path, strip_file_dir + "/")
    path = Path(stripped_file_path)
    if path.name in DENY_LIST:
        return
    zf.write(file_path, stripped_file_path)


def main() -> None:
    global strip_file_dir
    parser = argparse.ArgumentParser(description="Zip py source")
    parser.add_argument("paths", nargs="*", help="Paths to zip.")
    parser.add_argument(
        "--install-dir", "--install_dir", help="Root directory for all output files"
    )
    parser.add_argument(
        "--strip-dir",
        "--strip_dir",
        help="The absolute directory we want to remove from zip",
    )
    parser.add_argument(
        "--prepend-str",
        "--prepend_str",
        help="A string to prepend onto all paths of a file in the zip",
        default="",
    )
    parser.add_argument("--zip-name", "--zip_name", help="Output zip name")

    args = parser.parse_args()

    zip_file_name = args.install_dir + "/" + args.zip_name
    strip_file_dir = args.strip_dir
    prepend_str = args.prepend_str
    zf = ZipFile(zip_file_name, mode="w")

    for p in sorted(args.paths):
        if os.path.isdir(p):
            files = glob.glob(p + "/**/*.py", recursive=True)
            for file_path in sorted(files):
                # strip the absolute path
                write_to_zip(
                    file_path, strip_file_dir + "/", zf, prepend_str=prepend_str
                )
        else:
            write_to_zip(p, strip_file_dir + "/", zf, prepend_str=prepend_str)


if __name__ == "__main__":
    main()  # pragma: no cover
