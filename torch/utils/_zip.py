import argparse
from pathlib import Path
from zipfile import PyZipFile

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip py source")
    parser.add_argument("paths", nargs="*", help="Paths to zip.")
    parser.add_argument("--install_dir", help="Root directory for all output files")
    parser.add_argument("--zip_name", help="Output zip name")
    args = parser.parse_args()

    zip_file_name = args.install_dir + '/' + args.zip_name
    zf = PyZipFile(zip_file_name, mode='w')

    for p in args.paths:
        path = Path(p)
        if path.name in DENY_LIST:
            continue
        zf.write(p)
