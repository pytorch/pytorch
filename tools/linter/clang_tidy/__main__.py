import argparse
import pathlib
import os
import shutil
import subprocess
import re
import sys
from typing import List


from tools.linter.clang_tidy.run import run
from tools.linter.clang_tidy.generate_build_files import generate_build_files
from tools.linter.install.clang_tidy import INSTALLATION_PATH

def main() -> None:
    options = parse_args()

    if not pathlib.Path("build").exists():
        generate_build_files()

    # Check if clang-tidy executable exists
    exists = os.access(options.clang_tidy_exe, os.X_OK)

    if not exists:
        msg = (
            f"Could not find '{options.clang_tidy_exe}'\n"
            + "We provide a custom build of clang-tidy that has additional checks.\n"
            + "You can install it by running:\n"
            + "$ python3 tools/linter/install/clang_tidy.py"
        )
        raise RuntimeError(msg)

    result, _ = run(options)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
