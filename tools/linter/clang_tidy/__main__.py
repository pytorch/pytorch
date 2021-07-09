import argparse
import pathlib
import os
import shutil
import subprocess
import re
from typing import List


from tools.linter.clang_tidy.run import run
from tools.linter.clang_tidy.generate_build_files import generate_build_files


def clang_search_dirs() -> List[str]:
    # Compilers are ordered based on fallback preference
    # We pick the first one that is available on the system
    compilers = ["clang", "gcc", "cpp", "cc"]
    compilers = [c for c in compilers if shutil.which(c) is not None]
    if len(compilers) == 0:
        raise RuntimeError(f"None of {compilers} were found")
    compiler = compilers[0]

    result = subprocess.run(
        [compiler, "-E", "-x", "c++", "-", "-v"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    stderr = result.stderr.decode().strip().split("\n")
    search_start = r"#include.*search starts here:"
    search_end = r"End of search list."

    append_path = False
    search_paths = []
    for line in stderr:
        if re.match(search_start, line):
            if append_path:
                continue
            else:
                append_path = True
        elif re.match(search_end, line):
            break
        elif append_path:
            search_paths.append(line.strip())

    return search_paths


DEFAULTS = {
    "glob": [
        # The negative filters below are to exclude files that include onnx_pb.h or
        # caffe2_pb.h, otherwise we'd have to build protos as part of this CI job.
        # FunctionsManual.cpp is excluded to keep this diff clean. It will be fixed
        # in a follow up PR.
        # /torch/csrc/generic/*.cpp is excluded because those files aren't actually built.
        # deploy/interpreter files are excluded due to using macros and other techniquies
        # that are not easily converted to accepted c++
        "-torch/csrc/jit/passes/onnx/helper.cpp",
        "-torch/csrc/jit/passes/onnx/shape_type_inference.cpp",
        "-torch/csrc/jit/serialization/onnx.cpp",
        "-torch/csrc/jit/serialization/export.cpp",
        "-torch/csrc/jit/serialization/import.cpp",
        "-torch/csrc/jit/serialization/import_legacy.cpp",
        "-torch/csrc/onnx/init.cpp",
        "-torch/csrc/cuda/nccl.*",
        "-torch/csrc/cuda/python_nccl.cpp",
        "-torch/csrc/autograd/FunctionsManual.cpp",
        "-torch/csrc/generic/*.cpp",
        "-torch/csrc/jit/codegen/cuda/runtime/*",
        "-torch/csrc/deploy/interpreter/interpreter.cpp",
        "-torch/csrc/deploy/interpreter/interpreter.h",
        "-torch/csrc/deploy/interpreter/interpreter_impl.h",
        "-torch/csrc/deploy/interpreter/test_main.cpp",
    ],
    "paths": ["torch/csrc/"],
    "include-dir": ["/usr/lib/llvm-11/include/openmp"] + clang_search_dirs(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Clang-Tidy (on your Git changes)")
    parser.add_argument(
        "-e",
        "--clang-tidy-exe",
        default="clang-tidy",
        help="Path to clang-tidy executable",
    )
    parser.add_argument(
        "-g",
        "--glob",
        action="append",
        default=DEFAULTS["glob"],
        help="Only lint files that match these glob patterns "
        "(see documentation for `fnmatch` for supported syntax)."
        "If a pattern starts with a - the search is negated for that pattern.",
    )
    parser.add_argument(
        "-x",
        "--regex",
        action="append",
        default=[],
        help="Only lint files that match these regular expressions (from the start of the filename). "
        "If a pattern starts with a - the search is negated for that pattern.",
    )
    parser.add_argument(
        "-c",
        "--compile-commands-dir",
        default="build",
        help="Path to the folder containing compile_commands.json",
    )
    parser.add_argument(
        "--diff-file",
        help="File containing diff to use for determining files to lint and line filters",
    )
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        default=DEFAULTS["paths"],
        help="Lint only the given paths (recursively)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only show the command to be executed, without running it",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--config-file",
        help="Path to a clang-tidy config file. Defaults to '.clang-tidy'.",
    )
    parser.add_argument(
        "-k",
        "--keep-going",
        action="store_true",
        help="Don't error on compiler errors (clang-diagnostic-error)",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        action="store_true",
        help="Run clang tidy in parallel per-file (requires ninja to be installed).",
    )
    parser.add_argument(
        "--print-include-paths",
        action="store_true",
        help="Print the search paths used for include directives",
    )
    parser.add_argument(
        "-I",
        "--include-dir",
        action="append",
        default=DEFAULTS["include-dir"],
        help="Add the specified directory to the search path for include files",
    )
    parser.add_argument(
        "-s",
        "--suppress-diagnostics",
        action="store_true",
        help="Add NOLINT to suppress clang-tidy violations",
    )
    parser.add_argument(
        "extra_args", nargs="*", help="Extra arguments to forward to clang-tidy"
    )
    return parser.parse_args()


def main() -> None:
    if not pathlib.Path("build").exists():
        generate_build_files()
    options = parse_args()

    # Check if clang-tidy executable exists
    exists = os.access(options.clang_tidy_exe, os.X_OK) or shutil.which(
        options.clang_tidy_exe
    )
    if not exists:
        msg = (
            "Could not find 'clang-tidy' binary\n"
            "You can install it by running:\n"
            "   python3 tools/linter/install/clang_tidy.py"
        )
        print(msg)
        exit(1)

    return_code = run(options)
    if return_code != 0:
        raise RuntimeError("Warnings found in clang-tidy output!")


main()
