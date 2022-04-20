import argparse
import pathlib
import os
import shutil
import subprocess
import re
import sys
from sysconfig import get_paths as gp
from typing import List


from tools.linter.clang_tidy.run import run
from tools.linter.clang_tidy.generate_build_files import generate_build_files
from tools.linter.install.clang_tidy import INSTALLATION_PATH
from tools.linter.install.download_bin import PYTORCH_ROOT

# Returns '/usr/local/include/python<version number>'
def get_python_include_dir() -> str:
    return gp()['include']

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

    # There are source files include <torch/cuda.h>, <torch/torch.h> etc.
    # under torch/csrc/api/include folder. Since torch/csrc/api/include is not
    # a search path for clang-tidy, there will be clang-disagnostic errors
    # complaing those header files not found. Change the source code to include
    # full path like torch/csrc/api/include/torch/torch.h does not work well
    # since torch/torch.h includes torch/all.h which inturn includes more.
    # We would need recursively change mutliple files.
    # Adding the include path to the lint script should be a better solution.
    search_paths.append(
        os.path.join(PYTORCH_ROOT, "torch/csrc/api/include"),
    )
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
        "-torch/csrc/jit/serialization/mobile_bytecode_generated.cpp",
        "-torch/csrc/init_flatbuffer_module.cpp",
        "-torch/csrc/stub_with_flatbuffer.c",
        "-torch/csrc/onnx/init.cpp",
        "-torch/csrc/cuda/nccl.*",
        "-torch/csrc/cuda/python_nccl.cpp",
        "-torch/csrc/autograd/FunctionsManual.cpp",
        "-torch/csrc/generic/*.cpp",
        "-torch/csrc/jit/codegen/cuda/runtime/*",
        "-torch/csrc/deploy/interactive_embedded_interpreter.cpp",
        "-torch/csrc/deploy/interpreter/interpreter.cpp",
        "-torch/csrc/deploy/interpreter/interpreter.h",
        "-torch/csrc/deploy/interpreter/interpreter_impl.h",
        "-torch/csrc/deploy/interpreter/test_main.cpp",
        "-torch/csrc/deploy/test_deploy_python_ext.cpp",
    ],
    "paths": ["torch/csrc/"],
    "include-dir": [
        "/usr/lib/llvm-11/include/openmp",
        get_python_include_dir(),
        os.path.join(PYTORCH_ROOT, "third_party/pybind11/include")
    ] + clang_search_dirs(),
    "clang-tidy-exe": INSTALLATION_PATH,
    "compile-commands-dir": "build",
    "config-file": ".clang-tidy",
    "disable-progress-bar": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="clang-tidy wrapper script")
    parser.add_argument(
        "-e",
        "--clang-tidy-exe",
        default=DEFAULTS["clang-tidy-exe"],
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
        default=DEFAULTS["compile-commands-dir"],
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
    parser.add_argument("-q", "--quiet", action="store_true", help="Don't print output")
    parser.add_argument(
        "--config-file",
        default=DEFAULTS["config-file"],
        help="Path to a clang-tidy config file. Defaults to '.clang-tidy'.",
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
        "--disable-progress-bar",
        action="store_true",
        default=DEFAULTS["disable-progress-bar"],
        help="Disable the progress bar",
    )
    parser.add_argument(
        "extra_args", nargs="*", help="Extra arguments to forward to clang-tidy"
    )
    return parser.parse_args()


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
            + "$ python3 -m tools.linter.install.clang_tidy \n"
            + "from the pytorch folder"
        )
        raise RuntimeError(msg)

    result, _ = run(options)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
