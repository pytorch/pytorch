import argparse
import subprocess
import shutil
import re
import pathlib
from typing import List

from tools.linter.lint import Linter
from tools.linter.utils import CommandResult
from tools.linter.clang_tidy.run import run
from tools.linter.clang_tidy.generate_build_files import generate_build_files
from tools.linter.install.clang_tidy import INSTALLATION_PATH


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


class ClangTidy(Linter):
    name = "clang-tidy"
    exe = INSTALLATION_PATH
    options = argparse.Namespace(
        # required options
        glob=[
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
        paths=["torch/csrc/"],
        regex=[r"^.*\.c(c|pp)?$"],
        clang_tidy_exe=exe,
        compile_commands_dir="build",
        dry_run=None,
        quiet=False,
        config_file=".clang-tidy",
        print_include_paths=False,
        include_dir=["/usr/lib/llvm-11/include/openmp"] + clang_search_dirs(),
        suppress_diagnostics=False,
        disable_progress_bar=False,
        extra_args=[],
    )

    def build_parser(self, parser):
        parser.add_argument(
            "-e",
            "--clang-tidy-exe",
            default=self.options.clang_tidy_exe,
            help="Path to clang-tidy executable",
        )
        parser.add_argument(
            "-c",
            "--compile-commands-dir",
            default=self.options.compile_commands_dir,
            help="Path to the folder containing compile_commands.json",
        )
        parser.add_argument(
            "-n",
            "--dry-run",
            default=self.options.dry_run,
            action="store_true",
            help="Only show the command to be executed, without running it",
        )
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            default=self.options.quiet,
            help="Don't print output",
        )
        parser.add_argument(
            "--config-file",
            default=self.options.config_file,
            help="Path to a clang-tidy config file. Defaults to '.clang-tidy'.",
        )
        parser.add_argument(
            "--print-include-paths",
            action="store_true",
            default=self.options.print_include_paths,
            help="Print the search paths used for include directives",
        )
        parser.add_argument(
            "-I",
            "--include-dir",
            action="append",
            default=self.options.include_dir,
            help="Add the specified directory to the search path for include files",
        )
        parser.add_argument(
            "-s",
            "--suppress-diagnostics",
            action="store_true",
            default=self.options.suppress_diagnostics,
            help="Add NOLINT to suppress clang-tidy violations",
        )
        parser.add_argument(
            "--disable-progress-bar",
            default=self.options.disable_progress_bar,
            help="Disable progress bar (only works on linters that support this feature)",
        )
        parser.add_argument(
            "--extra_args",
            nargs="*",
            help="Extra arguments to forward to clang-tidy",
            default=self.options.extra_args,
        )
        return parser

    async def run(self, files, line_filters=None, options=options) -> CommandResult:
        if not pathlib.Path("build").exists():
            print("Generating build files")
            result = await generate_build_files()
            if result.failed():
                return result

        result, _ = await run(files, line_filters, options)

        return result
