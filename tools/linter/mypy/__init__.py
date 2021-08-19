import argparse
import shutil
import glob
import sys


from tools.linter.utils import CommandResult, run_cmd
from tools.linter.lint import Linter


class Mypy(Linter):
    name = "mypy"
    exe = shutil.which(name)
    options = argparse.Namespace(paths=["."], glob=["*.py"], regex=[])

    def build_parser(self, parser):
        return parser

    async def run(self, files, line_filters=None, options=options):
        result = CommandResult(0, "", "")

        # Run autogen code
        time = shutil.which("time")
        result += await run_cmd(
            [time, sys.executable, "-mtools.generate_torch_version", "--is_debug=false"]
        )

        result += await run_cmd(
            [
                time,
                sys.executable,
                "-mtools.codegen.gen",
                "-s",
                "aten/src/ATen",
                "-d",
                "build/aten/src/ATen",
            ]
        )

        result += await run_cmd(
            [
                time,
                sys.executable,
                "-mtools.pyi.gen_pyi",
                "--native-functions-path",
                "aten/src/ATen/native/native_functions.yaml",
                "--deprecated-functions-path",
                "tools/autograd/deprecated.yaml",
            ]
        )

        # Don't run mypy if autogen failed
        if result.failed():
            return result

        if options.verbose:
            print("Autogen output:", result)

        # Reset result. We don't want to show autogen output
        result = CommandResult(0, "", "")

        config_files = glob.glob("mypy*.ini")

        for config in config_files:
            result += await run_cmd([self.exe, "--config", config, *files])

        return result
