import argparse
import shutil
import sys
import os


from tools.linter.utils import run_cmd, CommandResult
from tools.linter.lint import Linter


class Shellcheck(Linter):
    name = "shellcheck"
    exe = shutil.which(name)
    options = argparse.Namespace(
        paths=[
            ".extracted_scripts",
            ".jenkins/pytorch"
        ],
        glob=["*.sh"],
        regex=[]
    )

    def build_parser(self, parser):
        return parser

    async def run(self, files, line_filters=None, options=options) -> CommandResult:
        if len(files) == 0:
            return CommandResult(0, "", "")

        if os.path.exists(".extracted_scripts"):
            shutil.rmtree(".extracted_scripts")
        result = await run_cmd([sys.executable, "tools/extract_scripts.py", "--out=.extracted_scripts"])

        if result.failed():
            return result

        result = await run_cmd([self.exe, "--external-sources", *files])
        if result.failed():
            result.stdout += (
                "\nShellCheck gave a nonzero exit code. Please fix the warnings "
                "listed above. Note that if a path in one of the above warning "
                "messages starts with .extracted_scripts/ then that means it "
                "is referring to a shell script embedded within another file, "
                "whose path is given by the path components immediately "
                "following the .extracted_scripts/ prefix."
            )
        return result
