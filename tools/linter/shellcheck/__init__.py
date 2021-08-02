import argparse
import shutil


from tools.linter.utils import run_cmd, CommandResult
from tools.linter.lint import Linter


class Shellcheck(Linter):
    name = "shellcheck"
    exe = shutil.which(name)
    options = argparse.Namespace(paths=["."], glob=["*.sh"], regex=[])

    def build_parser(self, parser):
        return parser

    async def run(self, files, line_filters=None, options=options):
        result = CommandResult(0, "", "")
        if len(files) == 0:
            return result
        return await run_cmd([self.exe, "--external-sources", *files])
