import argparse
import shutil

from tools.linter.utils import CommandResult, run_cmd
from tools.linter.utils.linter import Linter


class Flake8(Linter):
    name = "flake8"
    exe = shutil.which(name)
    options = argparse.Namespace(paths=["."], glob=["*.py"], regex=[])

    # def build_parser(self, parser):
    #     return parser

    async def run(self, files, line_filters=None, options=options):
        return await run_cmd([self.exe, "--config", ".flake8", *files])
