import argparse
import shutil

from tools.linter.utils import CommandResult
from tools.linter.lint import Linter

class Flake8(Linter):
    name = "flake8"
    exe = shutil.which(name)
    options = { **Linter.options, "glob": ["*.py"] }

    def build_parser(self, parser):
        return parser

    async def run(self, files, options={}):
        return await run_cmd([
            self.exe,
            "--config",
            ".flake8",
            *files,
        ])
