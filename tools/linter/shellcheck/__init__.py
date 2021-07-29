import argparse
import shutil


from tools.linter.utils import run_cmd, glob2regex
from tools.linter.lint import Linter


class Shellcheck(Linter):
    name = "shellcheck"
    exe = shutil.which(name)
    options = { **Linter.options, "glob": ["*.sh"] }

    def build_parser(self, parser):
        return parser

    async def run(self, files, options=None):
        return await run_cmd([
            self.exe,
            "--external-sources",
            *files
        ])
