import argparse


from tools.linter.lint import Linter
from tools.linter.utils import run_cmd, CommandResult


class Quickchecks(Linter):
    name = "Miscallenous lints"
    exe = __file__

    def build_parser(self, parser):
        return

    def run(self, options=None):
        result = CommandResult(0, "", "")
