import argparse


from tools.linter.lint import Linter


class Misc(Linter):
    name = "miscellaneous lints"
    exe = __file__

    def parse_args(self, args):
        parser = argparse.ArgumentParser(f"Run {self.name} on PyTorch")
        return parser.parse_args(args)

    def run(self, options=None):
        print("ran misc")
