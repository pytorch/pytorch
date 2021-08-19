import argparse
import os
from typing import List, Any

from . import CommandResult
from .filter_helpers import filter_files

class Linter:
    name = ""
    exe = ""
    options = argparse.Namespace()

    def __init__(self):
        exists = os.access(self.exe, os.X_OK)
        if not exists:
            raise RuntimeError(f"{self.exe} not found")

    def filter_files(self, files, glob: List[str], regex: List[str]):
        return filter_files(files, glob, regex)

    async def run(
        self, files: List[str], line_filters: List[str] = None, options: Any = options
    ) -> CommandResult:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name} using {self.exe}"

    def __repr__(self) -> str:
        return f"{self.name} using {self.exe}"

