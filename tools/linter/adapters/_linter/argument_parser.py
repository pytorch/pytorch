from __future__ import annotations

import argparse
import sys
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from typing_extensions import Never


class ArgumentParser(argparse.ArgumentParser):
    """
    Adds better help formatting and default arguments to argparse.ArgumentParser
    """

    def __init__(
        self,
        prog: str | None = None,
        usage: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        is_fixer: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(prog, usage, description, None, **kwargs)
        self._epilog = epilog

        help = "A list of files or directories to lint"
        self.add_argument("files", nargs="*", help=help)
        # TODO(rec): get fromfile_prefix_chars="@", type=argparse.FileType to work

        help = "Fix lint errors if possible" if is_fixer else argparse.SUPPRESS
        self.add_argument("-f", "--fix", action="store_true", help=help)

        help = "Run for lintrunner and print LintMessages which aren't edits"
        self.add_argument("-l", "--lintrunner", action="store_true", help=help)

        help = "Print more debug info"
        self.add_argument("-v", "--verbose", action="store_true", help=help)

    def exit(self, status: int = 0, message: str | None = None) -> Never:
        """
        Overriding this method is a workaround for argparse throwing away all
        line breaks when printing the `epilog` section of the help message.
        """
        argv = sys.argv[1:]
        if self._epilog and not status and "-h" in argv or "--help" in argv:
            print(self._epilog)
        super().exit(status, message)
