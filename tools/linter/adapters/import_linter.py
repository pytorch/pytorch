"""
Checks files to make sure there are no imports from disallowed third party
libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import token
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Set, TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


LINTER_CODE = "NEWLINE"
CURRENT_FILE_NAME = os.path.basename(__file__)
_MODULE_NAME_ALLOW_LIST: Set[str] = set()

# Add builtin modules.
if sys.version_info >= (3, 10):
    _MODULE_NAME_ALLOW_LIST.update(sys.stdlib_module_names)
else:
    assert (sys.version_info.major, sys.version_info.minor) == (3, 9)
    # Taken from `stdlib_list("3.9")` to avoid introducing a new dependency.
    _MODULE_NAME_ALLOW_LIST.update(
        [
            "__future__",
            "_abc",
            "_aix_support",
            "_ast",
            "_bootlocale",
            "_bootsubprocess",
            "_codecs",
            "_collections",
            "_collections_abc",
            "_compat_pickle",
            "_compression",
            "_crypt",
            "_functools",
            "_hashlib",
            "_imp",
            "_io",
            "_locale",
            "_lsprof",
            "_markupbase",
            "_operator",
            "_osx_support",
            "_peg_parser",
            "_posixsubprocess",
            "_py_abc",
            "_pydecimal",
            "_pyio",
            "_random",
            "_signal",
            "_sitebuiltins",
            "_socket",
            "_sre",
            "_ssl",
            "_stat",
            "_string",
            "_strptime",
            "_symtable",
            "_sysconfigdata_x86_64_conda_cos6_linux_gnu",
            "_sysconfigdata_x86_64_conda_linux_gnu",
            "_thread",
            "_threading_local",
            "_tracemalloc",
            "_uuid",
            "_warnings",
            "_weakref",
            "_weakrefset",
            "abc",
            "aifc",
            "antigravity",
            "argparse",
            "array",
            "ast",
            "asynchat",
            "asyncio",
            "asyncore",
            "atexit",
            "audioop",
            "base64",
            "bdb",
            "binascii",
            "binhex",
            "bisect",
            "builtins",
            "bz2",
            "cProfile",
            "calendar",
            "cgi",
            "cgitb",
            "chunk",
            "cmath",
            "cmd",
            "code",
            "codecs",
            "codeop",
            "collections",
            "colorsys",
            "compileall",
            "concurrent",
            "configparser",
            "contextlib",
            "contextvars",
            "copy",
            "copyreg",
            "crypt",
            "csv",
            "ctypes",
            "curses",
            "dataclasses",
            "datetime",
            "dbm",
            "decimal",
            "difflib",
            "dis",
            "distutils",
            "doctest",
            "email",
            "encodings",
            "ensurepip",
            "enum",
            "errno",
            "faulthandler",
            "fcntl",
            "filecmp",
            "fileinput",
            "fnmatch",
            "formatter",
            "fractions",
            "ftplib",
            "functools",
            "gc",
            "genericpath",
            "getopt",
            "getpass",
            "gettext",
            "glob",
            "graphlib",
            "grp",
            "gzip",
            "hashlib",
            "heapq",
            "hmac",
            "html",
            "http",
            "idlelib",
            "imaplib",
            "imghdr",
            "imp",
            "importlib",
            "inspect",
            "io",
            "ipaddress",
            "itertools",
            "json",
            "keyword",
            "lib2to3",
            "linecache",
            "locale",
            "logging",
            "lzma",
            "mailbox",
            "mailcap",
            "marshal",
            "math",
            "mimetypes",
            "mmap",
            "modulefinder",
            "msilib",
            "msvcrt",
            "multiprocessing",
            "netrc",
            "nis",
            "nntplib",
            "ntpath",
            "nturl2path",
            "numbers",
            "opcode",
            "operator",
            "optparse",
            "os",
            "ossaudiodev",
            "parser",
            "pathlib",
            "pdb",
            "pickle",
            "pickletools",
            "pipes",
            "pkgutil",
            "platform",
            "plistlib",
            "poplib",
            "posix",
            "posixpath",
            "pprint",
            "profile",
            "pstats",
            "pty",
            "pwd",
            "py_compile",
            "pyclbr",
            "pydoc",
            "pydoc_data",
            "queue",
            "quopri",
            "random",
            "re",
            "readline",
            "reprlib",
            "resource",
            "rlcompleter",
            "runpy",
            "sched",
            "secrets",
            "select",
            "selectors",
            "shelve",
            "shlex",
            "shutil",
            "signal",
            "site",
            "smtpd",
            "smtplib",
            "sndhdr",
            "socket",
            "socketserver",
            "spwd",
            "sqlite3",
            "sre_compile",
            "sre_constants",
            "sre_parse",
            "ssl",
            "stat",
            "statistics",
            "string",
            "stringprep",
            "struct",
            "subprocess",
            "sunau",
            "symbol",
            "symtable",
            "sys",
            "sysconfig",
            "syslog",
            "tabnanny",
            "tarfile",
            "telnetlib",
            "tempfile",
            "termios",
            "test",
            "textwrap",
            "this",
            "threading",
            "time",
            "timeit",
            "tkinter",
            "token",
            "tokenize",
            "trace",
            "traceback",
            "tracemalloc",
            "tty",
            "turtle",
            "turtledemo",
            "types",
            "typing",
            "unicodedata",
            "unittest",
            "urllib",
            "uu",
            "uuid",
            "venv",
            "warnings",
            "wave",
            "weakref",
            "webbrowser",
            "winreg",
            "winsound",
            "wsgiref",
            "xdrlib",
            "xml",
            "xmlrpc",
            "xxsubtype",
            "zipapp",
            "zipfile",
            "zipimport",
            "zlib",
            "zoneinfo",
        ]
    )

# Add the allowed third party libraries. Please avoid updating this unless you
# understand the risks -- see `_ERROR_MESSAGE` for why.
_MODULE_NAME_ALLOW_LIST.update(
    [
        "sympy",
        "einops",
        "libfb",
        "torch",
        "tvm",
        "_pytest",
        "tabulate",
        "optree",
        "typing_extensions",
        "triton",
        "functorch",
        "torchrec",
        "numpy",
        "torch_xla",
    ]
)

_ERROR_MESSAGE = """
Please do not import third-party modules in PyTorch unless they're explicit
requirements of PyTorch. Imports of a third-party library may have side effects
and other unintentional behavior. If you're just checking if a module exists,
use sys.modules.get("torchrec") or the like.
"""


def check_file(filepath: str) -> List[LintMessage]:
    path = Path(filepath)
    file = _linter.PythonFile("import_linter", path)
    lint_messages = []
    for line_number, line_of_tokens in enumerate(file.token_lines):
        # Skip indents
        idx = 0
        for tok in line_of_tokens:
            if tok.type == token.INDENT:
                idx += 1
            else:
                break

        # Look for either "import foo..." or "from foo..."
        if idx + 1 < len(line_of_tokens):
            tok0 = line_of_tokens[idx]
            tok1 = line_of_tokens[idx + 1]
            if tok0.type == token.NAME and tok0.string in {"import", "from"}:
                if tok1.type == token.NAME:
                    module_name = tok1.string
                    if module_name not in _MODULE_NAME_ALLOW_LIST:
                        msg = LintMessage(
                            path=filepath,
                            line=line_number,
                            char=None,
                            code="IMPORT",
                            severity=LintSeverity.ERROR,
                            name="Disallowed import",
                            original=None,
                            replacement=None,
                            description=_ERROR_MESSAGE,
                        )
                        lint_messages.append(msg)
    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filepaths",
        nargs="+",
        help="paths of files to lint",
    )
    args = parser.parse_args()

    # Check all files.
    all_lint_messages = []
    for filepath in args.filepaths:
        lint_messages = check_file(filepath)
        all_lint_messages.extend(lint_messages)

    # Print out lint messages.
    for lint_message in all_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
