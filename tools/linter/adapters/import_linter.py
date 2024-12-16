"""
Checks files to make sure there are no imports from disallowed third party
libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from enum import Enum
from typing import List, NamedTuple, Set


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

# Add the allowed third party libraries.
_MODULE_NAME_ALLOW_LIST.update(
    [
        "torch",
        "sympy",
        "torch_xla",
        "_pytest",
        "functorch",
        "the",
        "libfb",
        "typing_extensions",
        "triton",
        "numpy",
        "torchrec",
        "tabulate",
        "optree",
        "tvm",
    ]
)


def check_file(filename: str) -> List[LintMessage]:
    with open(filename) as f:
        lines = f.readlines()

    # The pattern: from/import word_that_doesn't_start_with_dot
    pattern = re.compile(r"^(?:import|from)\s+([a-zA-Z_][\w]*)")

    lint_messages = []
    for line_number, line in enumerate(lines):
        line_number += 1
        line = line.lstrip()
        match = pattern.search(line)
        if match:
            module_name = match.group(1)
            if module_name not in _MODULE_NAME_ALLOW_LIST:
                msg = LintMessage(
                    path=filename,
                    line=line_number,
                    char=None,
                    code="IMPORT",
                    severity=LintSeverity.ERROR,
                    name="Disallowed import",
                    original=None,
                    replacement=None,
                    description=f"""
importing from {module_name} is not allowed, if you believe there's a valid
reason, please add it to _MODULE_NAME_ALLOW_LIST in {CURRENT_FILE_NAME}
""",
                )
                lint_messages.append(msg)
    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    # Check all files.
    all_lint_messages = []
    for filename in args.filenames:
        lint_messages = check_file(filename)
        all_lint_messages.extend(lint_messages)

    # Print out lint messages.
    for lint_message in all_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
