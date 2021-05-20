import argparse
import inspect
import importlib
import sys

from types import ModuleType
from collections import namedtuple
from typing import List, Any, Set


CrawlError = namedtuple("CrawlError", ("reason", "path"))
CrawlResult = namedtuple("CrawlResult", ("public", "private", "errors", "module"))


def err_print(*args: Any) -> None:
    print(*args, file=sys.stderr)


system_module_names = {
    "__future__",
    "__main__",
    "_thread",
    "abc",
    "aifc",
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
    "collections.abc",
    "colorsys",
    "compileall",
    "concurrent",
    "concurrent.futures",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "cProfile",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "curses.ascii",
    "curses.panel",
    "curses.textpad",
    "dataclasses",
    "datetime",
    "dbm",
    "dbm.dumb",
    # "dbm.gnu",
    # "dbm.ndbm",
    "decimal",
    "difflib",
    "dis",
    "distutils",
    "distutils.archive_util",
    "distutils.bcppcompiler",
    "distutils.ccompiler",
    "distutils.cmd",
    "distutils.command",
    "distutils.command.bdist",
    "distutils.command.bdist_dumb",
    # "distutils.command.bdist_msi",
    # "distutils.command.bdist_packager",
    "distutils.command.bdist_rpm",
    "distutils.command.bdist_wininst",
    "distutils.command.build",
    "distutils.command.build_clib",
    "distutils.command.build_ext",
    "distutils.command.build_py",
    "distutils.command.build_scripts",
    "distutils.command.check",
    "distutils.command.clean",
    "distutils.command.config",
    "distutils.command.install",
    "distutils.command.install_data",
    "distutils.command.install_headers",
    "distutils.command.install_lib",
    "distutils.command.install_scripts",
    "distutils.command.register",
    "distutils.command.sdist",
    "distutils.core",
    "distutils.cygwinccompiler",
    "distutils.debug",
    "distutils.dep_util",
    "distutils.dir_util",
    "distutils.dist",
    "distutils.errors",
    "distutils.extension",
    "distutils.fancy_getopt",
    "distutils.file_util",
    "distutils.filelist",
    "distutils.log",
    "distutils.msvccompiler",
    "distutils.spawn",
    "distutils.sysconfig",
    "distutils.text_file",
    "distutils.unixccompiler",
    "distutils.util",
    "distutils.version",
    "doctest",
    "email",
    "email.charset",
    "email.contentmanager",
    "email.encoders",
    "email.errors",
    "email.generator",
    "email.header",
    "email.headerregistry",
    "email.iterators",
    "email.message",
    "email.mime",
    "email.parser",
    "email.policy",
    "email.utils",
    "encodings",
    "encodings.idna",
    # "encodings.mbcs",
    "encodings.utf_8_sig",
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
    "html.entities",
    "html.parser",
    "http",
    "http.client",
    "http.cookiejar",
    "http.cookies",
    "http.server",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "importlib.abc",
    "importlib.machinery",
    "importlib.metadata",
    "importlib.resources",
    "importlib.util",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "json.tool",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "logging.config",
    "logging.handlers",
    "lzma",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    # "msilib",
    # "msvcrt",
    "multiprocessing",
    "multiprocessing.connection",
    "multiprocessing.dummy",
    "multiprocessing.managers",
    "multiprocessing.pool",
    "multiprocessing.shared_memory",
    "multiprocessing.sharedctypes",
    "netrc",
    # "nis",
    "nntplib",
    "numbers",
    "operator",
    "optparse",
    "os",
    "os.path",
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
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
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
    "test.support",
    "test.support.bytecode_helper",
    "test.support.script_helper",
    "test.support.socket_helper",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "tkinter.colorchooser",
    "tkinter.commondialog",
    "tkinter.dnd",
    "tkinter.filedialog",
    "tkinter.font",
    "tkinter.messagebox",
    "tkinter.scrolledtext",
    "tkinter.simpledialog",
    "tkinter.tix",
    "tkinter.ttk",
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
    "unittest.mock",
    "urllib",
    "urllib.error",
    "urllib.parse",
    "urllib.request",
    "urllib.response",
    "urllib.robotparser",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    # "winreg",
    # "winsound",
    "wsgiref",
    "wsgiref.handlers",
    "wsgiref.headers",
    "wsgiref.simple_server",
    "wsgiref.util",
    "wsgiref.validate",
    "xdrlib",
    "xml",
    "xml.dom",
    "xml.dom.minidom",
    "xml.dom.pulldom",
    "xml.etree.ElementTree",
    "xml.parsers.expat",
    "xml.parsers.expat.errors",
    "xml.parsers.expat.model",
    "xml.sax",
    "xml.sax.handler",
    "xml.sax.saxutils",
    "xml.sax.xmlreader",
    "xmlrpc",
    "xmlrpc.client",
    "xmlrpc.server",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "zoneinfo",
    "astunparse",
}

system_modules = [importlib.import_module(name) for name in system_module_names]

def should_skip(obj: Any) -> bool:
    if not isinstance(obj, ModuleType):
        obj = getattr(obj, "__module__", None)
    
        if obj is None:
            return False

        return obj in system_module_names

    for module in system_modules:
        if obj == module:
            return True

    return False


def get_name(path: List[str], name: str) -> str:
    return ".".join(path + [name])


def add_item(path: List[str], name: str, out: CrawlResult) -> None:
    is_private = False
    for item in path:
        if item.startswith("_"):
            is_private = True
            break
    if name.startswith("_"):
        is_private = True

    if is_private:
        out.private.append(get_name(path, name))
    else:
        out.public.append(get_name(path, name))


def get_module_attributes(module: ModuleType) -> List[str]:
    attrs = []

    try:
        attrs += dir(module)
    except Exception as e:
        err_print(e)
        pass

    try:
        attrs += getattr(module, "__all__", [])
    except Exception as e:
        err_print(e)
        pass

    return attrs


def crawl_helper(
    obj: Any,
    name: str,
    path: List[str],
    seen_objects: Set[int],
    out: CrawlResult,
) -> None:
    if should_skip(obj):
        return

    if id(obj) in seen_objects:
        return
    
    seen_objects.add(id(obj))

    if isinstance(obj, ModuleType):
        attrs = get_module_attributes(obj)

        attrs = sorted(list(set(attrs)))
        add_item(path, name, out)

        for attr in attrs:
            try:
                next_obj = getattr(obj, attr)
                crawl_helper(next_obj, attr, path + [name], seen_objects, out)
            except ModuleNotFoundError as e:
                err_print(e)
                err_print(f"ERROR: whacky module {get_name(path, name)}")
                out.errors.append(
                    CrawlError(
                        reason="whacky module", path=get_name(path, name)
                    )
                )
            except AttributeError as e:
                err_print(e)
                err_print(f"ERROR: unaccessible attribute {get_name(path, name)}")
                out.errors.append(
                    CrawlError(
                        reason="unaccessible attribute", path=get_name(path, name)
                    )
                )

    elif inspect.isclass(obj):
        attrs = dir(obj)

        for attr in attrs:
            add_item(path + [name], attr, out)
    else:
        add_item(path, name, out)


def crawl(module: ModuleType) -> CrawlResult:
    result = CrawlResult(
        module=module,
        public=[],
        private=[],
        errors=[],
    )
    seen_objects: Set[int] = set()
    crawl_helper(module, module.__name__, [], seen_objects, result)
    return result


def main(module: ModuleType, public: bool, private: bool, errors: bool) -> None:
    r = crawl(module)

    if public:
        for item in r.public:
            print(item)
    if private:
        for item in r.private:
            print(item)
    if errors:
        for item in r.errors:
            print(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively list all reachable objects in a Python module (library)."
        + " This is used in PyTorch releases to determine the API surface changes"
    )
    parser.add_argument("--module", help="module to crawl", required=True)

    def add_flag(name: str, default: bool, help: str) -> None:
        parser.add_argument(f"--{name}", dest=name, help=help, action="store_true")
        parser.add_argument(f"--no-{name}", dest=name, help=help, action="store_false")
        parser.set_defaults(**{name: default})

    add_flag(name="public", default=True, help="list public APIs")
    add_flag(
        name="private",
        default=False,
        help="list private APIs (those that start with a _)",
    )
    add_flag(name="errors", default=False, help="show errors (unreachable APIs)")

    args = parser.parse_args()

    main(
        module=importlib.import_module(args.module),
        public=args.public,
        private=args.private,
        errors=args.errors,
    )
