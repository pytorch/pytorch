import ast
import os
import functools
import inspect
import sys
import tempfile
from typing import Any, List, Optional, Tuple
from textwrap import dedent
from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory


# this arbitrary-looking assortment of functionality is provided here
# to have a central place for overrideable behavior. The motivating
# use is the FB build environment, where this source file is replaced
# by an equivalent.

if sys.executable == 'torch_deploy':
    # __file__ is meaningless in the context of frozen torch used in torch deploy.
    # setting empty torch_parent should allow below functions to operate without crashing,
    # but it's unclear if there is a valid use case for them in the context of deploy.
    torch_parent = ""
else:
    if os.path.basename(os.path.dirname(__file__)) == 'shared':
        torch_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    else:
        torch_parent = os.path.dirname(os.path.dirname(__file__))

def get_file_path(*path_components: str) -> str:
    return os.path.join(torch_parent, *path_components)


def get_file_path_2(*path_components: str) -> str:
    return os.path.join(*path_components)


def get_writable_path(path: str) -> str:
    if os.access(path, os.W_OK):
        return path
    return tempfile.mkdtemp(suffix=os.path.basename(path))



def prepare_multiprocessing_environment(path: str) -> None:
    pass


def resolve_library_path(path: str) -> str:
    return os.path.realpath(path)


def get_source_lines_and_file(
    obj: Any,
    error_msg: Optional[str] = None,
) -> Tuple[List[str], int, Optional[str]]:
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    """
    filename = None  # in case getsourcefile throws
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        msg = (f"Can't get source for {obj}. TorchScript requires source access in "
               "order to carry out compilation, make sure original .py files are "
               "available.")
        if error_msg:
            msg += '\n' + error_msg
        raise OSError(msg) from e

    return sourcelines, file_lineno, filename


def normalize_source_lines(sourcelines: List[str]) -> List[str]:
    """
    This helper function accepts a list of source lines. It finds the
    indentation level of the function definition (`def`), then it indents
    all lines in the function body to a point at or greater than that
    level. This allows for comments and continued string literals that
    are at a lower indentation than the rest of the code.
    Args:
        sourcelines: function source code, separated into lines by
                        the '\n' character
    Returns:
        A list of source lines that have been correctly aligned
    """

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]

    # Find the line and line number containing the function definition
    for i, l in enumerate(sourcelines):
        if l.lstrip().startswith("def"):
            idx = i
            break
    fn_def = sourcelines[idx]

    # Get a string representing the amount of leading whitespace
    whitespace = fn_def.split("def")[0]

    # Add this leading whitespace to all lines before and after the `def`
    aligned_prefix = [whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]]
    aligned_suffix = [whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1:]]

    # Put it together again
    aligned_prefix.append(fn_def)
    return aligned_prefix + aligned_suffix


# Thin wrapper around SourceRangeFactory to store extra metadata
# about the function-to-be-compiled.
class SourceContext(SourceRangeFactory):
    def __init__(self, source, filename, file_lineno, leading_whitespace_len, uses_true_division=True):
        super(SourceContext, self).__init__(source, filename, file_lineno, leading_whitespace_len)
        self.uses_true_division = uses_true_division
        self.filename = filename


@functools.lru_cache(maxsize=None)
def make_source_context(*args):
    return SourceContext(*args)


def fake_range():
    return SourceContext('', None, 0, 0).make_raw_range(0, 1)


class ParsedDef:
    def __init__(self, ast, ctx, source):
        self.ast = ast
        self.ctx = ctx
        self.source = source


def parse_def(fn):
    sourcelines, file_lineno, filename = get_source_lines_and_file(fn, ErrorReport.call_stack())
    sourcelines = normalize_source_lines(sourcelines)
    source = ''.join(sourcelines)
    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError(f"Expected a single top-level function: {filename}:{file_lineno}")
    leading_whitespace_len = len(source.split('\n', 1)[0]) - len(dedent_src.split('\n', 1)[0])
    ctx = make_source_context(source, filename, file_lineno, leading_whitespace_len, True)
    return ParsedDef(py_ast, ctx, source)


TEST_MASTER_ADDR = '127.0.0.1'
TEST_MASTER_PORT = 29500
# USE_GLOBAL_DEPS controls whether __init__.py tries to load
# libtorch_global_deps, see Note [Global dependencies]
USE_GLOBAL_DEPS = True
# USE_RTLD_GLOBAL_WITH_LIBTORCH controls whether __init__.py tries to load
# _C.so with RTLD_GLOBAL during the call to dlopen.
USE_RTLD_GLOBAL_WITH_LIBTORCH = False
