import atexit
import re
import shutil
import tempfile
import textwrap
from typing import Iterator, List, Optional, Tuple
import uuid

from core.api import AutogradMode, AutoLabels, RuntimeMode, TimerArgs, GroupedBenchmark
from core.jit import generate_torchscript_file
from core.types import Definition, FlatDefinition, FlatIntermediateDefinition, Label


_TEMPDIR: Optional[str] = None
def get_temp_dir() -> str:
    global _TEMPDIR
    if _TEMPDIR is None:
        temp_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, path=temp_dir)
        _TEMPDIR = temp_dir
    return _TEMPDIR


def _flatten(
    key_prefix: Label,
    sub_schema: Definition,
    result: FlatIntermediateDefinition
) -> None:
    for k, value in sub_schema.items():
        if isinstance(k, tuple):
            assert all(isinstance(ki, str) for ki in k)
            key_suffix: Label = k
        elif k is None:
            key_suffix = ()
        else:
            assert isinstance(k, str)
            key_suffix = (k,)

        key: Label = key_prefix + key_suffix
        if isinstance(value, (TimerArgs, GroupedBenchmark)):
            assert key not in result, f"duplicate key: {key}"
            result[key] = value
        else:
            assert isinstance(value, dict)
            _flatten(key_prefix=key, sub_schema=value, result=result)


def flatten(schema: Definition) -> FlatIntermediateDefinition:
    result: FlatIntermediateDefinition = {}
    _flatten(key_prefix=(), sub_schema=schema, result=result)

    # Ensure that we produced a valid flat definition.
    for k, v in result.items():
        assert isinstance(k, tuple)
        assert all(isinstance(ki, str) for ki in k)
        assert isinstance(v, (TimerArgs, GroupedBenchmark))
    return result


def unpack(definitions: FlatIntermediateDefinition) -> FlatDefinition:
    results: List[Tuple[Label, AutoLabels, TimerArgs]] = []

    for label, args in definitions.items():
        if isinstance(args, TimerArgs):
            auto_labels = AutoLabels(
                RuntimeMode.EXPLICIT,
                AutogradMode.EXPLICIT,
                args.language
            )
            results.append((label, auto_labels, args))

        else:
            assert isinstance(args, GroupedBenchmark)

            model_path: Optional[str] = None
            ts_model_setup = args.ts_model_setup
            if ts_model_setup is not None:
                name: str = re.sub(r'[^a-z0-9_]', '_', '_'.join(label).lower())
                name = f"{name}_{uuid.uuid4()}"
                model_path = generate_torchscript_file(ts_model_setup, name=name, temp_dir=get_temp_dir())

            for auto_labels, timer_args in args.flatten(model_path):
                results.append((label, auto_labels, timer_args))

    return tuple(results)


def parse_stmts(stmts: str) -> Tuple[str, str]:
    """Parser for side-by-side Python and C++ stmts.

    For more complex statements, it can be useful to see Python and C++ code
    side by side. To this end, we provide an **extremely restricted** way
    to define Python and C++ code side-by-side. The schema should be mostly
    self explanatory, with the following non-obvious caveats:
      - Width for the left (Python) column MUST be 40 characters.
      - The column separator is " | ", not "|". Whitespace matters.
    """
    stmts = textwrap.dedent(stmts).strip()
    lines: List[str] = stmts.splitlines(keepends=False)
    assert len(lines) >= 3, f"Invalid string:\n{stmts}"

    column_header_pattern = r"^Python\s{35}\| C\+\+(\s*)$"
    signature_pattern = r"^: f\((.*)\)( -> (.+))?\s*$"
    separation_pattern = r"^[-]{40} | [-]{40}$"
    code_pattern = r"^(.{40}) \|($| (.*)$)"

    column_match = re.search(column_header_pattern, lines[0])
    if column_match is None:
        raise ValueError(
            f"Column header `{lines[0]}` "
            f"does not match pattern `{column_header_pattern}`")

    assert re.search(separation_pattern, lines[1])

    py_lines: List[str] = []
    cpp_lines: List[str] = []
    for l in lines[2:]:
        l_match = re.search(code_pattern, l)
        if l_match is None:
            raise ValueError(f"Invalid line `{l}`")
        py_lines.append(l_match.groups()[0])
        cpp_lines.append(l_match.groups()[2] or "")

        # Make sure we can round trip for correctness.
        l_from_stmts = f"{py_lines[-1]:<40} | {cpp_lines[-1]:<40}".rstrip()
        assert l_from_stmts == l.rstrip(), f"Failed to round trip `{l}`"

    return "\n".join(py_lines), "\n".join(cpp_lines)


def iter_parsed_lines(stmts: str) -> Iterator[Tuple[str, str]]:
    py_stmt, cpp_stmt = parse_stmts(stmts)
    py_lines = [l.rstrip() for l in py_stmt.splitlines(keepends=False)]
    cpp_lines = [l.rstrip() for l in cpp_stmt.splitlines(keepends=False)]
    assert len(py_lines) == len(cpp_lines)
    return zip(py_lines, cpp_lines)
