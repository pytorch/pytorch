from __future__ import annotations

# Generate a set of type tests for mypy or pyrefly

import collections
import pprint
from pathlib import Path
import token
import tokenize
import dataclasses
import sys

import argparse
from torchgen.gen import parse_native_yaml
from torchgen.model import Argument, OptionalType
from torchgen import model
from typing import Iterator, Optional, Union, NamedTuple, ClassVar


INDENT = "  "

# TODO: handle "Tensor?"

PYTORCH_ROOT = Path(__file__).absolute().parents[1]
NATIVE_PATH = PYTORCH_ROOT / "aten/src/ATen/native"
NATIVE_YAML_PATH = NATIVE_PATH / "native_functions.yaml"
TAGS_YAML_PATH = NATIVE_PATH / "tags.yaml"


class TypeTestCase(NamedTuple):
    method: str
    args: tuple[str, ...]
    result_type: str


def to_code(t: TestTypeCase) -> str:
    args = ",\n    ".join(t.args)
    return f"""\
    assert_type(
        {t.method}(
            {args},
        ),
        {t.result_type}
    )"""


def test_case(nf: model.NativeFunction) -> Iterator[TypeTestCase]:
    name = nf.func.name.name.base
    method_name = name + (nf.kind == SchemaKind.inplace) * "_"
    obj = "torch" if nf.kind == SchemaKind.functional else "TENSOR"
    args = ""

    yield TestTypeCase(
        method=f"{obj}.{method_name}",
        args=args,
        result_type=result_type,
    )


def lines_for_nf(name: str, nfs: list[model.NativeFunction]) -> list[str]:
    lines = [f"def test_{name}() -> None:"]

    for nf in nfs:
        method_name = name + (nf.kind == SchemaKind.inplace) * "_"
        obj = "torch" if nf.kind == SchemaKind.functional else "TENSOR"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mypy type tests")
    parser.add_argument("-o", "--output-dir", "--output_dir", help="output directory")
    parser.add_argument(
        "--dry-run", "--dry_run", type=bool, default=False, help="output directory"
    )
    options = parser.parse_args()

    if True:
        by_base_name = funcs_by_base_name()
        for k, v in by_base_name.items():
            for i in v:
                print(*parts(i), sep="")
                print()
    else:
        s = collections.Counter()
        args = args_by_base_name()
        s.update(j for a in args.values() for i in a.values() for j in i)
        print(*sorted(str(i) for i in s), '', sep='\n')
        print(*(f"{k}: {v}" for v, k in sorted((v, str(k)) for k, v in s.items())), sep="\n")
        pprint.pprint(args)


def funcs_by_base_name():
    funcs_by_base_name = {}
    for nf in native_functions():
        if accept(nf):
            funcs_by_base_name.setdefault(nf.func.name.name.base, []).append(nf)
    return funcs_by_base_name


def accept(nf):
    base_name =
    name = nf.func.name.name.base
    if name.startswith("_") or 'generated' in nf.tags:
        return False
    unknown_base = "dunder_method", "functional_override", "namespace"
    if any(getattr(nf.func.name.name, a) for a in unknown_base):
        return False
    unknown_args = "pre_self_positional",
    if any(getattr(nf.func.arguments, a) for a in unknown_args):
        return False
    if False:
        return name == 'randint'
    if False:
        return name in ('set_', 'set')
    return True


def get_args(nfs) -> dict[str, list[Any]]:
    all_args = [nf.func.arguments.flat_all for nf in nfs]
    all_names = [k for v in all_args for k in v]
    name_to_types = {a.name: set() for a in all_names}

    for args in all_args:
        param_types = {a.name: a.type for a in args}
        for name, types in name_to_types.items():
            type_ = param_types.get(name)
            # print(f"{name=}, {types=}, {type_=}", file=sys.stderr)
            if isinstance(type_, OptionalType):
                types.add(None)
                types.add(type_.elem)
            else:
                types.add(type_)

    return name_to_types


def native_functions():
    return parse_native_yaml(NATIVE_YAML_PATH, TAGS_YAML_PATH).native_functions


def args_by_base_name():
    return {k: get_args(v) for k, v in funcs_by_base_name().items()}


def parts(dc):
    indent = '\n'
    for t in tokenize.generate_tokens(iter(str(dc).splitlines()).__next__):
        if t.type == token.OP and t.string in ")]}":
            yield (indent := indent[:-len(INDENT)])

        yield t.string

        if t.type == token.OP:
            if t.string in "([{":
                yield (indent := indent + INDENT)
            elif t.string == ",":
                yield indent


class Type(NamedTuple):
    type_name: str
    variable_name: str
    primary: str = ""
    secondary: str = ""


_PARAMS = (
    # Type("Storage", "STORAGE", "Storage()"),
    # Type("Stream", "STREAM", "Stream()"),

    Type("Device", "DEVICE", "torch.device(type='cpu')", "torch.device(type='cuda')"),
    Type("DeviceIndex", "DEVICE_INDEX", "0", "-1"),
    Type("Dimname", "DIMNAME", "2", "1"),
    Type("Generator", "GENERATOR", "(i for i in range(3))"),
    Type("Layout", "LAYOUT", "torch.strided", "torch.sparse_coo"),
    Type("MemoryFormat", "MEMORY_FORMAT", "torch.contiguous_format", "torch.preserve_format"),
    Type("None", "None"),
    Type("Scalar", "SCALAR", "torch.tensor(7.125)", "torch.tensor(-11.0)"),
    Type("ScalarType", "DTYPE", "torch.int64"),
    Type("SymInt", "SYMINT", "9", "18"),
    Type("Tensor", "TENSOR", "torch.tensor((5, 7))", "torch.tensor((6, 8))"),
    Type("Tensor?", "TENSORQ", "torch.tensor((12, 18))", "None"),
    Type("bool", "BOOL", "True", "False"),
    Type("float", "FLOAT", "2.3", "1.0"),
    Type("int", "INT", "8", "19"),
    Type("str", "STR", "\"the string\"", "\"another\""),
)
PARAMETERS = {p.type_name: p for p in _PARAMS}


if __name__ == "__main__":
    main()
