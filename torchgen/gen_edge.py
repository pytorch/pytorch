import argparse
import tempfile
from collections import defaultdict

import dataclasses
import yaml
from torchgen.utils import context, make_file_manager, FileManager

from torchgen.gen import parse_native_yaml, LineLoader
from typing import Sequence, List, Dict

from torchgen.model import NativeFunction, Location, OperatorName, FunctionSchema

type_map = {
    "fp32": "ScalarType::Float",
    "fp64": "ScalarType::Double",
    "i32": "ScalarType::Int",
    "i64": "ScalarType::Long",
}


def parse_argument_types(func: FunctionSchema, types: Dict) -> Dict[str, Sequence[str]]:
    assert isinstance(types, dict), f"types: {types} is not a dict"
    loc = types.pop("__line__")
    res = defaultdict(list)
    args = func.arguments.flat_all
    arg_map = {a.name: a for a in args}
    with context(lambda: f"in line {loc}:"):
        for arg, allowed in types.items():
            assert arg in arg_map
            allowed = [t.strip() for t in allowed.split(',')]
            res[arg].extend(type_map[t] for t in allowed)
    return res


def parse_yaml_files(aten_yaml_path: str, tags_yaml_path: str, native_yaml_path: str) -> Sequence[NativeFunction]:
    aten_parsed_yaml = parse_native_yaml(
        aten_yaml_path,
        tags_yaml_path,
        None,
        skip_native_fns_gen=False,
    )
    aten_native_functions = aten_parsed_yaml.native_functions
    aten_function_map = {f.func.name: f for f in aten_native_functions}
    native_functions: List[NativeFunction] = []
    with open(native_yaml_path, "r") as f:
        es = yaml.load(f, Loader=LineLoader)
        for e in es:
            assert isinstance(e.get("__line__"), int), e
            loc = Location(native_yaml_path, e["__line__"])
            funcs = e.get("func")
            with context(lambda: f"in {loc}:\n  {funcs}"):
                inherits = e.get("inherits").split("::")[1]
                op_name = OperatorName.parse(inherits)
                assert op_name in aten_function_map, f"{op_name} is not an ATen op"
                aten_f = aten_function_map[op_name]
                allowed_types = e.get("type_constraints")
                types = parse_argument_types(aten_f.func, allowed_types)
                f = dataclasses.replace(aten_f, namespace="edge", type_constraints=types)
                native_functions.append(f)
    return native_functions


def gen_sources(native_functions: Sequence[NativeFunction], fm: FileManager) -> None:
    comma = ", "

    def gen_operator_registration(f: NativeFunction) -> str:
        types = []
        separator = ",\n\t\t\t\t"
        if f.type_constraints:
            print(f.type_constraints)
            for k in f.type_constraints:
                types.append(f"""{{"{k}", {{ {comma.join(t for t in f.type_constraints[k])} }} }}""")
        return f"""
        Operator(
            "edge::{f.func}",
            [](Stack &stack) {{
                c10::OperatorName name = c10::OperatorName("aten::add", "Tensor");
                std::shared_ptr<Operator> op = findOperatorFor(name);
                if (op) {{
                    op->getOperation()(stack);
                }}
            }},
            aliasAnalysisFromSchema(),
            {{ {separator.join(types)} }}
        )"""

    fm.write("RegisterEdgeOps.cpp", lambda: {
        "operators": comma.join(gen_operator_registration(f) for f in native_functions)
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate operator source files")
    # Although we don't refer to --source-path directly, make_file_manager()
    # expects it to point to a directory that contains a templates/ subdirectory
    # containing the file templates.
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for kernel templates",
    )
    parser.add_argument(
        "--edge_ops_yaml_path",
        help="path to the edge.yaml file to use.",
    )
    parser.add_argument(
        "--aten_yaml_path",
        help="path to native_functions.yaml file.",
    )
    # Note that make_file_manager() also looks at --install-dir.
    parser.add_argument(
        "-d", "--install_dir", help="output directory", default="build/generated"
    )
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    # Although we don't refer to --dry-run directly, make_file_manager() looks
    # for it.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    parser.add_argument(
        "--tags-path",
        help="Path to tags.yaml. Required by yaml parsing in codegen system.",
    )
    parser.add_argument(
        "--generate",
        type=str,
        nargs="*",
        choices=["headers", "sources"],
        default=["headers", "sources"],
        help="Generate only a subset of files",
    )
    options = parser.parse_args()
    assert options.tags_path, "tags.yaml is required by codegen yaml parsing."

    native_functions = parse_yaml_files(
        aten_yaml_path=options.aten_yaml_path,
        tags_yaml_path=options.tags_path,
        native_yaml_path=options.edge_ops_yaml_path,
    )
    cpu_fm = make_file_manager(options=options)

    gen_sources(native_functions, cpu_fm)


if __name__ == "__main__":
    main()
