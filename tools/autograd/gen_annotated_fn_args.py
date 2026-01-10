"""
For procedural tests needed for __torch_function__, we use this function
to export method names and signatures as needed by the tests in
test/test_overrides.py.

python -m tools.autograd.gen_annotated_fn_args \
       aten/src/ATen/native/native_functions.yaml \
       aten/src/ATen/native/tags.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/testing/_internal/generated
"""

from __future__ import annotations

import argparse
import os
import textwrap
from collections import defaultdict
from typing import Any, TYPE_CHECKING

import torchgen.api.python as python
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.utils import FileManager
from .gen_python_functions import (
    is_py_fft_function,
    is_py_linalg_function,
    is_py_nn_function,
    is_py_special_function,
    is_py_torch_function,
    is_py_variable_method,
    should_generate_py_binding,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchgen.model import Argument, BaseOperatorName, NativeFunction


def gen_annotated(
    native_yaml_path: str, tags_yaml_path: str, out: str, autograd_dir: str
) -> None:
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    mappings = (
        (is_py_torch_function, "torch._C._VariableFunctions"),
        (is_py_nn_function, "torch._C._nn"),
        (is_py_linalg_function, "torch._C._linalg"),
        (is_py_special_function, "torch._C._special"),
        (is_py_fft_function, "torch._C._fft"),
        (is_py_variable_method, "torch.Tensor"),
    )
    annotated_args: list[str] = []
    for pred, namespace in mappings:
        groups: dict[BaseOperatorName, list[NativeFunction]] = defaultdict(list)
        for f in native_functions:
            if not should_generate_py_binding(f) or not pred(f):
                continue
            groups[f.func.name.name].append(f)
        for group in groups.values():
            for f in group:
                annotated_args.append(f"{namespace}.{gen_annotated_args(f)}")

    template_path = os.path.join(autograd_dir, "templates")
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template(
        "annotated_fn_args.py",
        "annotated_fn_args.py.in",
        lambda: {
            "annotated_args": textwrap.indent("\n".join(annotated_args), "    "),
        },
    )


@with_native_function
def gen_annotated_args(f: NativeFunction) -> str:
    def _get_kwargs_func_exclusion_list() -> list[str]:
        # functions that currently don't work with kwargs in test_overrides.py
        return [
            "diagonal",
            "round_",
            "round",
            "scatter_",
        ]

    def _add_out_arg(
        out_args: list[dict[str, Any]], args: Sequence[Argument], *, is_kwarg_only: bool
    ) -> None:
        for arg in args:
            if arg.default is not None:
                continue
            out_arg: dict[str, Any] = {}
            out_arg["is_kwarg_only"] = str(is_kwarg_only)
            out_arg["name"] = arg.name
            out_arg["simple_type"] = python.argument_type_str(
                arg.type, simple_type=True
            )
            size_t = python.argument_type_size(arg.type)
            if size_t:
                out_arg["size"] = size_t
            out_args.append(out_arg)

    out_args: list[dict[str, Any]] = []
    _add_out_arg(out_args, f.func.arguments.flat_positional, is_kwarg_only=False)
    if f"{f.func.name.name}" not in _get_kwargs_func_exclusion_list():
        _add_out_arg(out_args, f.func.arguments.flat_kwarg_only, is_kwarg_only=True)

    return f"{f.func.name.name}: {repr(out_args)},"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate annotated_fn_args script")
    parser.add_argument(
        "native_functions", metavar="NATIVE", help="path to native_functions.yaml"
    )
    parser.add_argument("tags", metavar="TAGS", help="path to tags.yaml")
    parser.add_argument("out", metavar="OUT", help="path to output directory")
    parser.add_argument(
        "autograd", metavar="AUTOGRAD", help="path to template directory"
    )
    args = parser.parse_args()
    gen_annotated(args.native_functions, args.tags, args.out, args.autograd)


if __name__ == "__main__":
    main()
