"""
For procedural tests needed for __torch_function__, we use this function
to export method names and signatures as needed by the tests in
test/test_overrides.py.

python -m tools.autograd.gen_annotated_fn_args \
       aten/src/ATen/native/native_functions.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/testing/_internal/generated
"""

from collections import defaultdict
import argparse
import os
import textwrap

from typing import Dict, List, Any

from tools.codegen.gen import parse_native_yaml, FileManager
from tools.codegen.context import with_native_function
from tools.codegen.model import *
import tools.codegen.api.python as python
from .gen_python_functions import should_generate_py_binding, is_py_torch_function, is_py_nn_function, is_py_variable_method

def gen_annotated(native_yaml_path: str, out: str, autograd_dir: str) -> None:
    native_functions = parse_native_yaml(native_yaml_path)
    mappings = (
        (is_py_torch_function, 'torch._C._VariableFunctions'),
        (is_py_nn_function, 'torch._C._nn'),
        (is_py_variable_method, 'torch.Tensor'),
    )
    annotated_args: List[str] = []
    for pred, namespace in mappings:
        groups: Dict[BaseOperatorName, List[NativeFunction]] = defaultdict(list)
        for f in native_functions:
            if not should_generate_py_binding(f) or not pred(f):
                continue
            groups[f.func.name.name].append(f)
        for group in groups.values():
            for f in group:
                annotated_args.append(f'{namespace}.{gen_annotated_args(f)}')

    template_path = os.path.join(autograd_dir, 'templates')
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template('annotated_fn_args.py', 'annotated_fn_args.py', lambda: {
        'annotated_args': textwrap.indent('\n'.join(annotated_args), '    '),
    })

@with_native_function
def gen_annotated_args(f: NativeFunction) -> str:
    out_args: List[Dict[str, Any]] = []
    for arg in f.func.arguments.flat_positional:
        if arg.default is not None:
            continue
        out_arg: Dict[str, Any] = {}
        out_arg['name'] = arg.name
        out_arg['simple_type'] = python.argument_type_str(arg.type, simple_type=True)
        size = python.argument_type_size(arg.type)
        if size:
            out_arg['size'] = size
        out_args.append(out_arg)

    return f'{f.func.name.name}: {repr(out_args)},'

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate annotated_fn_args script')
    parser.add_argument('native_functions', metavar='NATIVE',
                        help='path to native_functions.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('autograd', metavar='AUTOGRAD',
                        help='path to template directory')
    args = parser.parse_args()
    gen_annotated(args.native_functions, args.out, args.autograd)

if __name__ == '__main__':
    main()
