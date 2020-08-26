"""
For procedural tests needed for __torch_function__, we use this function
to export method names and signatures as needed by the tests in
test/test_overrides.py. 

python -m tools.autograd.gen_autograd \
       build/aten/src/ATen/Declarations.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/testing/_internal/generated
"""

from .utils import write, CodeTemplate
from .gen_python_functions import (
    get_py_nn_functions,
    get_py_torch_functions,
    get_py_variable_methods,
    op_name,
)
import textwrap
from .gen_autograd import load_aten_declarations


def gen_annotated(aten_path, out, template_path):
    declarations = load_aten_declarations(aten_path)
    annotated_args = []
    for func in recurse_dict(get_py_torch_functions(declarations)):
        annotated_args.append(process_func("torch._C._VariableFunctions", func))

    for func in recurse_dict(get_py_nn_functions(declarations)):
        annotated_args.append(process_func("torch._C._nn", func))

    for func in recurse_dict(get_py_variable_methods(declarations)):
        annotated_args.append(process_func("torch.Tensor", func))

    annotated_args = textwrap.indent("\n".join(annotated_args), "    ")
    env = {"annotated_args": annotated_args}
    PY_ANNOTATED_ARGS = CodeTemplate.from_file(template_path + '/templates/annotated_fn_args.py')
    write(out, 'annotated_fn_args.py', PY_ANNOTATED_ARGS, env)


def process_func(namespace, func):
    args = func["arguments"]
    out_args = []
    for arg in args:
        if 'default' in arg or arg.get('kwarg_only', False) or arg.get('output', False):
            continue
        out_args.append({k: arg[k] for k in ('name', 'simple_type', 'size') if k in arg})

    return f"{namespace}.{op_name(func)}: {out_args!r},"


def recurse_dict(d):
    for e in d.values():
        for i in e:
            yield i


def main():
    parser = argparse.ArgumentParser(
        description='Generate annotated_fn_args script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('autograd', metavar='AUTOGRAD',
                        help='path to template directory')
    args = parser.parse_args()
    gen_annotated(args.declarations, args.out, args.autograd)


if __name__ == '__main__':
    main()
