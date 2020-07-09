from .utils import write, CodeTemplate
from .gen_python_functions import get_py_nn_functions, get_py_torch_functions, op_name
import textwrap
from .gen_autograd import load_aten_declarations


def gen_annotated(aten_path, out, template_path):
    declarations = load_aten_declarations(aten_path)
    annotated_args = []
    for func in recurse_dict(get_py_torch_functions(declarations)):
        annotated_args.append(process_func("torch._C._VariableFunctions", func))

    for func in recurse_dict(get_py_nn_functions(declarations)):
        annotated_args.append(process_func("torch._C._nn", func))

    annotated_args = textwrap.indent("\n".join(annotated_args), "    ")
    env = {"annotated_args": annotated_args}
    PY_ANNOTATED_ARGS = CodeTemplate.from_file(template_path + '/templates/annotated_fn_args.py')
    write(out, 'annotated_fn_args.py', PY_ANNOTATED_ARGS, env)


def process_func(namespace, func):
    return f"{namespace}.{op_name(func)}: {func!r},"


def recurse_dict(d):
    for e in d.values():
        for i in e:
            yield i
