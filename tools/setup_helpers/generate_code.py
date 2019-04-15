import argparse
import os
import sys

source_files = {'.py', '.cpp', '.h'}

DECLARATIONS_PATH = 'torch/share/ATen/Declarations.yaml'


# TODO: This is a little inaccurate, because it will also pick
# up setup_helper scripts which don't affect code generation
def all_generator_source():
    r = []
    for directory, _, filenames in os.walk('tools'):
        for f in filenames:
            if os.path.splitext(f)[1] in source_files:
                full = os.path.join(directory, f)
                r.append(full)
    return sorted(r)


inputs = [
    'torch/lib/THNN.h',
    'torch/lib/THCUNN.h',
    'torch/share/ATen/Declarations.yaml',
    'tools/autograd/derivatives.yaml',
    'tools/autograd/deprecated.yaml',
]

outputs = [
    'torch/csrc/autograd/generated/Functions.cpp',
    'torch/csrc/autograd/generated/Functions.h',
    'torch/csrc/autograd/generated/python_functions.cpp',
    'torch/csrc/autograd/generated/python_functions.h',
    'torch/csrc/autograd/generated/python_nn_functions.cpp',
    'torch/csrc/autograd/generated/python_nn_functions.h',
    'torch/csrc/autograd/generated/python_nn_functions_dispatch.h',
    'torch/csrc/autograd/generated/python_variable_methods.cpp',
    'torch/csrc/autograd/generated/python_variable_methods_dispatch.h',
    'torch/csrc/autograd/generated/variable_factories.h',
    'torch/csrc/autograd/generated/VariableType_0.cpp',
    'torch/csrc/autograd/generated/VariableType_1.cpp',
    'torch/csrc/autograd/generated/VariableType_2.cpp',
    'torch/csrc/autograd/generated/VariableType_3.cpp',
    'torch/csrc/autograd/generated/VariableType_4.cpp',
    'torch/csrc/autograd/generated/VariableType.h',
    'torch/csrc/jit/generated/register_aten_ops_0.cpp',
    'torch/csrc/jit/generated/register_aten_ops_1.cpp',
    'torch/csrc/jit/generated/register_aten_ops_2.cpp',
]


def generate_code(ninja_global=None,
                  declarations_path=None,
                  nn_path=None,
                  install_dir=None):
    # cwrap depends on pyyaml, so we can't import it earlier
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from tools.autograd.gen_autograd import gen_autograd
    from tools.jit.gen_jit_dispatch import gen_jit_dispatch

    from tools.nnwrap import generate_wrappers as generate_nn_wrappers

    # Build THNN/THCUNN.cwrap and then THNN/THCUNN.cpp. These are primarily
    # used by the legacy NN bindings.
    generate_nn_wrappers(nn_path, install_dir, 'tools/cwrap/plugins/templates')

    # Build ATen based Variable classes
    autograd_gen_dir = install_dir or 'torch/csrc/autograd/generated'
    jit_gen_dir = install_dir or 'torch/csrc/jit/generated'
    for d in (autograd_gen_dir, jit_gen_dir):
        if not os.path.exists(d):
            os.makedirs(d)
    gen_autograd(declarations_path or DECLARATIONS_PATH, autograd_gen_dir, 'tools/autograd')
    gen_jit_dispatch(declarations_path or DECLARATIONS_PATH, jit_gen_dir, 'tools/jit/templates')


def main():
    parser = argparse.ArgumentParser(description='Autogenerate code')
    parser.add_argument('--declarations-path')
    parser.add_argument('--nn-path')
    parser.add_argument('--ninja-global')
    parser.add_argument('--install_dir')
    options = parser.parse_args()
    generate_code(options.ninja_global,
                  options.declarations_path,
                  options.nn_path,
                  options.install_dir)


if __name__ == "__main__":
    main()
