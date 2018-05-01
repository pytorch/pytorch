import argparse
import os
import sys

source_files = {'.py', '.cpp', '.h'}

DECLARATIONS_PATH = 'torch/lib/tmp_install/share/ATen/Declarations.yaml'


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
    'torch/lib/tmp_install/share/ATen/Declarations.yaml',
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
    'torch/csrc/autograd/generated/VariableType.cpp',
    'torch/csrc/autograd/generated/VariableType.h',
    'torch/csrc/jit/generated/aten_dispatch.cpp',
    'torch/csrc/jit/generated/aten_dispatch.h',
]


def generate_code_ninja(w):
    all_inputs = all_generator_source() + inputs
    cmd = "{} {}".format(sys.executable, 'tools/setup_helpers/generate_code.py')
    w.writer.build(
        outputs, 'do_cmd', all_inputs,
        variables={
            'cmd': cmd,
            # Note [Unchanging results for ninja]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # generate_code.py will avoid bumping the timestamp on its
            # output files if the contents of the generated file did not
            # change.  To let Ninja take advantage of this, it must stat
            # the output files after the build.  See
            # https://groups.google.com/forum/#!topic/ninja-build/rExDmgDL2oc
            # for a more detailed discussion.
            'restat': True,
        })


def generate_code(ninja_global=None,
                  declarations_path=None,
                  nn_path=None):
    # if ninja is enabled, we just register this file as something
    # ninja will need to call if needed
    if ninja_global is not None:
        return generate_code_ninja(ninja_global)

    # cwrap depends on pyyaml, so we can't import it earlier
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from tools.autograd.gen_autograd import gen_autograd
    from tools.jit.gen_jit_dispatch import gen_jit_dispatch
    from tools.nnwrap import generate_wrappers as generate_nn_wrappers

    # Build THNN/THCUNN.cwrap and then THNN/THCUNN.cpp. These are primarily
    # used by the legacy NN bindings.
    generate_nn_wrappers(nn_path)

    # Build ATen based Variable classes
    autograd_gen_dir = 'torch/csrc/autograd/generated'
    jit_gen_dir = 'torch/csrc/jit/generated'
    for d in (autograd_gen_dir, jit_gen_dir):
        if not os.path.exists(d):
            os.mkdir(d)
    gen_autograd(declarations_path or DECLARATIONS_PATH, autograd_gen_dir)
    gen_jit_dispatch(declarations_path or DECLARATIONS_PATH, jit_gen_dir)


def main():
    parser = argparse.ArgumentParser(description='Autogenerate code')
    parser.add_argument('--declarations-path')
    parser.add_argument('--nn-path')
    parser.add_argument('--ninja-global')
    options = parser.parse_args()
    generate_code(options.ninja_global,
                  options.declarations_path,
                  options.nn_path)


if __name__ == "__main__":
    main()
