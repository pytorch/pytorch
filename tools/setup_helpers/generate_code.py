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


def generate_code(ninja_global=None,
                  declarations_path=None,
                  nn_path=None,
                  install_dir=None,
                  subset=None,
                  disable_autograd=False):
    # cwrap depends on pyyaml, so we can't import it earlier
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from tools.autograd.gen_autograd import gen_autograd, gen_autograd_python
    from tools.jit.gen_jit_dispatch import gen_jit_dispatch

    # Build ATen based Variable classes
    autograd_gen_dir = install_dir or 'torch/csrc/autograd/generated'
    jit_gen_dir = install_dir or 'torch/csrc/jit/generated'
    for d in (autograd_gen_dir, jit_gen_dir):
        if not os.path.exists(d):
            os.makedirs(d)

    if subset == "pybindings" or not subset:
        gen_autograd_python(declarations_path or DECLARATIONS_PATH, autograd_gen_dir, 'tools/autograd')

    if subset == "libtorch" or not subset:
        gen_autograd(
            declarations_path or DECLARATIONS_PATH,
            autograd_gen_dir,
            'tools/autograd',
            disable_autograd=disable_autograd,
        )
        gen_jit_dispatch(
            declarations_path or DECLARATIONS_PATH,
            jit_gen_dir,
            'tools/jit/templates',
            disable_autograd=disable_autograd)


def main():
    parser = argparse.ArgumentParser(description='Autogenerate code')
    parser.add_argument('--declarations-path')
    parser.add_argument('--nn-path')
    parser.add_argument('--ninja-global')
    parser.add_argument('--install_dir')
    parser.add_argument(
        '--subset',
        help='Subset of source files to generate. Can be "libtorch" or "pybindings". Generates both when omitted.'
    )
    parser.add_argument(
        '--disable-autograd',
        default=False,
        action='store_true',
        help='It can skip generating autograd related code when the flag is set',
    )
    options = parser.parse_args()
    generate_code(
        options.ninja_global,
        options.declarations_path,
        options.nn_path,
        options.install_dir,
        options.subset,
        options.disable_autograd,
    )


if __name__ == "__main__":
    main()
