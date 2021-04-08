import argparse
import os
import sys
import yaml

try:
    # use faster C loader if available
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeLoader as YamlLoader

source_files = {'.py', '.cpp', '.h'}

DECLARATIONS_PATH = 'torch/share/ATen/Declarations.yaml'
NATIVE_FUNCTIONS_PATH = 'aten/src/ATen/native/native_functions.yaml'

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
                  native_functions_path=None,
                  install_dir=None,
                  subset=None,
                  disable_autograd=False,
                  force_schema_registration=False,
                  operator_selector=None):
    from tools.autograd.gen_autograd import gen_autograd, gen_autograd_python
    from tools.autograd.gen_annotated_fn_args import gen_annotated
    from tools.codegen.selective_build.selector import SelectiveBuilder


    # Build ATen based Variable classes
    if install_dir is None:
        install_dir = 'torch/csrc'
        python_install_dir = 'torch/testing/_internal/generated'
    else:
        python_install_dir = install_dir
    autograd_gen_dir = os.path.join(install_dir, 'autograd', 'generated')
    jit_gen_dir = os.path.join(install_dir, 'jit', 'generated')
    for d in (autograd_gen_dir, jit_gen_dir, python_install_dir):
        if not os.path.exists(d):
            os.makedirs(d)
    runfiles_dir = os.environ.get("RUNFILES_DIR", None)
    data_dir = os.path.join(runfiles_dir, 'pytorch') if runfiles_dir else ''
    autograd_dir = os.path.join(data_dir, 'tools', 'autograd')
    tools_jit_templates = os.path.join(data_dir, 'tools', 'jit', 'templates')

    if subset == "pybindings" or not subset:
        gen_autograd_python(
            declarations_path or DECLARATIONS_PATH,
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            autograd_gen_dir,
            autograd_dir)

    if operator_selector is None:
        operator_selector = SelectiveBuilder.get_nop_selector()

    if subset == "libtorch" or not subset:

        gen_autograd(
            declarations_path or DECLARATIONS_PATH,
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            autograd_gen_dir,
            autograd_dir,
            disable_autograd=disable_autograd,
            operator_selector=operator_selector,
        )

    if subset == "python" or not subset:
        gen_annotated(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            python_install_dir,
            autograd_dir)


def get_selector_from_legacy_operator_selection_list(
        selected_op_list_path: str,
):
    with open(selected_op_list_path, 'r') as f:
        # strip out the overload part
        # It's only for legacy config - do NOT copy this code!
        selected_op_list = {
            opname.split('.', 1)[0] for opname in yaml.load(f, Loader=YamlLoader)
        }

    # Internal build doesn't use this flag any more. Only used by OSS
    # build now. Every operator should be considered a root operator
    # (hence generating unboxing code for it, which is consistent with
    # the current behaviour), and also be considered as used for
    # training, since OSS doesn't support training on mobile for now.
    #
    is_root_operator = True
    is_used_for_training = True

    from tools.codegen.selective_build.selector import SelectiveBuilder
    selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
        selected_op_list,
        is_root_operator,
        is_used_for_training,
    )

    return selector


def get_selector(selected_op_list_path, operators_yaml_path):
    # cwrap depends on pyyaml, so we can't import it earlier
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from tools.codegen.selective_build.selector import SelectiveBuilder

    assert not (selected_op_list_path is not None and
                operators_yaml_path is not None), \
        ("Expected at most one of selected_op_list_path and " +
         "operators_yaml_path to be set.")

    if selected_op_list_path is None and operators_yaml_path is None:
        return SelectiveBuilder.get_nop_selector()
    elif selected_op_list_path is not None:
        return get_selector_from_legacy_operator_selection_list(selected_op_list_path)
    else:
        return SelectiveBuilder.from_yaml_path(operators_yaml_path)


def main():
    parser = argparse.ArgumentParser(description='Autogenerate code')
    parser.add_argument('--declarations-path')
    parser.add_argument('--native-functions-path')
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
    parser.add_argument(
        '--selected-op-list-path',
        help='Path to the YAML file that contains the list of operators to include for custom build.',
    )
    parser.add_argument(
        '--operators_yaml_path',
        help='Path to the model YAML file that contains the list of operators to include for custom build.',
    )
    parser.add_argument(
        '--force_schema_registration',
        action='store_true',
        help='force it to generate schema-only registrations for ops that are not'
        'listed on --selected-op-list'
    )
    options = parser.parse_args()

    generate_code(
        options.ninja_global,
        options.declarations_path,
        options.nn_path,
        options.native_functions_path,
        options.install_dir,
        options.subset,
        options.disable_autograd,
        options.force_schema_registration,
        # options.selected_op_list
        operator_selector=get_selector(options.selected_op_list_path, options.operators_yaml_path),
    )


if __name__ == "__main__":
    main()
