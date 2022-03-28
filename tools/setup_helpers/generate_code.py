import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import yaml
from typing import Any, List, Optional, cast

# Note that even though this is deprecated, a certain Meta internal
# build is not compatible with importlib.resources file listing.
import pkg_resources

try:
    # use faster C loader if available
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeLoader as YamlLoader  # type: ignore[misc]

source_files = {'.py', '.cpp', '.h'}

NATIVE_FUNCTIONS_PATH = 'aten/src/ATen/native/native_functions.yaml'

# TODO: This is a little inaccurate, because it will also pick
# up setup_helper scripts which don't affect code generation
def all_generator_source() -> List[str]:
    r = []
    for directory, _, filenames in os.walk('tools'):
        for f in filenames:
            if os.path.splitext(f)[1] in source_files:
                full = os.path.join(directory, f)
                r.append(full)
    return sorted(r)


def generate_code(ninja_global: Optional[str] = None,
                  native_functions_path: Optional[str] = None,
                  install_dir: Optional[str] = None,
                  subset: Optional[str] = None,
                  disable_autograd: bool = False,
                  force_schema_registration: bool = False,
                  operator_selector: Any = None,
                  *,
                  autograd_dir: pathlib.Path) -> None:
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
    for d in (autograd_gen_dir, python_install_dir):
        if not os.path.exists(d):
            os.makedirs(d)

    if subset == "pybindings" or not subset:
        gen_autograd_python(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            autograd_gen_dir,
            autograd_dir)

    if operator_selector is None:
        operator_selector = SelectiveBuilder.get_nop_selector()

    if subset == "libtorch" or not subset:

        gen_autograd(
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
) -> Any:
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


def get_selector(
    selected_op_list_path: Optional[str],
    operators_yaml_path: Optional[str],
) -> Any:
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
        return SelectiveBuilder.from_yaml_path(cast(str, operators_yaml_path))


def main() -> None:
    parser = argparse.ArgumentParser(description='Autogenerate code')
    parser.add_argument('--native-functions-path')
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
    parser.add_argument(
        '--gen_lazy_ts_backend',
        action='store_true',
        help='Enable generation of the torch::lazy TorchScript backend'
    )
    parser.add_argument(
        '--per_operator_headers',
        action='store_true',
        help='Build lazy tensor ts backend with per-operator ATen headers, must match how ATen was built'
    )
    options = parser.parse_args()

    with tempfile.TemporaryDirectory() as autograd_dir_str:
        # Copy all of the tools.autograd resources to a temporary
        # directory so that code generation below can see it as local
        # files.
        autograd_dir = pathlib.Path(autograd_dir_str)
        copy_resource_to_dir("tools.autograd", "deprecated.yaml", autograd_dir)
        copy_resource_to_dir("tools.autograd", "derivatives.yaml", autograd_dir)

        templates_dir = autograd_dir / "templates/"
        templates_dir.mkdir()
        for template_name in pkg_resources.resource_listdir("tools.autograd", "templates"):
            if pkg_resources.resource_isdir("tools.autograd.templates", template_name):
                # Only copy one level deep.
                continue
            copy_resource_to_dir("tools.autograd.templates", template_name, templates_dir)

        generate_code(
            options.ninja_global,
            options.native_functions_path,
            options.install_dir,
            options.subset,
            options.disable_autograd,
            options.force_schema_registration,
            # options.selected_op_list
            operator_selector=get_selector(options.selected_op_list_path,
                                           options.operators_yaml_path),
            autograd_dir=autograd_dir,
        )

    if options.gen_lazy_ts_backend:
        aten_path = os.path.dirname(os.path.dirname(options.native_functions_path))
        ts_backend_yaml = os.path.join(aten_path, 'native/ts_native_functions.yaml')
        ts_native_functions = "torch/csrc/lazy/ts_backend/ts_native_functions.cpp"
        ts_node_base = "torch/csrc/lazy/ts_backend/ts_node.h"
        if options.install_dir is None:
            options.install_dir = "torch/csrc"
        lazy_install_dir = os.path.join(options.install_dir, "lazy/generated")
        if not os.path.exists(lazy_install_dir):
            os.makedirs(lazy_install_dir)

        assert os.path.isfile(ts_backend_yaml), f"Unable to access ts_backend_yaml: {ts_backend_yaml}"
        assert os.path.isfile(ts_native_functions), f"Unable to access {ts_native_functions}"
        from tools.codegen.gen_lazy_tensor import run_gen_lazy_tensor
        run_gen_lazy_tensor(aten_path=aten_path,
                            source_yaml=ts_backend_yaml,
                            output_dir=lazy_install_dir,
                            dry_run=False,
                            impl_path=ts_native_functions,
                            gen_ts_lowerings=True,
                            node_base="TsNode",
                            node_base_hdr=ts_node_base,
                            build_in_tree=True,
                            per_operator_headers=options.per_operator_headers)


def copy_resource_to_dir(package: str, name: str, dest_dir: pathlib.Path) -> None:
    """Copies the file resource from the package to the given directory."""
    # Note that this will read the local file in the event that there
    # is no archive to extract from, e.g. normal CMake and Bazel
    # builds.
    with pkg_resources.resource_stream(package, name) as in_file:
        dest = dest_dir / name
        with dest.open(mode="xb") as out_file:
            shutil.copyfileobj(in_file, out_file)


if __name__ == "__main__":
    main()
