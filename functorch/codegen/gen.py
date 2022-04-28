import os
import argparse
import pathlib

from torchgen.gen import FileManager, parse_native_yaml
from gen_vmap_plumbing import gen_all_vmap_plumbing

"""
INSTRUCTIONS

Step 1: You must have a PyTorch installation (in develop mode, i.e.
installed with python setup.py develop) in your current environment.
This script relies on the `tools` module from the PyTorch develop installation.

Step 2: Run this script.

# Replace the last argument with your path to native_functions.yaml
python codegen/gen.py -s /scratch/rzou/pt/debug-cpu/aten/src/ATen

NB: PyTorch's `tools` module is a giant hack (it somehow gets installed into your
environment when one does python setup.py develop), but it's highly likely that
PyTorch won't change it anytime soon because it's very messy to modify.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description='functorch codegen')
    parser.add_argument(
        '-s',
        '--source-path',
        help='path to source directory for ATen',
        default='/scratch/rzou/pt/debug-cpu/aten/src/ATen')
    parser.add_argument(
        '-o',
        '--output-dependencies',
        help='output a list of dependencies into the given file and exit')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='run without writing any files (still updates outputs)')
    parser.add_argument(
        '-d', '--install_dir', help='output directory',
        default='functorch/csrc')
    options = parser.parse_args()

    native_yaml_path = os.path.join(options.source_path, 'native/native_functions.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions, _ = parsed_yaml.native_functions, parsed_yaml.backend_indices
    template_dir = os.path.join(options.source_path, "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=options.dry_run)

    cpu_fm = make_file_manager(options.install_dir)
    cpu_fm.write('VmapGeneratedPlumbing.h', lambda: gen_all_vmap_plumbing(native_functions))

    if options.output_dependencies:
        depfile_path = pathlib.Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        for fm, prefix in [
                (cpu_fm, ""),
        ]:
            varname = prefix + depfile_stem
            path = depfile_path.parent / (prefix + depfile_name)
            fm.write_outputs(varname, str(path))


if __name__ == '__main__':
    main()
