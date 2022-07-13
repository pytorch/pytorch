import os
import argparse
from torchgen.gen import FileManager, parse_native_yaml
from torchgen.gen import get_torchgen_root
from gen_vmap_plumbing import gen_all_vmap_plumbing

"""
INSTRUCTIONS

Step 1: You must have a PyTorch installation (in develop mode, i.e.
installed with python setup.py develop) in your current environment.
This script relies on the `tools` module from the PyTorch develop installation.

Step 2: Run this script.

python codegen/gen.py
"""


def main() -> None:
    parser = argparse.ArgumentParser(description='functorch codegen')
    parser.add_argument(
        '-s',
        '--source-path',
        help='path to source directory for ATen',
        default=None)
    parser.add_argument(
        '-d', '--install_dir', help='output directory',
        default='functorch/csrc')
    options = parser.parse_args()
    generate_code(options.install_dir, options.source_path)


def generate_code(install_dir='functorch/csrc', source_path=None):
    if source_path is None:
        # infer the source path via torchgen
        source_path = os.path.join(get_torchgen_root(), "packaged/ATen")

    native_yaml_path = os.path.join(source_path, 'native/native_functions.yaml')
    tags_path = os.path.join(source_path, 'native/tags.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_path)
    native_functions, _ = parsed_yaml.native_functions, parsed_yaml.backend_indices
    template_dir = os.path.join(source_path, "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=False)

    cpu_fm = make_file_manager(install_dir)
    cpu_fm.write('VmapGeneratedPlumbing.h', lambda: gen_all_vmap_plumbing(native_functions))


if __name__ == '__main__':
    main()
