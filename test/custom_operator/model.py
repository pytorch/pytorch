import argparse
import os.path
import sys

import torch


SHARED_LIBRARY_EXTENSIONS = {'linux': 'so', 'darwin': 'dylib', 'win32': 'dll'}


def get_custom_op_library_path():
    extension = SHARED_LIBRARY_EXTENSIONS[sys.platform]
    path = os.path.abspath('build/libcustom_ops.{}'.format(extension))
    assert os.path.exists(path), path
    return path


class Model(torch.jit.ScriptModule):
    def __init__(self):
        super(Model, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        return torch.ops.custom.op_with_defaults(input)[0] + 1


def main():
    parser = argparse.ArgumentParser(
        description="Serialize a script module with custom ops"
    )
    parser.add_argument("--export-script-module-to", required=True)
    options = parser.parse_args()

    torch.ops.load_library(get_custom_op_library_path())

    model = Model()
    model.save(options.export_script_module_to)


if __name__ == '__main__':
    main()
