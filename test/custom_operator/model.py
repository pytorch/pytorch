import argparse
import os.path
import sys

import torch


def get_custom_op_library_path():
    if sys.platform.startswith("win32"):
        library_filename = "custom_ops.dll"
    elif sys.platform.startswith("darwin"):
        library_filename = "libcustom_ops.dylib"
    else:
        library_filename = "libcustom_ops.so"
    path = os.path.abspath("build/{}".format(library_filename))
    assert os.path.exists(path), path
    return path


class Model(torch.jit.ScriptModule):
    def __init__(self):
        super(Model, self).__init__()
        self.p = torch.nn.Parameter(torch.eye(5))

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


if __name__ == "__main__":
    main()
