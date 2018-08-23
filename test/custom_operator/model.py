import argparse
import os.path

import torch


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

    torch.ops.load_library(os.path.abspath('build/libcustom_ops.so'))

    model = Model()
    model.save(options.export_script_module_to)


if __name__ == '__main__':
    main()
