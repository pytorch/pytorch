import argparse
import torch


def to_custom_backend(module):
    """
    This is a helper that wraps torch._C._jit_to_test_backend and compiles
    only the forward method with an empty compile spec.

    Args:
        module: input ScriptModule.

    Returns:
        The module, lowered so that it can run on TestBackend.
    """
    lowered_module = torch._C._jit_to_test_backend(module._c, {"forward": {"": ""}})
    return lowered_module


class Model(torch.nn.Module):
    """
    Simple model used for testing that to_backend API supports saving, loading,
    and executing in C++.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b):
        return (a + b, a - b)


def main():
    parser = argparse.ArgumentParser(
        description="Lower a Module to a custom backend"
    )
    parser.add_argument("--export-module-to", required=True)
    options = parser.parse_args()

    # Lower an instance of Model to the custom backend (TestBackend) and export it
    # to the specified location.
    lowered_module = to_custom_backend(torch.jit.script(Model()))
    torch.jit.save(lowered_module, options.export_module_to)


if __name__ == "__main__":
    main()
