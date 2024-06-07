from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.package import PackageExporter
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE

packaging_directory = f"{Path(__file__).parent}/package_bc"
torch.package.package_exporter._gate_torchscript_serialization = False


def generate_bc_packages():
    """Function to create packages for testing backwards compatiblity"""
    if not IS_FBCODE or IS_SANDCASTLE:
        from package_a.test_nn_module import TestNnModule

        test_nn_module = TestNnModule()
        test_torchscript_module = torch.jit.script(TestNnModule())
        test_fx_module: torch.fx.GraphModule = symbolic_trace(TestNnModule())
        with PackageExporter(f"{packaging_directory}/test_nn_module.pt") as pe1:
            pe1.intern("**")
            pe1.save_pickle("nn_module", "nn_module.pkl", test_nn_module)
        with PackageExporter(
            f"{packaging_directory}/test_torchscript_module.pt"
        ) as pe2:
            pe2.intern("**")
            pe2.save_pickle(
                "torchscript_module", "torchscript_module.pkl", test_torchscript_module
            )
        with PackageExporter(f"{packaging_directory}/test_fx_module.pt") as pe3:
            pe3.intern("**")
            pe3.save_pickle("fx_module", "fx_module.pkl", test_fx_module)


if __name__ == "__main__":
    generate_bc_packages()
