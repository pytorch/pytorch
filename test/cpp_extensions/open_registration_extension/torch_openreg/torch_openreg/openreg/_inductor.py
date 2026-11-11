"""Inductor codegen support for the openreg device.

OpenReg is a CPU-simulated accelerator with no compute ISA of its own, so it
reuses the CPU C++ codegen classes. Direct pointer access from generated
kernels would trip OpenReg's PROT_NONE device-memory protection, so graphs
should be compiled with fallback_by_default=True and
always_keep_tensor_constants=True: every op then dispatches through ATen and
lands on OpenReg's registered kernels, while Inductor still drives buffer
planning and (for AOTI) the C++ wrapper.
"""

from torch._inductor.codegen.common import (
    NoOpDeviceOpOverrides,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen


class OpenRegCppWrapperCodegen(CppWrapperCpu):
    def __init__(self):
        # Must be set before CppWrapperCpu.__init__, which defaults to "cpu".
        self.device = "openreg"
        super().__init__()

    # CppWrapperCpu.create is a staticmethod hardcoded to CppWrapperCpu(),
    # which would silently drop this subclass.
    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        return OpenRegCppWrapperCodegen()


class OpenRegDeviceOpOverrides(NoOpDeviceOpOverrides):
    def aten_device_type(self) -> str:
        return "at::kPrivateUse1"

    def has_aten_c_shims(self) -> bool:
        return False


def register_inductor_backend() -> None:
    register_device_op_overrides("openreg", OpenRegDeviceOpOverrides())
    register_backend_for_device(
        "openreg",
        CppScheduling,
        PythonWrapperCodegen,
        OpenRegCppWrapperCodegen,
        WrapperFxCodegen,
    )
