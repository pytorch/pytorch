from __future__ import annotations

from textwrap import dedent

import torch
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen

import torch_openreg._C  # type: ignore[import]


class OpenRegMemoryGuard:
    def unprotect(self, args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "openreg":
                torch_openreg._C._memory_unprotect(arg)

    def protect(self, args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "openreg":
                torch_openreg._C._memory_protect(arg)


openreg_memory_guard = OpenRegMemoryGuard()


class OpenRegDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def set_device(self, device_idx):
        return f"torch_openreg._C._set_device({device_idx})"

    def synchronize(self):
        return "pass"

    def device_guard(self, device_idx):
        return "torch._ops.contextlib.nullcontext()"

    def cpp_kernel_type(self):
        return "void*"


class OpenRegWrapperCodegen(PythonWrapperCodegen):
    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        if is_subgraph:
            from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen

            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return OpenRegWrapperCodegen()

    def write_header(self):
        super().write_header()
        self.header.splice(
            "from torch_openreg.inductor_backend import openreg_memory_guard"
        )

    def generate_kernel_call(self, kernel_name, call_args, *, device=None, **kwargs):
        from torch._inductor.virtualized import V

        device = device or V.graph.get_current_device_or_throw()
        if device.type == "openreg":
            self.writeline(f"openreg_memory_guard.unprotect([{', '.join(call_args)}])")
        kwargs["device"] = device
        super().generate_kernel_call(kernel_name, call_args, **kwargs)
        if device.type == "openreg":
            self.writeline(f"openreg_memory_guard.protect([{', '.join(call_args)}])")

    def _generate_kernel_call_helper(
        self, kernel_name, call_args, *, device=None, **kwargs
    ):
        from torch._inductor.virtualized import V

        device = device or V.graph.get_current_device_or_throw()
        if device.type == "openreg":
            device = torch.device("cpu")
        super()._generate_kernel_call_helper(
            kernel_name, call_args, device=device, **kwargs
        )


register_device_op_overrides("openreg", OpenRegDeviceOpOverrides())
register_backend_for_device(
    "openreg",
    CppScheduling,
    OpenRegWrapperCodegen,
)
