from typing import Optional

from torch._inductor import ir
from torch._inductor.codegen import cpp, cpp_wrapper_cpu, wrapper
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V


class ExtensionWrapperCodegen(wrapper.PythonWrapperCodegen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        return ExtensionWrapperCodegen()

    def _generate_kernel_call_helper(self, kernel_name, call_args, *, device=None, **kwargs):
        device = device or V.graph.get_current_device_or_throw()
        if device.type == "extension_device":
            import torch
            device = torch.device("cpu")
        super()._generate_kernel_call_helper(
            kernel_name, call_args, device=device, **kwargs
        )


class ExtensionCppWrapperCodegen(cpp_wrapper_cpu.CppWrapperCpu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        return ExtensionCppWrapperCodegen()

    @staticmethod
    def get_device_include_path(device: str) -> str:
        if device == "extension_device":
            return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path("cpu")
        return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path(device)


class ExtensionScheduling(BaseScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return True

    def can_fuse_horizontal(self, node1, node2):
        return True

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_template(self, template_node, epilogue_nodes):
        pass

    def codegen_node(self, node):
        self._scheduling.codegen_node(node)

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()
