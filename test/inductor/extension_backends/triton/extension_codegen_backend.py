from torch._inductor.codegen import triton, wrapper
from torch._inductor.codegen.common import DeviceOpOverrides
from torch._inductor.scheduler import BaseScheduling


class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        super().__init__()


class ExtensionScheduling(BaseScheduling):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._triton_scheduling = triton.TritonScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return True

    def can_fuse_horizontal(self, node1, node2):
        return True

    def group_fn(self, sizes):
        return self._triton_scheduling.group_fn(sizes)

    def codegen_template(self, template_node, epilogue_nodes):
        pass

    def codegen_node(self, node):
        self._triton_scheduling.codegen_node(node)

    def codegen_sync(self):
        pass

    def flush(self):
        self._triton_scheduling.flush()


class CPUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"def {name}(name): None\n"

    def set_device(self, device_idx: int) -> str:  # noqa: ARG002 unused-argument
        return ""

    def synchronize(self) -> None:
        pass

    def device_guard(self, device_idx: int) -> str:  # noqa: ARG002 unused-argument
        return ""
