# mypy: allow-untyped-defs
from textwrap import dedent

from .common import DeviceOpOverrides, register_device_op_overrides


class CpuDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def set_device(self, device_idx):
        return "pass"

    def synchronize(self):
        return "pass"

    def device_guard(self, device_idx):
        return "pass"


register_device_op_overrides("cpu", CpuDeviceOpOverrides())
