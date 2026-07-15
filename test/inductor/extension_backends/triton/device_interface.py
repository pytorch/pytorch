from __future__ import annotations

import time

import torch
from torch._dynamo import device_interface  # noqa: PLC2701 import-private-name


class DeviceProperties:
    def __init__(self) -> None:
        self.major = 8  # TODO: bypass check for H100 in triton_heuristics.py
        self.max_threads_per_multi_processor = 1
        self.multi_processor_count = 80


class DeviceInterface(device_interface.DeviceInterface):
    class Event(torch.Event):
        def __init__(
            self,
            enable_timing: bool = False,
            blocking: bool = False,
            interprocess: bool = False,
        ) -> None:
            self.enable_timing = enable_timing
            self.recorded_time: int | None = None

        def record(self, stream) -> None:
            if not self.enable_timing:
                return
            assert self.recorded_time is None
            self.recorded_time = time.perf_counter_ns()

        def elapsed_time(self, end_event: DeviceInterface.Event) -> float:
            assert self.recorded_time
            assert end_event.recorded_time
            # convert to ms
            return (end_event.recorded_time - self.recorded_time) / 1000000

        def wait(self, stream) -> None:
            pass

        def query(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

    class device:  # noqa: N801 invalid-class-name # pyright: ignore [reportIncompatibleVariableOverride]
        def __init__(self, device) -> None:
            self.device = device

    class Worker(device_interface.DeviceInterface.Worker):
        @staticmethod
        def set_device(device: int) -> None:
            # No device index for our backend
            pass

        @staticmethod
        def current_device() -> int:
            # No device index for our backend
            return 0

        @staticmethod
        def get_device_properties(
            device=None,
        ) -> DeviceProperties:
            return DeviceProperties()

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def set_device(device) -> None:
        pass

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        assert device == 0, (
            f"Only device index 0 is supported, tried to set index to {device}"
        )
        return 0  # previous device is always 0

    @staticmethod
    def exchange_device(device: int) -> int:
        assert device == 0, (
            f"Only device index 0 is supported, tried to set index to {device}"
        )
        return 0  # previous device is always 0

    @staticmethod
    def get_raw_stream(device_index: int):
        return None

    @staticmethod
    def synchronize(device) -> None:
        pass

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_compute_capability(device) -> int:
        return 0
