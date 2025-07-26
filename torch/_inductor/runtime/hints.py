# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import functools
import typing
from enum import auto, Enum
from typing import Optional, Union

from torch.utils._triton import has_triton_package


# The following maximums only apply to runtime autotuning, when using FixedTritonConfig one may see larger values
# NOTE: if these fail asserts submit a PR to increase them
TRITON_MAX_BLOCK = {
    "X": 4096,
    "Y": 1024,
    "Z": 1024,
    "R0_": 4096 * 16,  # * 16 is multi-kernel only
    "R1_": 2048 * 16,  # * 16 is multi-kernel only
}
TRITON_MAX_RSPLIT = 64


class ReductionHint(Enum):
    INNER = 0
    OUTER = 1
    OUTER_TINY = 2
    DEFAULT = 3


class TileHint(Enum):
    SQUARE = 0
    DEFAULT = 1


# Define `AttrsDescriptorWrapper` function with clear conditional handling
if has_triton_package():
    import triton
    import triton.backends.compiler
    import triton.compiler.compiler

    if hasattr(triton.backends.compiler, "AttrsDescriptor"):
        # Triton 3.2.0 - the second implementation
        from triton.backends.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
        ):
            # Prepare the arguments for AttrsDescriptor
            kwargs = {
                "tt.divisibility": divisible_by_16,
                "tt.equal_to": equal_to_1,
            }

            # Instantiate AttrsDescriptor with the prepared arguments
            res = AttrsDescriptor.from_dict(
                {"arg_properties": kwargs, "cls": AttrsDescriptor.__name__}
            )
            assert res.property_values["tt.divisibility"] == 16
            assert res.property_values["tt.equal_to"] == 1
            return res

    elif hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        # Triton 3.0.0 - the original implementation
        from triton.compiler.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
        ):
            # Prepare the arguments for AttrsDescriptor
            kwargs = {
                "divisible_by_16": divisible_by_16,
                "equal_to_1": equal_to_1,
            }

            # Instantiate AttrsDescriptor with the prepared arguments
            return AttrsDescriptor(**kwargs)

    else:
        # Triton in 2025:
        # note: there's also a range of triton commits not currently supported
        # from ~Dec 9, 2024 to Jan 1 2025, in which AttrsDescriptors are still
        # used, but the contents are different.

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
        ):
            return {(x,): [["tt.divisibility", 16]] for x in divisible_by_16}

else:
    # Define a namedtuple as a fallback when AttrsDescriptor is not available
    AttrsDescriptorWrapper = collections.namedtuple(  # type: ignore[no-redef, name-match]
        "AttrsDescriptor",
        ["divisible_by_16", "equal_to_1"],
        defaults=[(), ()],
    )


_NUM_THREADS_PER_WARP = 32


class HeuristicType(Enum):
    PERSISTENT_REDUCTION = auto()
    POINTWISE = auto()
    REDUCTION = auto()
    SPLIT_SCAN = auto()
    TEMPLATE = auto()
    USER_AUTOTUNE = auto()
    FIXED = auto()


class AutotuneHint(Enum):
    ONE_ELEMENT_PER_THREAD = 0

    # Triton codegen tries to codegen set of AutotuneHints.
    # Enum.__repr__ looks like "<AutotuneHint.ELEMENTS_PER_WARP_32: 0>""
    # which isn't valid python.
    # Enum.__str__ will just return "AutotuneHint.ELEMENTS_PER_WARP_32".
    __repr__ = Enum.__str__


class DeviceProperties(typing.NamedTuple):
    """Copy device properties into a data structure not requiring torch to be imported"""

    type: str  # type: ignore[assignment]
    index: int  # type: ignore[assignment]
    multi_processor_count: int
    cc: int
    major: Optional[int] = None
    regs_per_multiprocessor: Optional[int] = None
    max_threads_per_multi_processor: Optional[int] = None
    warp_size: Optional[int] = None

    @classmethod
    @functools.cache
    def create(cls, device) -> DeviceProperties:
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type

        if torch.version.hip and device_type == "cuda":
            device_type = "hip"

        device_interface = get_interface_for_device(device)
        props = device_interface.get_device_properties(device)
        try:
            multi_processor_count = props.multi_processor_count
        except AttributeError:
            if device_type == "xpu":
                multi_processor_count = props.gpu_subslice_count
            elif device_type == "mps":
                # TODO: Fetch the actual value from ioreg
                multi_processor_count = 8
            elif device_type == "mtia":
                multi_processor_count = 64
            else:
                raise
        return cls(
            type=device_type,
            index=device.index,
            multi_processor_count=multi_processor_count,
            cc=device_interface.get_compute_capability(device),
            major=getattr(props, "major", None),
            regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
            max_threads_per_multi_processor=getattr(
                props, "max_threads_per_multi_processor", None
            ),
            warp_size=getattr(props, "warp_size", 32 if device_type != "cpu" else None),
        )


class HalideInputSpec(typing.NamedTuple):
    ctype: str
    name: str
    shape: Optional[list[str]] = None
    stride: Optional[list[str]] = None
    offset: Optional[str] = None
    alias_of: Optional[str] = None

    def bindings_type(self) -> str:
        if self.ctype in ("at::Half*", "at::BFloat16*"):
            return "uint16_t*"  # half not defined
        return self.ctype

    def halide_type(self) -> str:
        if self.ctype == "at::Half*":
            return "halide_type_t(halide_type_float, 16)"  # half not defined
        if self.ctype == "at::BFloat16*":
            return "halide_type_t(halide_type_bfloat, 16)"  # half not defined
        return f"halide_type_of<{self.ctype.replace('*', '')}>()"

    def is_scalar(self) -> bool:
        return self.shape is None

    def is_buffer(self) -> bool:
        return self.shape is not None


class HalideMeta(typing.NamedTuple):
    argtypes: list[HalideInputSpec]
    target: str
    scheduler: Optional[str] = None
    scheduler_flags: Optional[dict[str, Union[int, str]]] = None
    cuda_device: Optional[int] = None

    def args(self) -> list[str]:
        """Command line args to pass to halide generator"""
        args = [f"target={self.target}"]
        if self.scheduler:
            args.append(f"autoscheduler={self.scheduler}")
        if self.scheduler_flags:
            assert self.scheduler
            for k, v in self.scheduler_flags.items():
                args.append(f"autoscheduler.{k}={v}")
        return args

    def is_cuda(self) -> bool:
        return self.cuda_device is not None
