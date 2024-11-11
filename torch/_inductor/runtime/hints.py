# mypy: allow-untyped-defs
import collections
import typing
from enum import auto, Enum
from typing import Dict, List, Optional, Union


# NOTE: if these fail asserts submit a PR to increase them
TRITON_MAX_BLOCK = {
    "X": 4096,
    "Y": 1024,
    "Z": 1024,
    "R": 4096 * 16,  # * 16 is multi-kernel only
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


def _is_triton_available():
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


# Define `AttrsDescriptorWrapper` function with clear conditional handling
if _is_triton_available():
    try:
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

    except ImportError:
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
    cc: int
    major: Optional[int] = None
    regs_per_multiprocessor: Optional[int] = None
    max_threads_per_multi_processor: Optional[int] = None
    multi_processor_count: Optional[int] = None
    warp_size: Optional[int] = None

    @classmethod
    def create(cls, device):
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type

        if torch.version.hip and device_type == "cuda":
            device_type = "hip"

        device_interface = get_interface_for_device(device)
        if device_type in ["cuda", "hip", "xpu"]:
            props = device_interface.get_device_properties(device)
            return cls(
                type=device_type,
                index=device.index,
                cc=device_interface.get_compute_capability(device),
                major=props.major if hasattr(props, "major") else None,
                regs_per_multiprocessor=props.regs_per_multiprocessor
                if hasattr(props, "regs_per_multiprocessor")
                else None,
                max_threads_per_multi_processor=props.max_threads_per_multi_processor
                if hasattr(props, "max_threads_per_multi_processor")
                else None,
                multi_processor_count=props.multi_processor_count
                if hasattr(props, "multi_processor_count")
                else None,
                warp_size=props.warp_size if hasattr(props, "warp_size") else 32,
            )
        return cls(
            type=device_type,
            index=device.index,
            cc=device_interface.get_compute_capability(device),
        )


class HalideInputSpec(typing.NamedTuple):
    ctype: str
    name: str
    shape: Optional[List[str]] = None
    stride: Optional[List[str]] = None
    offset: Optional[str] = None
    alias_of: Optional[str] = None

    def bindings_type(self):
        if self.ctype in ("half*", "bfloat16*"):
            return "uint16_t*"  # half not defined
        return self.ctype

    def halide_type(self):
        if self.ctype == "half*":
            return "halide_type_t(halide_type_float, 16)"  # half not defined
        if self.ctype == "bfloat16*":
            return "halide_type_t(halide_type_bfloat, 16)"  # half not defined
        return f"halide_type_of<{self.ctype.replace('*', '')}>()"

    def is_scalar(self):
        return self.shape is None

    def is_buffer(self):
        return self.shape is not None


class HalideMeta(typing.NamedTuple):
    argtypes: List[HalideInputSpec]
    target: str
    scheduler: Optional[str] = None
    scheduler_flags: Optional[Dict[str, Union[int, str]]] = None
    cuda_device: Optional[int] = None

    def args(self):
        """Command line args to pass to halide generator"""
        args = [f"target={self.target}"]
        if self.scheduler:
            args.append(f"autoscheduler={self.scheduler}")
        if self.scheduler_flags:
            assert self.scheduler
            for k, v in self.scheduler_flags.items():
                args.append(f"autoscheduler.{k}={v}")
        return args

    def is_cuda(self):
        return self.cuda_device is not None
