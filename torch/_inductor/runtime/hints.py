import collections
import typing
from dataclasses import fields
from enum import auto, Enum
from typing import Optional


# NOTE: if these fail asserts submit a PR to increase them
TRITON_MAX_BLOCK = {
    "X": 2048,
    "Y": 1024,
    "Z": 1024,
    "R": 4096 * 16,  # * 16 is multi-kernel only
}


class ReductionHint(Enum):
    INNER = 0
    OUTER = 1
    OUTER_TINY = 2
    DEFAULT = 3


class TileHint(Enum):
    SQUARE = 0
    DEFAULT = 1


# Attempt to import AttrsDescriptor from Triton
try:
    from triton.compiler.compiler import AttrsDescriptor

    attrs_descriptor_available = True
    # Determine if 'ids_of_folded_args' is a valid field for AttrsDescriptor
    attr_desc_fields = {f.name for f in fields(AttrsDescriptor)}
    ids_of_folded_args_available = "ids_of_folded_args" in attr_desc_fields
    divisible_by_8_available = "divisible_by_8" in attr_desc_fields
except ImportError:
    attrs_descriptor_available = False

# Define `instance_descriptor` function with clear conditional handling
if attrs_descriptor_available:

    def instance_descriptor(
        divisible_by_16=None,
        equal_to_1=None,
        ids_of_folded_args=None,
        divisible_by_8=None,
    ):
        # Prepare the arguments for AttrsDescriptor
        kwargs = {
            "divisible_by_16": divisible_by_16,
            "equal_to_1": equal_to_1,
        }

        # Conditionally add 'ids_of_folded_args' if it's available in AttrsDescriptor
        if ids_of_folded_args_available:
            kwargs["ids_of_folded_args"] = ids_of_folded_args
        if divisible_by_8_available:
            kwargs["divisible_by_8"] = divisible_by_8

        # Instantiate AttrsDescriptor with the prepared arguments
        return AttrsDescriptor(**kwargs)

else:
    # Define a namedtuple as a fallback when AttrsDescriptor is not available
    instance_descriptor = collections.namedtuple(  # type: ignore[no-redef]
        "instance_descriptor",
        ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"],
        defaults=[tuple(), tuple(), tuple(), tuple()],
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
    ELEMENTS_PER_WARP_32 = 0

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

    @classmethod
    def create(cls, device):
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type if torch.version.hip is None else "hip"
        device_interface = get_interface_for_device(device)
        if device_type == "cuda":
            props = device_interface.get_device_properties(device)
            return cls(
                type=device_type,
                index=device.index,
                cc=device_interface.get_compute_capability(device),
                major=props.major,
                regs_per_multiprocessor=props.regs_per_multiprocessor,
                max_threads_per_multi_processor=props.max_threads_per_multi_processor,
                multi_processor_count=props.multi_processor_count,
            )
        return cls(
            type=device_type,
            index=device.index,
            cc=device_interface.get_compute_capability(device),
        )
