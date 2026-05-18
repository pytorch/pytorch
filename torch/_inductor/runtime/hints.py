# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import contextvars
import functools
import typing
from enum import auto, Enum

import torch
from torch.utils._triton import has_triton_package


# The following maximums only apply to runtime autotuning, when using FixedTritonConfig one may see larger values
# NOTE: if these fail asserts submit a PR to increase them
TRITON_MAX_BLOCK = {
    "X": 8192 if torch.version.hip else 4096,
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
            pointer_range_32=None,
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
            pointer_range_32=None,
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
            pointer_range_32=None,
        ):
            # pyrefly: ignore [not-iterable]
            # Build attr dict merging divisibility and pointer_range per arg index,
            # since a single arg can carry both attributes.
            result = {(x,): [["tt.divisibility", 16]] for x in (divisible_by_16 or ())}
            for x in pointer_range_32 or ():
                key = (x,)
                if key in result:
                    result[key].append(["tt.pointer_range", 32])
                else:
                    result[key] = [["tt.pointer_range", 32]]
            return result

else:
    # Define a namedtuple as a fallback when AttrsDescriptor is not available
    AttrsDescriptorWrapper = collections.namedtuple(  # type: ignore[no-redef, name-match]
        # pyrefly: ignore [invalid-argument]
        "AttrsDescriptor",
        ["divisible_by_16", "equal_to_1", "pointer_range_32"],
        defaults=[(), (), ()],
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
    major: int | None = None
    regs_per_multiprocessor: int | None = None
    max_threads_per_multi_processor: int | None = None
    max_threads_per_block: int | None = None
    warp_size: int | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}.create_from_device_str("
            f"{self.type!r}, {self.index!r})"
        )

    @classmethod
    @functools.cache
    def create(cls, device) -> DeviceProperties:
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type

        if torch.version.hip and device_type == "cuda":
            device_type = "hip"

        device_interface = get_interface_for_device(device)
        props = device_interface.Worker.get_device_properties(device)
        try:
            multi_processor_count = props.multi_processor_count
        except AttributeError:
            if device_type == "xpu":
                multi_processor_count = props.gpu_subslice_count
            elif device_type == "mtia":
                multi_processor_count = 64
            else:
                raise
        return cls(
            type=device_type,
            index=device.index,
            multi_processor_count=multi_processor_count,
            cc=cls._compute_capability_from_properties(
                device_type, props, device_interface, device
            ),
            major=getattr(props, "major", None),
            regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
            max_threads_per_multi_processor=getattr(
                props, "max_threads_per_multi_processor", None
            ),
            max_threads_per_block=getattr(props, "max_threads_per_block", 1024),
            warp_size=getattr(props, "warp_size", 32 if device_type != "cpu" else None),
        )

    @staticmethod
    def _compute_capability_from_properties(
        device_type: str, props, device_interface, device
    ):
        if (
            device_type == "cuda"
            and hasattr(props, "major")
            and hasattr(props, "minor")
        ):
            return props.major * 10 + props.minor
        if device_type == "hip" and hasattr(props, "gcnArchName"):
            return props.gcnArchName.split(":", 1)[0]
        return device_interface.get_compute_capability(device)

    @classmethod
    def create_from_device_str(
        cls, device_str: str, index: int | None = None
    ) -> DeviceProperties:
        device_type, parsed_index = _device_type_and_index(device_str)
        if parsed_index is not None:
            index = parsed_index

        if (device_props := _triton_meta_device_props.get()) is not None:
            if _matches_device(device_props, device_type, index):
                return device_props

        # PyTorch exposes ROCm devices as cuda devices. DeviceProperties.create()
        # normalizes the resulting properties back to type="hip".
        torch_device_type = "cuda" if device_type == "hip" else device_type
        if index is None:
            device = torch.device(torch_device_type)
        else:
            device = torch.device(torch_device_type, index)

        if index is None and torch_device_type != "cpu":
            from torch._dynamo.device_interface import get_interface_for_device

            device_interface = get_interface_for_device(device)
            index = device_interface.current_device()
            device = torch.device(torch_device_type, index)

        return cls.create(device)


_triton_meta_device_str: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "triton_meta_device_str", default=None
)
_triton_meta_device_props: contextvars.ContextVar[DeviceProperties | None] = (
    contextvars.ContextVar("triton_meta_device_props", default=None)
)


def _device_type_and_index(device_str: str) -> tuple[str, int | None]:
    raw_device_type, _, raw_index = device_str.partition(":")
    torch_device_type = "cuda" if raw_device_type == "hip" else raw_device_type
    if raw_index:
        device = torch.device(torch_device_type, int(raw_index))
    else:
        device = torch.device(torch_device_type)
    device_type = "hip" if raw_device_type == "hip" else device.type
    return device_type, device.index


def _canonical_device_type(device_type: str) -> str:
    return "cuda" if device_type == "hip" else device_type


def _matches_device(
    device_props: DeviceProperties, device_type: str, index: int | None
) -> bool:
    return _canonical_device_type(device_type) == _canonical_device_type(
        device_props.type
    ) and (index is None or device_props.index == index)


@contextlib.contextmanager
def triton_meta_device_context(
    device_str: str, device_props: DeviceProperties | None = None
):
    device_str_token = _triton_meta_device_str.set(device_str)
    device_props_token = _triton_meta_device_props.set(device_props)
    try:
        yield
    finally:
        _triton_meta_device_props.reset(device_props_token)
        _triton_meta_device_str.reset(device_str_token)


def runtime_device_properties(device_props: DeviceProperties) -> DeviceProperties:
    device_str = _triton_meta_device_str.get()
    if device_str is None:
        return device_props

    device_type, parsed_index = _device_type_and_index(device_str)
    if not _matches_device(device_props, device_type, None):
        return device_props

    return DeviceProperties.create_from_device_str(device_str, parsed_index)


def triton_meta_device_cache_key(
    device_str: str, device_props: DeviceProperties | None = None
) -> str:
    if device_props is None:
        device_props = DeviceProperties.create_from_device_str(device_str)
    return repr((device_str, device_props._asdict()))


class HalideInputSpec(typing.NamedTuple):
    ctype: str
    name: str
    shape: list[str] | None = None
    stride: list[str] | None = None
    offset: str | None = None
    alias_of: str | None = None

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
    scheduler: str | None = None
    scheduler_flags: dict[str, int | str] | None = None
    cuda_device: int | None = None

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
