from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional, TYPE_CHECKING
from typing_extensions import Self

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.exc import CUDACompileError
from torch._inductor.utils import is_gpu


if TYPE_CHECKING:
    import sympy

    from collections.abc import Sequence

    from torch._prims_common import ShapeType, StrideType


log = logging.getLogger(__name__)


def generate_example_value(
    size: ShapeType,
    stride: StrideType,
    device: torch.device,
    dtype: torch.dtype,
    extra_size: int,
    allocation_size: Optional[Sequence[sympy.Expr]] = None,
) -> torch.Tensor:
    # preserve rng states to avoid the rand_strided call below changes
    # the rng states for the real model code.
    with preserve_rng_state():
        if allocation_size is None or allocation_size == size:
            return rand_strided(
                size,
                stride,
                device=device,
                dtype=dtype,
                extra_size=extra_size,
            )
        else:
            return rand_strided(
                allocation_size,
                stride,
                device=device,
                dtype=dtype,
                extra_size=extra_size,
            ).as_strided(size, stride)


@dataclasses.dataclass
class BenchmarkTensors:
    """Represents a set of inputs and outputs for autotuning with a template"""

    input_tensors: list[torch.Tensor]
    output_tensor: torch.Tensor

    def unpack(self) -> tuple[list[torch.Tensor], torch.Tensor]:
        return self.input_tensors, self.output_tensor


@dataclasses.dataclass
class AutotuneArgs:
    """During autotuning, we need to pass the same inputs to all choices.
    Note:
        Since we typically have a mix of external choices and triton choices, we create
        two lists of inputs for the same underlying buffers:
        - External inputs (for aten kernels): Include offset for sliced tensors
        - Triton inputs: Use base pointer for sliced tensors, without offset
    """

    triton: BenchmarkTensors
    extern: BenchmarkTensors
    expected: Optional[torch.Tensor] = None

    def get_benchmark_tensors(self, extern: bool = False) -> BenchmarkTensors:
        """Returns the inputs and output tensors for a given choice."""
        bench_tensors = self.extern if extern else self.triton
        return bench_tensors

    @classmethod
    def from_choice_args(
        cls,
        example_inputs: list[torch.Tensor],
        example_inputs_extern: list[torch.Tensor],
        out: torch.Tensor,
        out_extern: torch.Tensor,
        expected: Optional[torch.Tensor] = None,
    ) -> Self:
        """Factory method to create AutotuneInputs from separate inputs/outputs"""
        return cls(
            triton=BenchmarkTensors(example_inputs, out),
            extern=BenchmarkTensors(example_inputs_extern, out_extern),
            expected=expected,
        )

    def verify(self, **kwargs: Any) -> None:
        """Verify the correctness of the benchmarking results"""

        torch.testing.assert_close(self.extern.output_tensor, self.expected, **kwargs)


class Benchmarkable:
    """
    Interface for anything that can be benchmarked.
    """

    def is_extern(self) -> bool:
        return False

    def benchmark(self, *args: torch.Tensor, out: Optional[torch.Tensor]) -> float:
        raise NotImplementedError


def benchmark_one_choice(
    choice: Benchmarkable,
    autotune_args: AutotuneArgs,
    verify: Optional[dict[str, Any]],
) -> float:
    is_extern = choice.is_extern()
    benchmark_tensors = autotune_args.get_benchmark_tensors(is_extern)
    inpts, output = benchmark_tensors.unpack()
    output.zero_()
    result = choice.benchmark(*inpts, out=output)
    device_type = next(
        (tensor.device.type for tensor in inpts if is_gpu(tensor.device.type)),
        "cuda",
    )
    device_interface = get_interface_for_device(device_type)
    if device_interface.is_available():
        device_interface.synchronize()  # shake out any CUDA errors

    if verify and autotune_args.expected is not None:
        autotune_args.verify(**verify)
    return result


def benchmark_choices(
    choices: list[Benchmarkable],
    autotune_args: AutotuneArgs,
    verify: Optional[dict[str, Any]],
) -> dict[Benchmarkable, float]:
    timings = {}
    for choice in choices:
        try:
            timing = benchmark_one_choice(choice, autotune_args, verify)
        except CUDACompileError as e:
            log.error(
                "CUDA compilation error during autotuning: \n%s. \nIgnoring this choice.",
                str(e),
            )
            timing = float("inf")
        except NotImplementedError as e:
            log.warning("Not yet implemented: %s", e)
            timing = float("inf")
        except RuntimeError as e:
            msg = str(e)
            if "invalid argument" in msg:
                msg += (
                    "\n\nThis may mean this GPU is too small for max_autotune mode.\n\n"
                )
            else:
                if "illegal memory access" in msg:
                    msg += "\n\nEither error in template or triton bug.\n"
            log.error(
                "Runtime error during autotuning: \n%s. \nIgnoring this choice.",
                msg,
            )
            timing = float("inf")
        except AssertionError as e:
            raise AssertionError(  # noqa: B904
                f"Incorrect result from choice {choice}\n\n{e}"
            )
        except Exception as e:
            try:
                from triton.runtime.autotuner import OutOfResources

                if isinstance(e, OutOfResources):
                    log.warning(e)
                    timing = float("inf")
                else:
                    raise e
            except ImportError:
                raise e from None

        timings[choice] = timing

    return timings
