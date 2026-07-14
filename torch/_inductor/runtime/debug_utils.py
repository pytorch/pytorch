import functools
import logging
import threading
import weakref

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

local = threading.local()
local.memory_tracker = None


class BufferMemoryTracker:
    """
    Tracks inductor runtime allocations and deallocations to compare against
    expected behavior.
    """

    def __init__(self) -> None:
        self.tensor_tracker: dict[str, torch.storage.UntypedStorage] = (
            weakref.WeakValueDictionary()  # type: ignore[assignment]
        )
        self.died_since_last_step: OrderedSet[str] = OrderedSet()
        self.added_since_last_step: OrderedSet[str] = OrderedSet()
        self.error = (
            torch._inductor.config.test_configs.track_memory_lifecycle == "assert"
        )

    def set_tensor(self, name: str, tensor: torch.Tensor) -> None:
        storage = tensor.untyped_storage()

        self.added_since_last_step.add(name)
        self.tensor_tracker[name] = storage

        def on_tensor_death() -> None:
            self.died_since_last_step.add(name)

        weakref.finalize(storage, on_tensor_death)

    def advance_step(self) -> None:
        self.died_since_last_step.clear()
        self.added_since_last_step.clear()

    def log_or_raise(self, msg: str) -> None:
        if self.error:
            raise RuntimeError(msg)
        else:
            log.info(msg)

    def check_step_delta(
        self,
        expected_allocated: list[str],
        expected_freed: list[str],
        is_final_step: bool,
    ) -> None:
        """Check only the delta changes since last step"""

        # Check expected deaths - we dont currently distinguish between nodes which die in last step
        # and are returned as outputs, so skip if final_step.
        if not is_final_step:
            missing_deaths = OrderedSet(expected_freed) - self.died_since_last_step
            if missing_deaths:
                self.log_or_raise(
                    f"Expected tensors to die but still alive: {missing_deaths}"
                )

        # Check for unexpected deaths
        unexpected_deaths = self.died_since_last_step - OrderedSet(expected_freed)
        if unexpected_deaths:
            self.log_or_raise(f"Unexpected tensor deaths: {unexpected_deaths}")

        # Check newly alive tensors - separate messages like deaths
        actual_allocated = self.added_since_last_step
        expected_allocated_set = OrderedSet(expected_allocated)

        extra_alive = actual_allocated - expected_allocated_set
        if extra_alive:
            self.log_or_raise(f"Unexpected allocated tensors: {extra_alive}")

        missing_alive = expected_allocated_set - actual_allocated
        if missing_alive:
            self.log_or_raise(
                f"Expected allocated tensors but missing: {missing_alive}"
            )

        # Reset for next step
        self.advance_step()

        if is_final_step:
            local.memory_tracker = None


def get_mem_tracker() -> BufferMemoryTracker:
    if local.memory_tracker is None:
        local.memory_tracker = BufferMemoryTracker()
    return local.memory_tracker


def track_tensor(tensor: torch.Tensor, name: str) -> None:
    get_mem_tracker().set_tensor(name, tensor)


def tracked_empty_strided(
    size: list[int],
    stride: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    o = torch.empty_strided(size, stride, dtype=dtype, device=device)
    track_tensor(o, name)
    return o


def check_memory_step(
    allocated: list[str], freed: list[str], is_final_step: bool = False
) -> None:
    tracker = get_mem_tracker()
    tracker.check_step_delta(allocated, freed, is_final_step)


@functools.lru_cache(None)
def register_check_mem_op() -> None:
    lib = torch.library.Library("_inductor_debug", "FRAGMENT")  # noqa: TOR901
    lib.define(
        "check_memory_step(str[] allocated, str[] freed, bool is_final_step) -> ()"
    )
    lib.impl("check_memory_step", check_memory_step, "BackendSelect")
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

    _register_effectful_op(
        torch.ops._inductor_debug.check_memory_step.default,
        _EffectType.ORDERED,
    )
