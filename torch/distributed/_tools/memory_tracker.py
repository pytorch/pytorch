from collections import defaultdict

from typing import (
    Any,
    Callable,
    Dict,
    List,
    no_type_check,
    Sequence,
)

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode

BYTES_PER_MB = 1024 * 1024.0


class MemoryProfileDispatchMode(TorchDispatchMode):
    """
    Run in ``TorchDispatchMode`` to get memory stats at operator level.
    """

    def __init__(self, memory_tracker) -> None:
        self.memory_tracker = memory_tracker

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        rs = func(*args, **kwargs)
        if func == torch.ops.aten.detach.default:
            return rs
        func_name: str = (
            self.memory_tracker._cur_module_name
            + "."
            + func.__name__
            + "_"
            + str(self.memory_tracker._operator_names[func.__name__])
        )
        self.memory_tracker._operator_names[func.__name__] = (
            self.memory_tracker._operator_names[func.__name__] + 1
        )
        self.memory_tracker._record_memory_stats(func_name)

        return rs


class MemoryTracker:
    """
    Collect and plot the memory stats including ``memories_allocated``, ``memories_active``
    and ``memories_reserved`` at operator level.
    It also prints a summary for the top 20 operators that generate the most memories.

    Example usage:

        >>> net.cuda()
        >>> input = input.cuda()

        >>> mem_tracker = MemoryTracker()
        >>> mem_tracker.start_monitor(net)

        >>> net.zero_grad(True)
        >>> loss = net(input)
        >>> if isinstance(loss, dict):
        >>>    loss = loss['out']
        >>> loss.sum().backward()
        >>> net.zero_grad(set_to_none=True)

        >>> mem_tracker.stop()
        >>> mem_tracker.summary()
        >>> mem_tracker.show_traces()
    """

    def __init__(self) -> None:
        torch._C._log_api_usage_once("torch.distributed.memory_tracker")
        self._hooks: List[RemovableHandle] = []
        self._operator_names: Dict[str, int] = defaultdict(int)
        self.memories_allocated: Dict[int, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.memories_active: Dict[int, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.memories_reserved: Dict[int, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._markers: Dict[str, int] = defaultdict(int)
        self._cur_module_name: str = ""
        self._op_index: int = 0

    @no_type_check
    def start_monitor(self, root_module: nn.Module) -> None:
        """
        Register module hooks and entering ``MemoryProfileDispatchMode``, so that
        operator level memory stats can be tracked during module runtime.
        """
        self._clear_state()
        root_module.__setattr__("_memory_tracker_is_root", True)
        for name, m in root_module.named_modules():
            if m is not root_module:
                m.__setattr__("_memory_tracker_is_root", False)
            # fused_proxy_group does not support hooks
            if ".fused_proxy_grouped_embedding_bag" in name:
                continue
            # hook ordering with other hooks added by users is not managed, so
            # the memory stats tracked here may not completely accurate.
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            # it does not work well with jagged tensor somehow, the root cause is not
            # clear and remove it for now as it does not really capture important info.
            # h3 = m.register_backward_hook(self._create_backward_hook(name))
            self._hooks.extend([h1, h2])
        torch.cuda.empty_cache()
        assert getattr(self, "profile_mode", None) is None
        self.profile_mode = MemoryProfileDispatchMode(self)
        self.profile_mode.__enter__()

    @no_type_check
    def stop(self) -> None:
        """
        Remove module hooks and exit ``MemoryProfileDispatchMode`` to stop
        tracking memory stats at operator level.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        assert getattr(self, "profile_mode", None) is not None
        self.profile_mode.__exit__(None, None, None)
        self.profile_mode = None

    @no_type_check
    def summary(self, top: int = 20) -> None:
        """
        Print out the top operators that generate the most memories. The number
        of the top operators can be configured.
        """
        op_diff: Dict[str, float] = defaultdict(float)
        op_name, previous_allocated_memory = self.memories_allocated[0]
        for i in range(1, self._op_index):
            op_name, current_allocated_memory = self.memories_allocated[i]
            op_diff[op_name] = current_allocated_memory - previous_allocated_memory
            previous_allocated_memory = current_allocated_memory

        print("------------------------------------------------")
        print(f"Top {top} ops that generates memory are:")
        for k, v in sorted(op_diff.items(), key=lambda item: item[1], reverse=True)[
            :top
        ]:
            print(f"{k}: {v}MB")
        print("------------------------------------------------")

    @no_type_check
    def show_traces(self) -> None:
        """
        Show the traces of ``memory_allocated``, ``memory_active`` and ``memory_reserved`` at
        operator level and the marker 'fw_bw_boundary' at the boundary of forward pass
        and backward pass.
        """
        import matplotlib.pyplot as plt

        y_1 = [mb for (name, mb) in self.memories_allocated.values()]
        y_2 = [mb for (name, mb) in self.memories_active.values()]
        y_3 = [mb for (name, mb) in self.memories_reserved.values()]
        min_val = min(y_1 + y_2 + y_3)
        max_val = max(y_1 + y_2 + y_3)
        x = list(i for i in range(len(y_1)))
        fig = plt.figure(figsize=(16, 8))
        plt.plot(x, list(y_1), label="memory_allocated")
        plt.plot(x, list(y_2), label="memory_active")
        plt.plot(x, list(y_3), label="memory_reserved")
        plt.xlabel("# Operator Calls")
        plt.ylabel("Memory (MB)")
        for marker_name, marker in self._markers.items():
            if marker_name == "fw_bw_boundary":
                plt.plot(
                    [marker, marker], [min_val, max_val], "r", lw=2, label=marker_name
                )
            else:
                plt.plot(
                    [marker, marker], [min_val, max_val], "k-", lw=2, label=marker_name
                )
        plt.legend()

    def _create_pre_forward_hook(self, name: str) -> Callable:
        """
        The pre_foward_hook is to insert current module name with forward prefix for the operator
        name, also it inserts the marker "fw_start" when the forward pass begins.
        """

        def _pre_forward_hook(module: nn.Module, inputs: Any) -> None:
            self._cur_module_name = f"{name}.forward"
            if (
                hasattr(module, "_memory_tracker_is_root")
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_start")

        return _pre_forward_hook

    def _create_post_forward_hook(self, name: str) -> Callable:
        """
        The post_forward_hook inserts the marker 'fw_bw_boundary' at the boundary
        of forward pass and backward pass.
        """

        def _post_forward_hook(
            module: nn.Module,
            inputs: Sequence[torch.Tensor],
            outputs: Sequence[torch.Tensor],
        ) -> None:
            if (
                hasattr(module, "_memory_tracker_is_root")
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_bw_boundary")

        return _post_forward_hook

    def _create_backward_hook(self, name: str) -> Callable:
        """
        The backward_hook inserts the current module name with backward prefix for the operator name.
        """

        def _backward_hook(
            module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
        ) -> None:
            self._cur_module_name = f"{name}.backward"

        return _backward_hook

    @no_type_check
    def _record_memory_stats(self, fn_name: str) -> None:
        """
        Record current memory allocated, current memory active and current memory reserved.
        The memory stats dict is indexed with ``self._op_index``.
        """
        memory_allocated: float = torch.cuda.memory_allocated() / BYTES_PER_MB
        memory_reserved: float = torch.cuda.memory_reserved() / BYTES_PER_MB
        memory_active: float = (
            torch.cuda.memory_stats().get("active_bytes.all.current", 0) / BYTES_PER_MB
        )
        self.memories_allocated[self._op_index] = (fn_name, memory_allocated)
        self.memories_reserved[self._op_index] = (fn_name, memory_reserved)
        self.memories_active[self._op_index] = (fn_name, memory_active)
        self._op_index += 1

    def _add_marker(self, marker_name: str) -> None:
        """
        Set the marker's x-axis value.
        """
        marker_val = len(self.memories_allocated.values())
        self._markers[marker_name] = marker_val

    def _clear_state(self) -> None:
        """
        Clear states when start_monitor() is called.
        """
        self._operator_names.clear()
        self.memories_allocated.clear()
        self.memories_active.clear()
        self.memories_reserved.clear()
        self._markers.clear()
        self._cur_module_name = ""
        self._op_index = 0
