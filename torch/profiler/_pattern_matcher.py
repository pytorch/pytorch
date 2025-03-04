# mypy: allow-untyped-defs
import json
import math
import os
import re
from typing import Optional

import torch
import torch.utils.benchmark as benchmark
from torch._C._profiler import (
    _EventType,
    _ExtraFields_PyCall,
    _ExtraFields_PyCCall,
    _ExtraFields_TorchOp,
    _ProfilerEvent,
)
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs


class Pattern:
    """
    Base class for all patterns, subclass this class and implement match()
    to define custom patterns.

    In subclass, define description and skip property.
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        self.prof = prof
        self.should_benchmark = should_benchmark
        self.name = "Please specify a name for pattern"
        self.description = "Please specify a description for pattern"
        self.url = ""
        assert prof.profiler is not None and prof.profiler.kineto_results is not None
        self.event_tree = prof.profiler.kineto_results.experimental_event_tree()
        self.tid_root: dict[int, list[_ProfilerEvent]] = {}
        for event in self.event_tree:
            self.tid_root.setdefault(event.start_tid, []).append(event)

    @property
    def skip(self):
        return False

    def report(self, event: _ProfilerEvent):
        msg = (
            f"{self.description}\n[Source Code Location] {source_code_location(event)}"
        )
        return msg

    def eventTreeTraversal(self):
        """
        Traverse the event tree and yield all events.
        Override this method in subclass to customize the traversal.
        """
        yield from traverse_dfs(self.event_tree)

    def summary(self, events: list[_ProfilerEvent]):
        default_summary = f"{self.name}: {len(events)} events matched."
        if self.should_benchmark:
            # If benchmark summary is not empty, use it.
            return (
                self.benchmark_summary(events)
                if hasattr(self, "benchmark")  # type: ignore[attr-defined]
                else default_summary
            )
        return default_summary

    def benchmark_summary(self, events: list[_ProfilerEvent]):
        def format_time(time_ns: int):
            unit_lst = ["ns", "us", "ms"]
            for unit in unit_lst:
                if time_ns < 1000:
                    return f"{time_ns:.2f} {unit}"
                time_ns //= 1000
            return f"{time_ns:.2f} s"

        assert hasattr(self, "benchmark"), "Please implement benchmark()"
        shapes_factor_map = self.benchmark(events)  # type: ignore[attr-defined]
        original_time = sum(event.duration_time_ns for event in events)
        new_time = sum(
            shapes_factor_map[input_shapes(event)] * event.duration_time_ns
            for event in events
        )
        return (
            f"{self.name}: {len(events)} events matched. "
            f"Total Estimated Speedup: {format_time(original_time - new_time)} ({round(original_time / new_time, 2)}X)"
        )

    def match(self, event: _ProfilerEvent):
        """
        Return True if the event matches the pattern.
        This method should be overriden in subclass.
        """
        raise NotImplementedError

    def matched_events(self):
        if self.skip:
            return []
        matched_events = [
            event for event in self.eventTreeTraversal() if self.match(event)
        ]
        return matched_events

    def root_of(self, event: _ProfilerEvent):
        while event.parent:
            event = event.parent
        return event

    def siblings_of(self, event: _ProfilerEvent):
        if event.parent:
            children = event.parent.children
        else:
            children = self.tid_root[event.start_tid]
        index = children.index(event)
        return children[:index], children[index + 1 :]

    def next_of(self, event: _ProfilerEvent):
        _, next_events = self.siblings_of(event)
        return next_events[0] if next_events else None

    def prev_of(self, event: _ProfilerEvent):
        prev_events, _ = self.siblings_of(event)
        return prev_events[-1] if prev_events else None

    def go_up_until(self, event: _ProfilerEvent, predicate):
        if not event:
            return None
        while event.parent and not predicate(event):
            event = event.parent
        return event


# Patterns


class NamePattern(Pattern):
    def __init__(self, prof: profile, name: str, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.description = f"Matched Name Event: {name}"
        self.name = name

    def match(self, event: _ProfilerEvent):
        return re.search(self.name, event.name) is not None


class ExtraCUDACopyPattern(Pattern):
    """
    This pattern identifies if we creates a constant tensor on CPU and immediately moves it to GPU.
    example: torch.zeros((100, 100)).to("cuda")

    Pattern:
    build-in method                 |build-in method
        ...                         |    aten::to
            aten::fill_/aten::zero_ |        aten::_to_copy

    Algorithm:
    We start at node aten::to, go parent events' previous events,
    and check if we have a aten::fill_/aten::zero_ as we keep going down the tree.
    We always select the last child in the children list when we go down the tree.
    If at any step we failed, it is not a match.
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Extra CUDA Copy Pattern"
        self.description = "Filled a CPU tensor and immediately moved it to GPU. Please initialize it on GPU."
        self.url = "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#create-tensors-directly-on-the-target-device"
        self.init_ops = {
            "aten::fill_",
            "aten::zero_",
            "aten::normal_",
            "aten::uniform_",
        }

    @property
    def skip(self):
        return not self.prof.with_stack or not self.prof.record_shapes

    def match(self, event):
        # TODO: We should also check tensor identities
        if event.name != "aten::to":
            return False
        to_event = event
        if not event.children:
            return False
        event = event.children[-1]
        if event.name != "aten::_to_copy":
            return False
        if not event.children:
            return False
        event = event.children[-1]
        if event.name != "aten::copy_":
            return False
        # aten::copy_ should have the first 2 args dtype the same
        dtypes = input_dtypes(event)
        if len(dtypes) < 2:
            return False
        if dtypes[0] is None or dtypes[0] != dtypes[1]:
            return False
        event = to_event
        # Up one level
        event = event.parent
        if event is None:
            return False
        # Check if we have a aten::fill_ in previous leaf
        event = self.prev_of(event)
        if event is None:
            return False
        while event.children:
            event = event.children[-1]
            # aten::zero_ is a special optimzation case where fill_ is not called
            if event.name in self.init_ops:
                return True
        return event.name in self.init_ops
        # TODO: Check if tensor is reused

    def benchmark(self, events: list[_ProfilerEvent]):
        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        for shape in shapes_factor_map:
            size = shape[0]
            to_timer = benchmark.Timer(
                stmt='torch.ones(size).to("cuda")', globals={"size": size}
            )
            de_timer = benchmark.Timer(
                stmt='torch.ones(size, device="cuda")', globals={"size": size}
            )
            to_time = to_timer.timeit(10).mean
            de_time = de_timer.timeit(10).mean
            shapes_factor_map[shape] = de_time / to_time
        return shapes_factor_map


class ForLoopIndexingPattern(Pattern):
    """
    This pattern identifies if we use a for loop to index a tensor that
    can be vectorized.
    example:
    tensor = torch.empty((100, 100))
    for i in range(100):
        tensor[i] = i

    Pattern:
    aten::select | ... | aten::select | ... (Repeat)

    Algorithm:
    We start at node aten::select, and we check if we can find this alternating patterns.
    We also keep a dictionary to avoid duplicate match in the for loop.
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "For Loop Indexing Pattern"
        self.description = "For loop indexing detected. Vectorization recommended."
        self.visited: set[int] = set()

    def eventTreeTraversal(self):
        """
        We need to use BFS traversal order to avoid duplicate match.
        """
        yield from traverse_bfs(self.event_tree)

    def match(self, event: _ProfilerEvent):
        if event.name != "aten::select":
            return False
        if event.id in self.visited:
            return False
        repeat_count = 1
        _, next = self.siblings_of(event)
        if len(next) <= 1:
            return False

        # Custom event list matching
        def same_ops(list1, list2):
            if len(list1) != len(list2):
                return False
            for op1, op2 in zip(list1, list2):
                if op1.name != op2.name:
                    return False
            return True

        # Record the ops between two aten::select
        next_select_idx = index_of_first_match(next, lambda e: e.name == "aten::select")
        if next_select_idx is None:
            return False
        indexing_ops = [event] + next[:next_select_idx]
        next = next[len(indexing_ops) - 1 :]
        for i in range(0, len(next), len(indexing_ops)):
            if same_ops(indexing_ops, next[i : i + len(indexing_ops)]):
                repeat_count += 1
                self.visited.add(next[i].id)
            else:
                break
        return repeat_count >= 10


class FP32MatMulPattern(Pattern):
    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "FP32 MatMul Pattern"
        self.description = (
            "You are currently using GPU that supports TF32. "
            "Please enable TF32 by setting 'torch.backends.cuda.matmul.allow_tf32 = True'"
        )
        self.url = "https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"

    @property
    def skip(self):
        if torch.version.hip is not None:
            has_tf32 = False
        else:
            # Anything less than sm_80 is not Ampere which doesn't support TF32
            has_tf32 = all(int(arch[3:]) >= 80 for arch in torch.cuda.get_arch_list())
        return has_tf32 is False or super().skip or not self.prof.record_shapes

    def match(self, event: _ProfilerEvent):
        # If we saw this pattern once, we don't need to match it again
        if event.tag != _EventType.TorchOp:
            return False
        assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
        if event.name == "aten::mm":
            if event.extra_fields.allow_tf32_cublas is False:
                return True
        return False

    def report(self, event: _ProfilerEvent):
        return self.description

    def benchmark(self, events: list[_ProfilerEvent]):
        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        for shape in shapes_factor_map:
            matrixA = torch.randn(shape[0], device="cuda", dtype=torch.float32)
            matrixB = torch.randn(shape[1], device="cuda", dtype=torch.float32)
            fp32_timer = benchmark.Timer(
                stmt="torch.mm(matrixA, matrixB)",
                globals={"matrixA": matrixA, "matrixB": matrixB},
            )
            tf32_timer = benchmark.Timer(
                stmt="torch.mm(matrixA, matrixB)",
                setup="torch.backends.cuda.matmul.allow_tf32 = True",
                globals={"matrixA": matrixA, "matrixB": matrixB},
            )
            torch.backends.cuda.matmul.allow_tf32 = False
            fp32_time = fp32_timer.timeit(10).mean
            tf32_time = tf32_timer.timeit(10).mean
            shapes_factor_map[shape] = tf32_time / fp32_time
        return shapes_factor_map


class OptimizerSingleTensorPattern(Pattern):
    """
    This pattern identifies if we are using the single-tensor version of an optimizer.
    example:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    By adding foreach=True to enable multi-tensor optimizer, we can gain speedup when
    the kernels are relatively small.

    Pattern:
    XXXXX: _single_tenser_<OPTIMIZER_NAME>

    Algorithm:
    String match
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Optimizer Single Tensor Pattern"
        self.optimizers_with_foreach = ["adam", "sgd", "adamw"]
        self.description = (
            "Deteced optimizer running with single tensor implementation. "
            "Please enable multi tensor implementation by passing 'foreach=True' into optimizer."
        )
        self.url = ""

    def match(self, event: _ProfilerEvent):
        for optimizer in self.optimizers_with_foreach:
            if event.name.endswith(f"_single_tensor_{optimizer}"):
                return True
        return False


class SynchronizedDataLoaderPattern(Pattern):
    """
    This pattern identifies if we are using num_workers=0 in DataLoader.
    example:
    torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    Add num_workers=N to the arguments. N depends on system configuration.

    Pattern:
    dataloader.py(...): __iter__
        dataloader.py(...): _get_iterator
            NOT dataloader.py(...): check_worker_number_rationality

    Algorithm:
    If we don't see check_worker_number_rationality call in the dataloader __iter__,
    It is not an asynchronous dataloader.

    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Synchronized DataLoader Pattern"
        self.description = (
            "Detected DataLoader running with synchronized implementation. "
            "Please enable asynchronous dataloading by setting num_workers > 0 when initializing DataLoader."
        )
        self.url = (
            "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html"
            "#enable-async-data-loading-and-augmentation"
        )

    def match(self, event: _ProfilerEvent):
        def is_dataloader_function(name: str, function_name: str):
            return name.startswith(
                os.path.join("torch", "utils", "data", "dataloader.py")
            ) and name.endswith(function_name)

        # TODO: fixme! Due to lifetime issues of the function name, this field might
        # actually point to an already freed string when the even is a PyCall.
        # Just silently skip this to unblock testing.
        try:
            event.name
        except UnicodeDecodeError:
            return False

        if not is_dataloader_function(event.name, "__iter__"):
            return False
        if not event.children:
            return False
        event = event.children[0]
        if not is_dataloader_function(event.name, "_get_iterator"):
            return False
        if not event.children:
            return False
        event = event.children[0]
        return not is_dataloader_function(event.name, "check_worker_number_rationality")
        # TODO: We should also check if the loader is bottleneck.


class GradNotSetToNonePattern(Pattern):
    """
    This pattern identifies if we are not setting grad to None in zero_grad.
    example:
    optimizer.zero_grad()
    By setting set_to_none=True, we can gain speedup

    Pattern:
    XXXXX: _zero_grad
        NOT aten::zeros
            aten::zero_

    aten::zero_ is called on each parameter in the model.
    We also want to make sure it is not called by aten::zeros.

    Algorithm:
    String match
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Gradient Set To Zero Instead of None Pattern"
        self.description = (
            "Detected gradient set to zero instead of None. "
            "Please add 'set_to_none=True' when calling zero_grad()."
        )
        self.url = (
            "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html"
            "#disable-gradient-calculation-for-validation-or-inference"
        )

    def match(self, event: _ProfilerEvent):
        if not event.name.endswith(": zero_grad"):
            return False
        if not event.children:
            return False

        for sub_event in traverse_dfs(event.children):
            if (
                sub_event.name == "aten::zero_"
                and sub_event.parent.name != "aten::zeros"
            ):
                return True
        # TODO: We should also check if the optimizer's numerical behavior will change.
        return False


class Conv2dBiasFollowedByBatchNorm2dPattern(Pattern):
    """
    This pattern identifies if we are enabling bias in Conv2d which is followed by BatchNorm2d.
    Bias doesn't do anything when followed by batchnorm.
    Pattern:
    nn.Module: Conv2d            | nn.Module: BatchNorm2d
        ...
            aten::conv2d AND dtype of third argument is not null
    The third argument is the bias
    Algorithm:
    String match
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Enabling Bias in Conv2d Followed By BatchNorm Pattern"
        self.description = "Detected bias enabled in Conv2d that is followed by BatchNorm2d. Please set 'bias=False' in Conv2d."
        self.url = (
            "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html"
            "#disable-bias-for-convolutions-directly-followed-by-a-batch-norm"
        )

    @property
    def skip(self):
        return self.prof.record_shapes is False or super().skip

    def match(self, event: _ProfilerEvent):
        if event.name != "aten::conv2d":
            return False
        if len(input_dtypes(event)) < 3 or input_dtypes(event)[2] is None:
            return False
        # This means bias=True
        event = self.go_up_until(
            event, lambda e: e.name.startswith("nn.Module: Conv2d")
        )
        if not event:
            return False
        event = self.next_of(event)
        if not event:
            return False
        return event.name.startswith("nn.Module: BatchNorm2d")


class MatMulDimInFP16Pattern(Pattern):
    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Matrix Multiplication Dimension Not Aligned Pattern"
        self.description = "Detected matmul with dimension not aligned. Please use matmul with aligned dimension."
        self.url = "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-mixed-precision-and-amp"

    @property
    def skip(self):
        return not self.prof.with_stack or not self.prof.record_shapes

    def match(self, event: _ProfilerEvent):
        def mutiple_of(shapes, multiple):
            return all(dim % multiple == 0 for shape in shapes for dim in shape[-2:])

        if event.name not in ("aten::mm", "aten::bmm", "aten::addmm"):
            return False
        if not input_dtypes(event):
            return False
        arg_dtype = input_dtypes(event)[0]
        if arg_dtype in (torch.bfloat16, torch.half) and not mutiple_of(
            input_shapes(event), 8
        ):
            return True
        return False

    def benchmark(self, events: list[_ProfilerEvent]):
        def closest_multiple(shapes, multiple):
            return [multiple * math.ceil(shape / multiple) for shape in shapes]

        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        for shape in shapes_factor_map:
            matrixA = torch.randn(shape[0], device="cuda", dtype=torch.float16)
            matrixB = torch.randn(shape[1], device="cuda", dtype=torch.float16)
            not_aligned_dim_timer = benchmark.Timer(
                stmt="torch.mm(matrixA, matrixB)",
                globals={"matrixA": matrixA, "matrixB": matrixB},
            )
            matrixA = torch.randn(
                closest_multiple(shape[0], 8), device="cuda", dtype=torch.float16
            )
            matrixB = torch.randn(
                closest_multiple(shape[1], 8), device="cuda", dtype=torch.float16
            )
            aligned_dim_timer = benchmark.Timer(
                stmt="torch.mm(matrixA, matrixB)",
                globals={"matrixA": matrixA, "matrixB": matrixB},
            )
            not_aligned_dim_time = not_aligned_dim_timer.timeit(10).mean
            aligned_dim_time = aligned_dim_timer.timeit(10).mean
            shapes_factor_map[shape] = aligned_dim_time / not_aligned_dim_time
        return shapes_factor_map


def source_code_location(event: Optional[_ProfilerEvent]):
    while event:
        if event.tag == _EventType.PyCall or event.tag == _EventType.PyCCall:
            assert isinstance(
                event.extra_fields, (_ExtraFields_PyCall, _ExtraFields_PyCCall)
            )
            if not event.extra_fields.caller.file_name.startswith("torch" + os.sep):
                return f"{event.extra_fields.caller.file_name}:{event.extra_fields.caller.line_number}"
        event = event.parent
    return "No source code location found"


def input_shapes(event: _ProfilerEvent):
    assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
    return tuple(tuple(getattr(i, "sizes", ())) for i in event.extra_fields.inputs)


def input_dtypes(event: _ProfilerEvent):
    assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
    return tuple(getattr(i, "dtype", None) for i in event.extra_fields.inputs)


def report_all_anti_patterns(
    prof,
    should_benchmark: bool = False,
    print_enable: bool = True,
    json_report_dir: Optional[str] = None,
):
    report_dict: dict = {}
    anti_patterns = [
        ExtraCUDACopyPattern(prof, should_benchmark),
        # ForLoopIndexingPattern(prof, should_benchmark),
        FP32MatMulPattern(prof, should_benchmark),
        OptimizerSingleTensorPattern(prof, should_benchmark),
        SynchronizedDataLoaderPattern(prof, should_benchmark),
        GradNotSetToNonePattern(prof, should_benchmark),
        Conv2dBiasFollowedByBatchNorm2dPattern(prof, should_benchmark),
        MatMulDimInFP16Pattern(prof, should_benchmark),
    ]
    reported = set()
    summaries = []
    message_list = [f"{'-' * 40}TorchTidy Report{'-' * 40}"]
    message_list.append("Matched Events:")

    for anti_pattern in anti_patterns:
        matched_events = anti_pattern.matched_events()
        if not matched_events:
            continue
        summaries.append(anti_pattern.summary(matched_events))
        for event in matched_events:
            report_msg = anti_pattern.report(event)
            if report_msg not in reported:
                message_list.append(report_msg)
                reported.add(report_msg)
                src_location, line_no = source_code_location(event).split(":")
                report_dict.setdefault(src_location, []).append(
                    {
                        "line_number": int(line_no),
                        "name": anti_pattern.name,
                        "url": anti_pattern.url,
                        "message": anti_pattern.description,
                    }
                )

    if json_report_dir is not None:
        json_report_path = os.path.join(json_report_dir, "torchtidy_report.json")
        if os.path.exists(json_report_path):
            with open(json_report_path) as f:
                exisiting_report = json.load(f)
                exisiting_report.update(report_dict)
                report_dict = exisiting_report
        with open(json_report_path, "w") as f:
            json.dump(report_dict, f, indent=4)

    message_list.append("Summary:")
    message_list += summaries
    message_list.append(f"{'-' * 40}TorchTidy Report{'-' * 40}")
    if print_enable:
        print("\n".join(message_list))
