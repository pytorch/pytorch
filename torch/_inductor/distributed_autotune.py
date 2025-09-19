from __future__ import annotations

import contextlib
import itertools
from typing import Any, Optional, TYPE_CHECKING, Union
from typing_extensions import override
from unittest.mock import patch

import sympy

import torch._logging
import torch.fx
from torch._dynamo.distributed import get_compile_pg
from torch.utils._ordered_set import OrderedSet

from . import config, select_algorithm
from .ir import (
    Buffer,
    ChoiceCaller,
    Layout,
    MultiTemplateBuffer,
    OperationBuffer,
    ShapeAsConstantBuffer,
    StorageBox,
    TensorBox,
)
from .kernel_inputs import MMKernelInputs
from .scheduler import Dep, replace_operation_buffer, SchedulerNode
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


_DISTRIBUTED_AUTOTUNE_INDEX = "distributed_autotune_index"
_autotuned_index: Optional[int] = None


def schedule(scheduler: torch._inductor.scheduler.Scheduler) -> None:
    """
    Finish the distributed autotuning by propagating the autotuning results
    between the ranks and then replacing the placeholder with the real Buffer.
    """
    assert config.distributed_autotune
    autotune_results = _autotune_local(scheduler)
    choices_by_index = _sync(autotune_results)
    _autotune_remote(scheduler, choices_by_index)


@contextlib.contextmanager
def graph_context() -> Generator[None, None, None]:
    """
    Wrapped around processing a graph, sets up figuring out which ranks tune
    which shapes.
    """
    global _autotuned_index
    assert _autotuned_index is None
    _autotuned_index = 0
    try:
        yield
    finally:
        _autotuned_index = None


def maybe_autotune_remote(
    name: str, inputs: list[Buffer], layout: Layout
) -> Optional[TensorBox | ShapeAsConstantBuffer]:
    """
    Used by an op (like `mm`) to determine if the op should be autotuned
    locally (returns None) or remotely (returns a placeholder Buffer).
    """
    if not config.distributed_autotune:
        return None

    if not (compile_pg := get_compile_pg()):
        return None

    global _autotuned_index
    assert _autotuned_index is not None
    index = _autotuned_index
    _autotuned_index += 1

    V.current_node.meta[_DISTRIBUTED_AUTOTUNE_INDEX] = index
    if index % compile_pg.size() == compile_pg.rank():
        return None

    return torch._inductor.ir.TensorBox.create(
        _DistributedAutotuneBuffer(name, inputs, layout)
    )


class _DistributedAutotuneBuffer(MultiTemplateBuffer):
    # Name of the kernel being autotuned.
    _kernel_name: str

    def __init__(
        self,
        kernel_name: str,
        inputs: list[Buffer],
        layout: Layout,
    ) -> None:
        super().__init__(
            layout,
            inputs,
            choice_timings_fn=self._dummy_choice_timings,
            unfiltered_choices=[],
            allowed_prologue_inps=OrderedSet({}),
        )

        self._kernel_name = kernel_name

    def _dummy_choice_timings(
        self, _hint_override: Optional[int]
    ) -> dict[ChoiceCaller, float]:
        raise NotImplementedError

    def autotune(self, choice: _DistributedChoice) -> TensorBox:
        from .select_algorithm import autotune_select_algorithm

        with patch.object(V.graph, "scheduler", None):
            # Original inputs with StorageBox
            kernel_inputs = MMKernelInputs([*self.original_inputs])

            choices: list[Any] = []
            choice.maybe_append_choice(
                choices,
                input_nodes=kernel_inputs.nodes(),
                layout=self.layout,
            )
            buffer = autotune_select_algorithm(
                self._kernel_name,
                choices,
                kernel_inputs.nodes(),
                self.layout,
            )
            assert isinstance(buffer, TensorBox)
            return buffer


# Can we make this async?
def _sync(autotune_results: list[_DistributedChoice]) -> Sequence[_DistributedChoice]:
    if not (compile_pg := get_compile_pg()):
        return ()

    # Perform allgather
    # We should spin this up when we are done autotuning
    # Currently we do autotuning synchronously and
    # For each sent over result {mm1024,1024,...: (128, 128, 64, 5, 8)}
    # do maybe_append_choice
    # "[('cuda', 'torch.bfloat16', 1024, 1024, 1024, 1, 0), ('cuda', 'torch.bfloat16', 1024, 2048, 2048, 1, 0)]"
    all_states: list[list[_DistributedChoice]] = [None] * compile_pg.size()  # type: ignore[list-item]
    torch.distributed.all_gather_object(all_states, autotune_results, group=compile_pg)

    node_count = sum(len(x) for x in all_states)
    choices_by_index: list[_DistributedChoice] = [None] * node_count  # type: ignore[list-item]

    # Technically we could figure this out via "unzipping" but it's safer to do it this way.
    for i, other_results in enumerate(all_states):
        assert other_results is not None
        for choice in other_results:
            assert isinstance(choice, _DistributedChoice)
            choices_by_index[choice.index] = choice

    assert all(x is not None for x in choices_by_index)
    return choices_by_index


class _DistributedChoice:
    """
    This is a serializer for the autotune choice.
    """

    # TODO: Maybe there's a better way to do this?

    index: int

    def __init__(self, index: int) -> None:
        self.index = index

    def maybe_append_choice(self, choices: list[Any], **kwargs: Any) -> None:
        raise NotImplementedError

    @staticmethod
    def serialize_choice(index: int, choice: ChoiceCaller) -> _DistributedChoice:
        # We need a better way to do this
        if isinstance(choice, select_algorithm.ExternKernelCaller):
            return _ExternKernelDistributedChoice(index, choice)
        elif isinstance(choice, select_algorithm.TritonTemplateCaller):
            return _TritonTemplateDistributedChoice(index, choice)
        else:
            raise RuntimeError(f"TODO: {type(choice)}")


class _ExternKernelDistributedChoice(_DistributedChoice):
    def __init__(
        self, index: int, choice: torch._inductor.select_algorithm.ExternKernelCaller
    ) -> None:
        super().__init__(index)
        self.name = choice.choice.name

    @override
    def maybe_append_choice(self, choices: list[Any], **kwargs: Any) -> None:
        if self.name == "mm":
            torch._inductor.kernel.mm.aten_mm.maybe_append_choice(choices, **kwargs)
        else:
            raise RuntimeError(f"UNIMPLEMENTED kernel {self.name}")


class _TritonTemplateDistributedChoice(_DistributedChoice):
    def __init__(
        self, index: int, choice: torch._inductor.select_algorithm.TritonTemplateCaller
    ) -> None:
        super().__init__(index)
        self.kwargs: dict[str, Union[int, str, bool]] = {}
        cfgs = choice.description.split(",")
        for cfg in cfgs:
            key, val = cfg.split("=")
            key, val = key.strip(), val.strip()
            if val == "True":
                self.kwargs[key] = True
            elif val == "False":
                self.kwargs[key] = False
            elif val.isdigit():
                self.kwargs[key] = int(val)
            else:
                self.kwargs[key] = val.replace("'", "")
        k = choice.input_nodes[0].get_size()[1]
        self.kwargs["EVEN_K"] = (
            sympy.gcd(k, self.kwargs["BLOCK_K"]) == self.kwargs["BLOCK_K"]
        )

    @override
    def maybe_append_choice(self, choices: list[Any], **kwargs: Any) -> None:
        from .kernel.mm import mm_template

        mm_template.maybe_append_choice(choices, **kwargs, **self.kwargs)


def _autotune_local(
    scheduler: torch._inductor.scheduler.Scheduler,
) -> list[_DistributedChoice]:
    autotune_results: list[_DistributedChoice] = []

    for node in scheduler.nodes:
        if not isinstance(node, SchedulerNode):
            continue
        if not isinstance(node.node, MultiTemplateBuffer):
            continue
        if isinstance(node.node, _DistributedAutotuneBuffer):
            continue

        # We force autotuning here
        # Still takes advantage of async precompile
        # We need all the configs before fusion
        multi_node = node.node
        assert multi_node is not None
        min_choice, _ = multi_node.get_min_choice()
        assert node.node.origin_node is not None
        index = node.node.origin_node.meta[_DISTRIBUTED_AUTOTUNE_INDEX]

        choice = _DistributedChoice.serialize_choice(index, min_choice)
        autotune_results.append(choice)

    return autotune_results


def _autotune_remote(
    scheduler: torch._inductor.scheduler.Scheduler,
    choices_by_index: Sequence[_DistributedChoice],
) -> None:
    for i, node in enumerate(scheduler.nodes):
        if isinstance(node, SchedulerNode) and isinstance(
            node.node, _DistributedAutotuneBuffer
        ):
            assert node.node.origin_node is not None
            index = node.node.origin_node.meta[_DISTRIBUTED_AUTOTUNE_INDEX]
            replacement_buf = node.node.autotune(choices_by_index[index])

            out_storage = replacement_buf.data
            assert isinstance(out_storage, StorageBox)
            out_buffer = out_storage.data
            assert isinstance(out_buffer, OperationBuffer)
            assert node.node.layout == out_buffer.layout
            replace_operation_buffer(node.node, out_buffer)
            new_scheduler_node = scheduler.create_scheduler_node(out_buffer)

            scheduler.nodes[i] = new_scheduler_node
            scheduler.name_to_node[node.get_name()] = new_scheduler_node
            scheduler.name_to_fused_node[node.get_name()] = new_scheduler_node

            # We need to reflect the mutation renames that were recorded in the original node
            mutation_renames = {}
            for dep in itertools.chain(node.read_writes.reads, node.unmet_dependencies):
                if real_name := scheduler.mutation_real_name.get(dep.name, None):
                    mutation_renames[real_name] = dep.name

            def rename_deps(deps: OrderedSet[Dep]) -> OrderedSet[Dep]:
                return OrderedSet(dep.rename(mutation_renames) for dep in deps)

            new_scheduler_node.unmet_dependencies = rename_deps(
                new_scheduler_node.unmet_dependencies
            )
            new_scheduler_node.read_writes.reads = rename_deps(
                new_scheduler_node.read_writes.reads
            )

            for new_out, old_out in zip(
                new_scheduler_node.get_outputs(), node.get_outputs()
            ):
                scheduler.name_to_buf[old_out.get_name()] = new_out
                new_out.users = old_out.users

            new_scheduler_node.min_order = node.min_order
            new_scheduler_node.max_order = node.max_order
            new_scheduler_node.last_usage = node.last_usage
