from __future__ import annotations

import contextlib
import itertools
from typing import Any, Optional, TYPE_CHECKING, Union
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
from .kernel_inputs import KernelInputs, MMKernelInputs
from .scheduler import Dep, replace_operation_buffer, SchedulerNode
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


_DISTRIBUTED_AUTOTUNE_INDEX = "distributed_autotune_index"
_autotuned_index: Optional[int] = None
_autotuned_local_count: int = 0


def schedule(scheduler: torch._inductor.scheduler.Scheduler) -> None:
    """
    Finish the distributed autotuning by propagating the autotuning results
    between the ranks and then replacing the placeholder with the real Buffer.
    """
    assert config.distributed_autotune
    autotune_results = _autotune_local_nodes(scheduler)
    choices_by_index = _sync(autotune_results)
    _autotune_remote_nodes(scheduler, choices_by_index)


@contextlib.contextmanager
def graph_context() -> Generator[None, None, None]:
    """
    Wrapped around processing a graph, sets up figuring out which ranks tune
    which shapes.
    """
    global _autotuned_index, _autotuned_local_count
    assert _autotuned_index is None
    _autotuned_index = 0
    _autotuned_local_count = 0
    try:
        yield
    finally:
        _autotuned_index = None
        _autotuned_local_count = -1


def maybe_autotune_remote(
    name: str, choices: list[ChoiceCaller], inputs: list[Buffer], layout: Layout
) -> Optional[TensorBox | ShapeAsConstantBuffer]:
    """
    Used by an op (like `mm`) to determine if the op should be autotuned
    locally (returns None) or remotely (returns a placeholder Buffer).
    """
    if not config.distributed_autotune:
        return None

    if len(choices) == 1:
        return None

    if not (compile_pg := get_compile_pg()):
        return None

    global _autotuned_index, _autotuned_local_count
    assert _autotuned_index is not None
    index = _autotuned_index
    _autotuned_index += 1

    V.current_node.meta[_DISTRIBUTED_AUTOTUNE_INDEX] = index
    if index % compile_pg.size() == compile_pg.rank():
        _autotuned_local_count += 1
        return None

    return torch._inductor.ir.TensorBox.create(
        _DistributedAutotuneBuffer(name, inputs, layout)
    )


class _DistributedAutotuneBuffer(MultiTemplateBuffer):
    """
    A MultiTemplateBuffer which represents a kernel being autotuned on a
    different rank. When `schedule` is called this will be replaced by the
    "real" buffer.
    """

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

    def autotune(self, ser_choice: _SerializedChoice) -> TensorBox:
        """
        Given a _SerializedChoice (autotune results from another rank)
        compute the final TensorBox.
        """

        from .select_algorithm import autotune_select_algorithm

        with patch.object(V.graph, "scheduler", None):
            kernel_inputs = MMKernelInputs([*self.original_inputs])
            assert isinstance(self.layout, Layout)
            choice = ser_choice.get_choice(self.layout, kernel_inputs)
            buffer = autotune_select_algorithm(
                self._kernel_name,
                [choice],
                kernel_inputs.nodes(),
                self.layout,
            )
            assert isinstance(buffer, TensorBox)
            return buffer


# Can we make this async?
def _sync(autotune_results: list[_SerializedChoice]) -> Sequence[_SerializedChoice]:
    """
    Perform the all_gather to collect the autotune results from all the ranks.
    """

    compile_pg = get_compile_pg()
    assert compile_pg

    # Perform allgather
    all_states: list[list[_SerializedChoice]] = [None] * compile_pg.size()  # type: ignore[list-item]
    torch.distributed.all_gather_object(all_states, autotune_results, group=compile_pg)

    node_count = sum(len(x) for x in all_states)
    # It's faster to briefly lie about the type than to unzip the results and append.
    choices_by_index: list[_SerializedChoice] = [None] * node_count  # type: ignore[list-item]

    check_count = 0
    for i, other_results in enumerate(all_states):
        for choice in other_results:
            assert isinstance(choice, _SerializedChoice)
            assert choices_by_index[choice.index] is None
            choices_by_index[choice.index] = choice
            check_count += 1

    assert node_count == check_count, f"count mismatch: {node_count} != {check_count}"
    return choices_by_index


class _SerializedChoice:
    """
    This is a serializer for the autotune choice. KernelTemplateChoice can't
    be serialized directly (the template and inputs prevent this) so we need to
    serialize it by parts and reconstruct later on.
    """

    def __init__(self, index: int, choice: ChoiceCaller) -> None:
        self.index = index
        self.template_uid = _SerializedChoice._template_uid_from_choice(choice)
        self.kwargs = self._compute_kwargs(choice.description)

    def get_choice(
        self, layout: Layout, inputs: KernelInputs
    ) -> Optional[ChoiceCaller]:
        """
        Deserialize the ChoiceCaller and return it.
        """

        template = self._template_from_uid()

        kwargs = {**self.kwargs}
        if "BLOCK_K" in kwargs:
            # TODO: Do we really need to externally compute this value? If it's
            # needed I'm surprised it's not just part of the original template
            # description.
            # This needs the actual 'k' to figure out the value.
            k = inputs.nodes()[0].get_size()[1]
            kwargs["EVEN_K"] = sympy.gcd(k, kwargs["BLOCK_K"]) == kwargs["BLOCK_K"]

        extra_kwargs: dict[str, Any] = {}
        from .kernel_template_choice import (
            DictKernelTemplateParams,
            KernelTemplateChoice,
        )

        params = DictKernelTemplateParams(kwargs)
        ktc = KernelTemplateChoice(template, params, extra_kwargs, layout, inputs)
        return ktc.choice

    @staticmethod
    def _compute_kwargs(description: str) -> dict[str, Union[int, str, bool]]:
        """
        Given a template description turn it into input kwargs.
        """

        # TODO: It seems like it would be better if the template could provide
        # this directly instead of having to parse a string.
        kwargs: dict[str, Union[int, str, bool]] = {}
        for cfg in description.split(","):
            key, val = cfg.split("=", 1)
            key, val = key.strip(), val.strip()
            if val == "True":
                kwargs[key] = True
            elif val == "False":
                kwargs[key] = False
            elif val.isdigit():
                kwargs[key] = int(val)
            else:
                assert val.startswith("'") and val.endswith("'")
                kwargs[key] = val[1:-1]
        return kwargs

    @staticmethod
    def _template_uid_from_choice(choice: ChoiceCaller) -> str:
        """
        Given a ChoiceCaller figure out which template represents it. This
        is reversed by _template_from_uid().
        """

        # We need a better way to do this - right now we need to add each
        # supported template directly.
        if isinstance(choice, select_algorithm.ExternKernelCaller):
            if choice.choice.name == "mm":
                return "torch._inductor.kernel.mm.aten_mm"
            else:
                raise RuntimeError(f"TODO: kernel {choice.choice.name!r}")
        elif isinstance(choice, select_algorithm.TritonTemplateCaller):
            return "torch._inductor.kernel.mm.mm_template"
        else:
            raise RuntimeError(f"TODO: {type(choice)}")

    def _template_from_uid(self) -> Any:
        """
        See _template_uid_from_choice().
        """
        parts = self.template_uid.split(".")
        obj = globals()[parts[0]]
        for k in parts[1:]:
            obj = getattr(obj, k)
        return obj


def _autotune_local_nodes(
    scheduler: torch._inductor.scheduler.Scheduler,
) -> list[_SerializedChoice]:
    """
    Go through the nodes in the scheduler and autotune the kernels which
    should be autotuned by this rank.
    """

    autotune_results: list[_SerializedChoice] = []

    for node in scheduler.nodes:
        if not isinstance(node, SchedulerNode):
            continue
        if isinstance(node.node, _DistributedAutotuneBuffer):
            continue
        if not isinstance(node.node, MultiTemplateBuffer):
            continue

        # We force autotuning here
        # Still takes advantage of async precompile
        # We need all the configs before fusion
        multi_node = node.node
        assert multi_node is not None
        min_choice, _ = multi_node.get_min_choice()
        assert node.node.origin_node is not None
        index = node.node.origin_node.meta[_DISTRIBUTED_AUTOTUNE_INDEX]

        choice = _SerializedChoice(index, min_choice)
        autotune_results.append(choice)

    assert len(autotune_results) == _autotuned_local_count, (
        f"not enough local autotuned nodes found ({len(autotune_results)} != {_autotuned_local_count})"
    )
    return autotune_results


def _autotune_remote_nodes(
    scheduler: torch._inductor.scheduler.Scheduler,
    choices_by_index: Sequence[_SerializedChoice],
) -> None:
    """
    Go through the nodes in the scheduler and autotune the nodes that were
    autotuned on remote ranks.
    """

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
