from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, TYPE_CHECKING, Union
from unittest.mock import patch

import sympy

import torch._logging
import torch.distributed as dist
import torch.fx
from torch.utils._ordered_set import OrderedSet
from . import config, select_algorithm
from .ir import (
    Buffer,
    ChoiceCaller,
    Layout,
    MultiTemplateBuffer,
    OperationBuffer,
    StorageBox,
    TensorBox,
)
from .kernel_inputs import KernelInputs, MMKernelInputs
from .scheduler import SchedulerNode
from .virtualized import NullHandler, V


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


_DISTRIBUTED_AUTOTUNE_KEY = "distributed_autotune"

_AUTOTUNE_PG: dist.ProcessGroup | None = None


@dataclasses.dataclass
class _DistributedAutotuneState:
    """
    State used to track autotuning during a graph_context()
    """

    # This is the next operator index. Used to figure out which rank should do
    # the autotuning.
    autotuned_index: int = 0

    # For debugging - used to make sure that we autotune the same number of
    # local operators that we expected to.
    autotuned_local_count: int = 0


@dataclasses.dataclass
class _DistributedAutotuneInfo:
    index: int
    local: bool


def get_autotune_pg() -> dist.ProcessGroup | None:
    if dist.is_available() and dist.is_initialized():
        global _AUTOTUNE_PG
        if _AUTOTUNE_PG is None:
            _AUTOTUNE_PG = dist.distributed_c10d._new_group_with_tag(
                pg_tag="pt2_distributed_autotune_pg"
            )
        return _AUTOTUNE_PG

    return None


def schedule(scheduler: torch._inductor.scheduler.Scheduler) -> None:
    """
    Finish the distributed autotuning by propagating the autotuning results
    between the ranks and then replacing the placeholder with the real Buffer.
    """
    assert config.distributed_max_autotune_gemm
    autotune_results = _autotune_local_nodes(scheduler)
    choices_by_index = _sync(autotune_results)
    _autotune_remote_nodes(scheduler, choices_by_index)


@contextlib.contextmanager
def graph_context() -> Generator[None, None, None]:
    """
    Wrapped around processing a graph, sets up figuring out which ranks tune
    which shapes.
    """
    assert not isinstance(
        V.get_distributed_autotune_state(check_poisoned=False),  # type: ignore[call-arg]
        _DistributedAutotuneState,
    )
    V.set_distributed_autotune_state(_DistributedAutotuneState())
    try:
        yield
    finally:
        V.set_distributed_autotune_state(NullHandler())


def maybe_autotune_remote(
    name: str, choices: list[ChoiceCaller], inputs: list[Buffer], layout: Layout
) -> TensorBox | None:
    """
    Used by an op (like `mm`) to determine if the op should be autotuned
    locally (returns None) or remotely (returns a placeholder Buffer).
    """
    if not config.distributed_max_autotune_gemm:
        return None

    if not (autotune_pg := get_autotune_pg()):
        return None

    if len(choices) <= 1:
        return None

    state = V.distributed_autotune_state
    index = state.autotuned_index
    state.autotuned_index += 1
    local = index % autotune_pg.size() == autotune_pg.rank()

    V.current_node.meta[_DISTRIBUTED_AUTOTUNE_KEY] = _DistributedAutotuneInfo(
        index, local
    )
    if local:
        state.autotuned_local_count += 1
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
        self, _hint_override: int | None
    ) -> dict[ChoiceCaller, float]:
        # This should never get called. It means that a remote autotune was
        # scheduled but never filled in.
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

    autotune_pg = get_autotune_pg()
    assert autotune_pg

    # Perform allgather
    all_states: list[list[_SerializedChoice]] = [None] * autotune_pg.size()  # type: ignore[list-item]
    torch.distributed.all_gather_object(all_states, autotune_results, group=autotune_pg)

    node_count = sum(len(x) for x in all_states)
    # It's faster to briefly lie about the type than to unzip the results and append.
    choices_by_index: list[_SerializedChoice] = [None] * node_count  # type: ignore[list-item]

    check_count = 0
    for other_results in all_states:
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

    def get_choice(self, layout: Layout, inputs: KernelInputs) -> ChoiceCaller | None:
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
        if not description:
            return {}

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

        if (inner_node := node.node) is None:
            continue

        if isinstance(inner_node, _DistributedAutotuneBuffer):
            # This is marked for remote autotuning.
            continue

        if not isinstance(inner_node, MultiTemplateBuffer):
            continue

        if (origin_node := inner_node.origin_node) is None:
            continue

        if (meta := origin_node.meta) is None:
            continue

        info = meta.get(_DISTRIBUTED_AUTOTUNE_KEY)
        if info is None:
            continue

        assert info.local

        # We force autotuning here
        # Still takes advantage of async precompile
        # We need all the configs before fusion
        min_choice, _ = inner_node.get_min_choice()

        choice = _SerializedChoice(info.index, min_choice)
        autotune_results.append(choice)

    state = V.distributed_autotune_state
    assert len(autotune_results) == state.autotuned_local_count, (
        f"incorrect local autotuned nodes found ({len(autotune_results)} != {state.autotuned_local_count})"
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
            (dist_node := node.node), _DistributedAutotuneBuffer
        ):
            assert dist_node.origin_node is not None
            info = dist_node.origin_node.meta[_DISTRIBUTED_AUTOTUNE_KEY]
            out_tensorbox = dist_node.autotune(choices_by_index[info.index])

            out_storage = out_tensorbox.data
            assert isinstance(out_storage, StorageBox)
            out_buffer = out_storage.data
            assert isinstance(out_buffer, OperationBuffer)

            assert out_buffer.layout == dist_node.layout

            scheduler._replace_node(out_buffer, dist_node, i, node)
