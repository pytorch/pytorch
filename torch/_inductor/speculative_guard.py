from __future__ import annotations

import dataclasses
import operator
import threading
import weakref
from typing import TYPE_CHECKING, TypeGuard

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.source import (
    AttrSource,
    DictGetItemSource,
    LocalSource,
    NNModuleSource,
    UnspecializedParamBufferSource,
)
from torch._guards import Source

from .codegen.common import custom_backend_passes


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch._functorch._aot_autograd.schemas import ViewAndMutationMeta

    from .graph import GraphLowering


@dataclasses.dataclass(frozen=True)
class _TensorInputSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    requires_grad: bool
    is_conj: bool
    is_neg: bool

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> _TensorInputSpec | None:
        if tensor.layout is not torch.strided:
            return None
        try:
            return cls(
                tuple(int(dim) for dim in tensor.shape),
                tuple(int(stride) for stride in tensor.stride()),
                int(tensor.storage_offset()),
                tensor.dtype,
                tensor.device,
                tensor.layout,
                tensor.requires_grad,
                tensor.is_conj(),
                tensor.is_neg(),
            )
        except (TypeError, ValueError):
            return None

    def matches(self, value: object) -> TypeGuard[torch.Tensor]:
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            return False
        try:
            return (
                tuple(value.shape) == self.shape
                and value.dtype is self.dtype
                and value.device == self.device
                and value.layout is self.layout
                and value.stride() == self.stride
                and value.storage_offset() == self.storage_offset
                and value.requires_grad is self.requires_grad
                and value.is_conj() is self.is_conj
                and value.is_neg() is self.is_neg
            )
        except (RuntimeError, TypeError, ValueError):
            return False


class _SpeculationState(threading.local):
    def __init__(self) -> None:
        self.ticket: _InductorSpeculationTicket | None = None


class _InductorSpeculationTicket:
    def __init__(
        self,
        descriptor: _InductorSpeculationDescriptor,
        inputs: tuple[torch.Tensor, ...],
        output: object,
        event: torch.cuda.Event,
    ) -> None:
        self.descriptor = descriptor
        self.inputs = inputs
        self.output = output
        self.event = event
        self.state = "launched"

    def commit_to_cached_code(self) -> None:
        self.descriptor.commit(self)

    def finish(self) -> None:
        self.descriptor.finish(self)

    def abort(self) -> None:
        self.descriptor.abort(self)


class _InductorSpeculationDescriptor:
    def __init__(
        self,
        compiled_fn: Callable[..., object],
        input_names: tuple[str | None, ...],
        input_specs: tuple[_TensorInputSpec, ...],
        device_index: int,
    ) -> None:
        self.compiled_fn = compiled_fn
        self.input_names = input_names
        self.input_specs = input_specs
        self.device_index = device_index
        self.state = _SpeculationState()
        self.static_input_refs: (
            tuple[weakref.ReferenceType[torch.Tensor] | None, ...] | None
        ) = None
        self.static_input_versions: tuple[int | None, ...] | None = None

    def launch(
        self, frame_locals: dict[str, object]
    ) -> _InductorSpeculationTicket | None:
        if self.state.ticket is not None:
            return None

        if self.static_input_refs is None or self.static_input_versions is None:
            return None

        inputs: list[torch.Tensor] = []
        for name, spec, static_input_ref, version in zip(
            self.input_names,
            self.input_specs,
            self.static_input_refs,
            self.static_input_versions,
        ):
            if name is None:
                if static_input_ref is None:
                    return None
                static_input = static_input_ref()
                if static_input is None:
                    return None
                try:
                    if (
                        static_input._version != version
                        or not spec.matches(static_input)
                    ):
                        return None
                except RuntimeError:
                    return None
                value = static_input
            else:
                try:
                    value = frame_locals[name]
                except KeyError:
                    return None
                if not spec.matches(value):
                    return None
            inputs.append(value)

        input_tuple = tuple(inputs)
        try:
            event = torch.cuda.Event()
            output = self.compiled_fn(*input_tuple)
            event.record(torch.cuda.current_stream(self.device_index))
        except Exception:
            torch.cuda.synchronize(self.device_index)
            return None
        return _InductorSpeculationTicket(self, input_tuple, output, event)

    def commit(self, ticket: _InductorSpeculationTicket) -> None:
        if ticket.descriptor is not self or ticket.state != "launched":
            raise RuntimeError("invalid Inductor speculation ticket commit")
        if self.state.ticket is not None:
            raise RuntimeError("overlapping Inductor speculation tickets")
        torch.cuda.current_stream(self.device_index).wait_event(ticket.event)
        ticket.state = "committed"
        self.state.ticket = ticket

    def call(self, *args: object, **kwargs: object) -> object:
        ticket = self.state.ticket
        if ticket is None:
            output = self.compiled_fn(*args, **kwargs)
            if not kwargs:
                self._bind_static_inputs(args)
            return output
        if kwargs or len(args) != len(ticket.inputs) or any(
            actual is not expected for actual, expected in zip(args, ticket.inputs)
        ):
            try:
                ticket.event.synchronize()
            finally:
                self.state.ticket = None
                ticket.state = "consumed"
                ticket.output = None
                ticket.inputs = ()
            output = self.compiled_fn(*args, **kwargs)
            if not kwargs:
                self._bind_static_inputs(args)
            return output

        self.state.ticket = None
        ticket.state = "consumed"
        output = ticket.output
        ticket.output = None
        ticket.inputs = ()
        return output

    def _bind_static_inputs(self, args: tuple[object, ...]) -> None:
        if len(args) != len(self.input_names):
            return
        static_input_refs = []
        versions = []
        try:
            for name, value in zip(self.input_names, args):
                if name is None:
                    if type(value) not in (torch.Tensor, torch.nn.Parameter):
                        return
                    static_input_refs.append(weakref.ref(value))
                    versions.append(value._version)
                else:
                    static_input_refs.append(None)
                    versions.append(None)
        except RuntimeError:
            return
        self.static_input_refs = tuple(static_input_refs)
        self.static_input_versions = tuple(versions)

    def finish(self, ticket: _InductorSpeculationTicket) -> None:
        if ticket.descriptor is not self or ticket.state != "consumed":
            self.abort(ticket)
            raise RuntimeError("cached bytecode did not consume speculative output")
        ticket.state = "finished"

    def abort(self, ticket: _InductorSpeculationTicket) -> None:
        try:
            ticket.event.synchronize()
        finally:
            if self.state.ticket is ticket:
                self.state.ticket = None
            ticket.output = None
            ticket.inputs = ()
            ticket.state = "aborted"


class _SpeculativeGuardCallable:
    _torchdynamo_speculation_descriptor: _InductorSpeculationDescriptor

    def __init__(self, descriptor: _InductorSpeculationDescriptor) -> None:
        self.descriptor = descriptor
        self._torchdynamo_speculation_descriptor = descriptor

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.descriptor.call(*args, **kwargs)


def _is_static_module_tensor_source(source: Source) -> bool:
    found_param_or_buffer = False
    current = source
    while True:
        if type(current) is LocalSource:
            return found_param_or_buffer
        if isinstance(current, UnspecializedParamBufferSource):
            if current.member not in ("_buffers", "_parameters"):
                return False
            found_param_or_buffer = True
            current = current.base
        elif isinstance(current, NNModuleSource):
            current = current.base
        elif type(current) is AttrSource:
            if current.member != "_modules":
                return False
            current = current.base
        elif type(current) is DictGetItemSource:
            if isinstance(current.index, Source):
                return False
            current = current.base
        else:
            return False


def _is_zero_dropout_attention(node: torch.fx.Node) -> bool:
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is None or "scaled_dot_product" not in schema.name:
        return False
    dropout_index = next(
        (
            index
            for index, argument in enumerate(schema.arguments)
            if argument.name == "dropout_p"
        ),
        None,
    )
    if dropout_index is None:
        return False
    dropout_p = node.kwargs.get("dropout_p")
    if dropout_p is None:
        dropout_p = (
            node.args[dropout_index]
            if len(node.args) > dropout_index
            else schema.arguments[dropout_index].default_value
        )
    return type(dropout_p) in (float, int) and dropout_p == 0.0


def _graph_has_unsafe_effects(gm: torch.fx.GraphModule) -> bool:
    for node in gm.graph.nodes:
        if node.op not in ("placeholder", "get_attr", "output", "call_function"):
            return True
        if node.op != "call_function":
            continue

        target = node.target
        if target is operator.getitem:
            continue
        schema = getattr(target, "_schema", None)
        if schema is None:
            return True
        namespace = schema.name.split("::", 1)[0]
        if namespace not in ("aten", "prims", "inductor"):
            return True
        unsafe_tags = (
            torch.Tag.cudagraph_unsafe,
            torch.Tag.data_dependent_output,
            torch.Tag.dynamic_output_shape,
            torch.Tag.maybe_aliasing_or_mutating,
        )
        if schema.is_mutable or any(tag in target.tags for tag in unsafe_tags):
            return True
        if (
            torch.Tag.nondeterministic_seeded in target.tags
            and not _is_zero_dropout_attention(node)
        ):
            return True
        if "assert" in schema.name:
            return True
    return False


def _metadata_has_unsafe_effects(metadata: ViewAndMutationMeta | None) -> bool:
    if metadata is None:
        return True
    return bool(
        metadata.tokens
        or metadata.grad_enabled_mutation is not None
        or metadata.num_backward_tokens
        or metadata.num_graphsafe_rng_states
        or any(
            info.mutates_data
            or info.mutates_metadata
            or info.mutation_inductor_storage_resize
            for info in metadata.input_info
        )
    )


def is_speculative_guard_safe(
    gm: torch.fx.GraphModule, metadata: ViewAndMutationMeta | None
) -> bool:
    custom_passes = (
        inductor_config.post_grad_custom_pre_pass,
        inductor_config.post_grad_custom_post_pass,
        inductor_config._pre_fusion_custom_pass,
        inductor_config._post_fusion_custom_pass,
    )
    runtime_instrumentation = (
        inductor_config.generate_intermediate_hooks,
        inductor_config.nan_asserts,
        inductor_config.profile_bandwidth,
        inductor_config.runtime_triton_nan_asserts,
    )
    return (
        not any(custom_passes)
        and not any(runtime_instrumentation)
        and not any(custom_backend_passes.values())
        and not _graph_has_unsafe_effects(gm)
        and not _metadata_has_unsafe_effects(metadata)
    )


def is_scheduler_graph_speculative_guard_safe(graph: GraphLowering) -> bool:
    from .dependencies import MemoryDep

    for fused_node in graph.scheduler.nodes:
        for node in fused_node.get_nodes():
            if node.has_side_effects():
                return False
            if any(
                isinstance(dep, MemoryDep) and dep.is_indirect()
                for dep in (*node.read_writes.reads, *node.read_writes.writes)
            ):
                return False
    return True


def maybe_wrap_speculative_guard_callable(
    compiled_fn: Callable[..., object],
    input_sources: Sequence[object],
    example_inputs: Sequence[object],
    compiled_graph: object | None,
) -> Callable[..., object]:
    if (
        compiled_graph is None
        or not getattr(compiled_graph, "speculative_guard_eval_eligible", False)
        or not getattr(compiled_graph, "fx_kwargs", {}).get("is_inference", False)
        or set(getattr(compiled_graph, "device_types", ())) != {"cuda"}
        or len(getattr(compiled_graph, "device_idxs", ())) != 1
        or getattr(compiled_graph, "mutated_input_idxs", ())
        or getattr(compiled_graph, "cudagraph_info", None) is not None
        or getattr(compiled_graph, "partition_maps", None)
        or len(input_sources) != len(example_inputs)
        or not input_sources
        or not all(isinstance(source, Source) for source in input_sources)
    ):
        return compiled_fn

    specs = []
    for value in example_inputs:
        if not isinstance(value, torch.Tensor):
            return compiled_fn
        spec = _TensorInputSpec.from_tensor(value)
        if spec is None:
            return compiled_fn
        specs.append(spec)

    input_names = []
    for source in input_sources:
        if type(source) is LocalSource:
            input_names.append(source.local_name)
        elif _is_static_module_tensor_source(source):
            input_names.append(None)
        else:
            return compiled_fn

    descriptor = _InductorSpeculationDescriptor(
        compiled_fn,
        tuple(input_names),
        tuple(specs),
        next(iter(getattr(compiled_graph, "device_idxs"))),
    )
    return _SpeculativeGuardCallable(descriptor)
