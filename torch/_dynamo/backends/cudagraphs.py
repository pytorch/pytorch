"""
This module implements CUDA graphs support for TorchDynamo backends.

CUDA graphs allow for capturing and replaying GPU operations, which can significantly
reduce CPU overhead in GPU-accelerated PyTorch models. This module provides:

- CUDA graph creation and management for both forward and backward passes
- Input mutation detection and handling
- Device compatibility checking
- Stack trace management for debugging
- Integration with TorchInductor's cudagraph trees

The backend supports two main modes:
1. cudagraphs: Full CUDA graph support with both forward and backward pass optimization
2. cudagraphs_inner: Lower-level CUDA graph implementation used for benchmarking

Key components:
- CudagraphsBackend: Main backend class for CUDA graph integration
- Mutation detection utilities to ensure graph safety
- Device mapping and compatibility checks
- Stack trace collection for debugging
"""

import functools
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
import torch.fx
from torch._dynamo import config
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import boxed_nop
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    check_multiple_devices_or_any_cpu_nodes,
    format_default_skip_message,
    get_mutation_stack_trace,
    get_placeholder_info,
    log_cudagraph_skip_and_bump_counter,
)
from torch._inductor.utils import (
    BoxedBool,
    count_tangents,
    get_first_incompatible_cudagraph_node,
    num_fw_fixed_arguments,
    output_node,
)
from torch.multiprocessing.reductions import StorageWeakRef

from .registry import register_backend


def find_input_mutations(g: torch.fx.Graph) -> set[int]:
    def meta_fk(meta: dict[str, Any]) -> Any:
        return meta["val"] if "val" in meta else meta["fake_result"]

    inputs = defaultdict(set)
    input_idx = 0
    mutated_inputs = set()
    for n in g.nodes:
        if n.op == "placeholder":
            if isinstance(meta_fk(n.meta), torch.Tensor):
                inputs[StorageWeakRef(meta_fk(n.meta)._typed_storage())].add(input_idx)
            input_idx += 1
        elif n.op == "call_function":
            if not hasattr(n.target, "_schema"):
                continue

            schema = n.target._schema
            for i, arg in enumerate(schema.arguments):
                if i < len(n.args):
                    argument = n.args[i]
                else:
                    if arg.name not in n.kwargs:
                        continue
                    argument = n.kwargs[arg.name]
                mut_arg = False
                if arg.alias_info:
                    if arg.alias_info.is_write:
                        mut_arg = True
                if mut_arg:
                    # TODO: not correct for args that contain tensors in a struct
                    # like list
                    mutated_inputs |= inputs[
                        StorageWeakRef(meta_fk(argument.meta)._typed_storage())
                    ]

        # TODO: error on unrecognized nodes
    return mutated_inputs


def get_device_node_mapping(
    gm: torch.fx.GraphModule,
) -> dict[torch.device, torch.fx.Node]:
    device_node_mapping: dict[torch.device, torch.fx.Node] = {}
    for n in gm.graph.nodes:
        t = n.meta.get("val", None)
        if isinstance(t, torch.Tensor) and t.device not in device_node_mapping:
            device_node_mapping[t.device] = n
    return device_node_mapping


def check_for_mutation_ignore_cuda_graph_managed_tensor(
    aot_model: torch.fx.GraphModule, num_fixed: int
) -> Optional[str]:
    mutation_indices = find_input_mutations(aot_model.graph) - set(range(num_fixed))
    if not mutation_indices:
        return None

    placeholders = get_placeholder_info(aot_model.graph)
    return get_mutation_stack_trace(placeholders, mutation_indices)


def check_for_skip(aot_model: torch.fx.GraphModule, num_fixed: int) -> Optional[str]:
    if not config.cudagraph_backend_support_input_mutation:
        if mut_skip := check_for_mutation_ignore_cuda_graph_managed_tensor(
            aot_model, num_fixed
        ):
            return mut_skip

    if skip := check_multiple_devices_or_any_cpu_nodes(
        get_device_node_mapping(aot_model)
    ):
        return skip

    if node := get_first_incompatible_cudagraph_node(aot_model):
        return format_default_skip_message(f"incompatible op ({node.name})")

    return None


def get_device_index(gm: torch.fx.GraphModule) -> int:
    device = next(iter(get_device_node_mapping(gm)))
    assert device.type == "cuda"
    return device.index


def get_stack_traces(gm: torch.fx.GraphModule) -> list[Optional[str]]:
    output = output_node(gm)
    assert len(output.args) == 1
    args = output.args[0]
    if not hasattr(args, "__iter__"):
        return []
    return [
        (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
        for arg in args  # type: ignore[union-attr]
    ]


def cudagraphs(dynamo_model: torch.fx.GraphModule, dynamo_inputs: Sequence[Any]) -> Any:
    from torch._inductor.cudagraph_trees import cudagraphify_impl

    do_cudagraphs = BoxedBool(True)
    boxed_device_index = BoxedDeviceIndex(None)

    def forward_cudagraphs(
        aot_model: torch.fx.GraphModule,
        aot_inputs: list[Any],
        is_inference: bool = False,
    ) -> Any:
        interp = boxed_nop(aot_model, aot_inputs)
        fixed = num_fw_fixed_arguments(len(dynamo_inputs), len(aot_inputs))
        if skip_msg := check_for_skip(aot_model, fixed):
            BoxedBool.disable(do_cudagraphs)
            log_cudagraph_skip_and_bump_counter(
                f"skipping cudagraphs due to {skip_msg}"
            )
            return interp

        boxed_device_index.set(get_device_index(aot_model))
        out = cudagraphify_impl(
            interp,
            aot_inputs,
            range(fixed),
            device_index=boxed_device_index.value,
            is_backward=False,
            is_inference=False,  # Q: should forward is_inference here?
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholder_info(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        out._boxed_call = True  # type: ignore[attr-defined]
        return out

    def backward_cudagraphs(
        aot_model: torch.fx.GraphModule, aot_inputs: list[Any]
    ) -> Any:
        interp = boxed_nop(aot_model, aot_inputs)
        if not do_cudagraphs:
            return aot_model

        fixed = count_tangents(aot_model)
        if skip_msg := check_for_skip(aot_model, fixed):
            log_cudagraph_skip_and_bump_counter(
                f"skipping cudagraphs due to {skip_msg}"
            )

            # See [Backward Generation Handling]
            device_idx = boxed_device_index.value
            if device_idx is None:
                device_idx = 0  # Default to device 0 if not set
            manager = torch._inductor.cudagraph_trees.get_manager(
                device_idx, create_if_none_exists=False
            )
            assert manager is not None

            def fn(inputs: list[Any]) -> Any:
                # pyrefly: ignore [missing-attribute]
                manager.set_to_running_backward()
                return aot_model(inputs)

            fn._boxed_call = True  # type: ignore[attr-defined]
            return fn

        out = cudagraphify_impl(
            interp,
            aot_inputs,
            range(fixed),
            device_index=get_device_index(aot_model),
            is_backward=True,
            is_inference=False,
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholder_info(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        out._boxed_call = True  # type: ignore[attr-defined]
        return out

    aot_cudagraphs = aot_autograd(
        fw_compiler=forward_cudagraphs,
        bw_compiler=backward_cudagraphs,
        inference_compiler=functools.partial(forward_cudagraphs, is_inference=True),
        keep_inference_input_mutations=torch._dynamo.config.cudagraph_backend_keep_input_mutation,
    )
    return aot_cudagraphs(dynamo_model, dynamo_inputs)


class CudagraphsBackend:
    compiler_name = "cudagraphs"

    @staticmethod
    def reset() -> None:
        from torch._inductor.cudagraph_trees import reset_cudagraph_trees

        reset_cudagraph_trees()

    @staticmethod
    def __call__(model: torch.fx.GraphModule, inputs: Sequence[Any]) -> Any:
        return cudagraphs(model, inputs)


# aot_cudagraphs only applies CUDA graphs to the graph.  It is also helpful
# for debugging and can serve as a perf baseline.
register_backend(name="cudagraphs", compiler_fn=CudagraphsBackend())


def cudagraphs_inner(
    model: Callable[..., Any],
    inputs: Sequence[Any],
    copy_outputs: bool = True,
    copy_inputs: bool = True,
) -> Callable[..., Sequence[Any]]:
    """This isn't registered as a backend, but is used in some benchmarks"""
    assert isinstance(inputs, (list, tuple))
    if copy_inputs:
        # pyrefly: ignore [bad-argument-type]
        static_inputs = [torch.zeros_like(x) for x in inputs]
    else:
        static_inputs = list(inputs)

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs: Any) -> Sequence[Any]:
        assert len(static_inputs) == len(new_inputs)
        if copy_inputs:
            for dst, src in zip(static_inputs, new_inputs):
                dst.copy_(src)
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run
