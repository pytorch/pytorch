# mypy: allow-untyped-defs
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...autotune_process import (
    GPUDeviceBenchmarkMixin,
    TensorMeta,
    TritonBenchmarkRequest,
)
from ...codecache import code_hash
from ...select_algorithm import TritonTemplateCaller
from ...virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable

    from ...ir import Buffer, TensorBox


@dataclass
class _RocmSparseTensorMeta(TensorMeta):
    num_blocks: int | None = None
    causal_role: int | None = None

    def to_tensor(self) -> torch.Tensor:
        tensor = super().to_tensor()
        role = self.causal_role
        if role is not None:
            block_count = self.sizes[-1] if role % 2 == 0 else self.sizes[-2]
            rows = torch.arange(block_count, dtype=self.dtype, device=self.device)
            if role in (8, 10):
                return tensor.fill_(1)
            if role == 12:
                return tensor.copy_(rows.expand(self.sizes))
            if role == 14:
                return tensor.copy_((block_count - 1 - rows).expand(self.sizes))
            if role in (9, 11):
                return tensor.copy_(rows.unsqueeze(-1).expand(self.sizes))
            columns = torch.arange(self.sizes[-1], dtype=self.dtype, device=self.device)
            if role == 13:
                return tensor.copy_(columns.expand(self.sizes))
            if role == 15:
                indices = (rows.unsqueeze(-1) + 1 + columns).clamp(max=block_count - 1)
                return tensor.copy_(indices.expand(self.sizes))
            raise AssertionError(f"Unexpected causal sparse role: {role}")
        if self.num_blocks is not None:
            return tensor.fill_(self.num_blocks)
        return tensor.copy_(
            torch.arange(self.sizes[-1], dtype=self.dtype, device=self.device).expand(
                self.sizes
            )
        )


def prepare_backward_benchmark_requests(
    choices, input_nodes, *, causal=False
) -> dict[int, Callable[[Buffer], torch.Tensor]]:
    """Preserve the Flex sparse input generators in benchmark subprocesses."""
    sparse_meta = {}
    input_gen_fns: dict[int, Callable[[Buffer], torch.Tensor]] = {}
    for index in range(8, 16):
        node = input_nodes[index]
        meta = TensorMeta.from_irnodes(node)
        if not isinstance(meta, TensorMeta):
            raise AssertionError("Expected sparse input tensor metadata")
        num_blocks = (
            V.graph.sizevars.optimization_hint(input_nodes[index + 1].get_size()[-1])
            if index % 2 == 0
            else None
        )
        benchmark_meta = _RocmSparseTensorMeta(
            meta.device,
            meta.dtype,
            meta.sizes,
            meta.strides,
            meta.offset,
            meta.name,
            num_blocks,
            index if causal else None,
        )
        sparse_meta[node.get_name()] = benchmark_meta
        input_gen_fns[index] = lambda _node, meta=benchmark_meta: meta.to_tensor()
    for choice in choices:
        request = choice.bmreq
        request.input_tensor_meta = [
            sparse_meta.get(meta.name, meta) for meta in request.input_tensor_meta
        ]
    return input_gen_fns


def _request_key(request: TritonBenchmarkRequest):
    return (
        request.kernel_name,
        request.module_cache_key,
        tuple(request.extra_args),
        request.num_stages,
        request.num_warps,
        request.num_consumer_groups,
        request.num_buffers_warp_spec,
        request.matrix_instr_nonkdim,
        request.waves_per_eu,
        request.kpack,
        request.workspace_size,
        request.workspace_zero_fill,
    )


class RocmBackwardBenchmarkRequest(GPUDeviceBenchmarkMixin, TritonBenchmarkRequest):
    """Measure both backward phases through one benchmark closure."""

    def __init__(
        self,
        input_tensor_meta: list[TensorMeta],
        dkdv: TritonBenchmarkRequest,
        dq: TritonBenchmarkRequest,
        dq_output_index: int,
    ) -> None:
        input_indices = {
            meta.name: index for index, meta in enumerate(input_tensor_meta)
        }
        if None in input_indices or len(input_indices) != len(input_tensor_meta):
            raise AssertionError("Expected unique names for backward benchmark inputs")
        self.dkdv = dkdv
        self.dq = dq
        self.dq_output_index = dq_output_index
        self.dkdv_input_indices = tuple(
            input_indices[meta.name] for meta in dkdv.input_tensor_meta
        )
        self.dq_input_indices = tuple(
            input_indices[meta.name] for meta in dq.input_tensor_meta
        )
        dq_output_meta = input_tensor_meta[dq_output_index]
        if any(
            getattr(dq_output_meta, field) != getattr(dq.output_tensor_meta, field)
            for field in ("device", "dtype", "sizes", "strides", "offset")
        ):
            raise AssertionError("Expected matching dQ output and input metadata")
        key = code_hash(
            repr(
                (
                    _request_key(dkdv),
                    _request_key(dq),
                    self.dkdv_input_indices,
                    self.dq_input_indices,
                    dq_output_index,
                )
            )
        )
        super().__init__(
            kernel_name="flex_attention_backward_rocm_split",
            input_tensor_meta=input_tensor_meta,
            output_tensor_meta=dkdv.output_tensor_meta,
            extra_args=(),
            module_path=dkdv.module_path,
            module_cache_key=key,
            num_stages=dkdv.num_stages,
            num_warps=dkdv.num_warps,
        )
        for field in (
            "config_max_autotune",
            "config_cudagraph_benchmarking",
            "cudagraph_unroll",
        ):
            if getattr(dkdv, field) != getattr(dq, field):
                raise AssertionError("Expected one benchmark policy for both phases")
        self.config_max_autotune = dkdv.config_max_autotune
        self.config_cudagraph_benchmarking = dkdv.config_cudagraph_benchmarking
        self.cudagraph_unroll = dkdv.cudagraph_unroll

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        dkdv_run = self.dkdv.make_run_fn(
            *(input_tensors[index] for index in self.dkdv_input_indices), out=out
        )
        dq_run = self.dq.make_run_fn(
            *(input_tensors[index] for index in self.dq_input_indices),
            out=input_tensors[self.dq_output_index],
        )

        def run() -> None:
            dkdv_run()
            dq_run()

        return run

    def cleanup_run_fn(self) -> None:
        try:
            self.dkdv.cleanup_run_fn()
        finally:
            self.dq.cleanup_run_fn()

    def precompile(self) -> None:
        try:
            self.dkdv.precompile()
            self.dq.precompile()
            self.n_regs = max(self.dkdv.n_regs, self.dq.n_regs)
        finally:
            self.cleanup_run_fn()


class RocmBackwardChoiceCaller(TritonTemplateCaller):
    """Select two backward kernels as one candidate without workspace reductions."""

    def __init__(
        self,
        input_nodes: list[Buffer],
        dkdv: TritonTemplateCaller,
        dq: TritonTemplateCaller,
    ) -> None:
        if len(input_nodes) != 16:
            raise AssertionError("Expected the 16-input backward ABI without captures")
        if dkdv.workspace_arg is not None or dq.workspace_arg is not None:
            raise AssertionError("Separate backward candidates cannot use workspace")
        unique_nodes = {node.get_name(): node for node in input_nodes}
        input_names = list(unique_nodes)
        input_tensor_meta = TensorMeta.from_irnodes(list(unique_nodes.values()))
        if not isinstance(input_tensor_meta, list):
            raise AssertionError("Expected a list of backward input tensor metadata")
        bmreq = RocmBackwardBenchmarkRequest(
            input_tensor_meta,
            dkdv.bmreq,
            dq.bmreq,
            input_names.index(input_nodes[6].get_name()),
        )
        super().__init__(
            name=f"rocm_backward_split_{bmreq.module_cache_key}",
            input_nodes=input_nodes,
            layout=dkdv.layout,
            make_kernel_render=None,
            description=f"dK/dV: {dkdv.description}; dQ: {dq.description}",
            bmreq=bmreq,
            log_info=dict(dkdv.info_dict(), ROCM_SEPARATE_DQ=True),
            mutated_inputs=[input_nodes[6], input_nodes[7]],
        )
        self.dkdv = dkdv
        self.dq = dq

    def hash_key(self) -> str:
        return f"rocm_backward_split_{self.bmreq.module_cache_key}"

    def output_node(self) -> TensorBox:
        return self.dkdv.output_node()

    def output_grad_query(self) -> TensorBox:
        # Benchmarking aliases the original dQ input; codegen returns a new dQ buffer.
        return self.dq.output_node()

    def get_make_kernel_render(self):
        raise NotImplementedError("Separate backward requires immediate selection")
