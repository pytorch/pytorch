import contextlib
import logging
import os
from collections.abc import Callable
from typing import Any, cast, TypeVar
from unittest.mock import patch

import torch
import torch.utils
from torch.utils._ordered_set import OrderedSet

from ..._dynamo.utils import counters
from .. import config, ir
from ..kernel.mm_common import mm_args
from ..select_algorithm import ChoiceCaller, DataProcessorTemplateWrapper
from ..utils import parallel_num_threads
from ..virtualized import V
from .cpp import get_export_declaration
from .cpp_gemm_template import (
    CppGemmTemplate,
    expand_bias,
    gen_2d_view_of_epilogue_buf,
    prune_tensors,
    transpose_w,
)
from .cpp_micro_gemm import CppMicroGemmAMX, create_micro_gemm
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import (
    create_epilogue_with_attr,
    DTYPE_TO_CPP,
    GemmBlocking,
    get_gemm_template_output_and_compute_dtype,
)


log = logging.getLogger(__name__)

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
{{micro_gemm.codegen_define(kernel)}}

extern "C" {{export_declaration}}
{{kernel.def_kernel(inputs=kernel_args, outputs=Y_list, aliases=aliases)}}
{
    {{kernel.maybe_codegen_profile()}}
    {{ template.codegen_blocks(
        num_threads, N, K, micro_gemm, is_dynamic_M, kernel, GemmOuts[0], config, L1_cache_size, L2_cache_size, X_list[0], W_list[0]
    ) }}
{%- if num_threads > 1 %}
    {%- set use_dynamic_threads = ((config.cpp.threads < 1) and (num_threads == cpu_count)) or config.cpp.dynamic_threads %}
    {%- if use_dynamic_threads %}
    #pragma omp parallel
    {%- else %}
    #pragma omp parallel num_threads({{num_threads}})
    {%- endif %}
    {
        {{ template.codegen_multi_threads_params()|indent(8, false) }}
{%- else %}
    {
        {{ template.codegen_single_thread_params(is_dynamic_M)|indent(8, false) }}
{%- endif %}
        {{ micro_gemm.codegen_init(kernel) }}
{%- set acc_buf_name_list=[] %}
{%- set acc_buf_name_prefix = "local_acc_buf_" %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set acc_buf_name = acc_buf_name_prefix + gemm_idx|string %}
    {{ kernel.define_buffer(acc_buf_name, ["Mc_blocks*Mr", "Nc_blocks*Nr"], acc_buf_dtype) }}
    {%- set acc_buf_name_list=acc_buf_name_list.append(acc_buf_name) %}
{%- endfor %}
        for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
            {{ template.codegen_m_loop_params()|indent(12, false) }}
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                {{ template.codegen_n_loop_params()|indent(16, false) }}
{%- set acc_list=[] %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set acc_list = acc_list.append( kernel.local_buffers[acc_buf_name_list[gemm_idx]] ) %}
    {{ kernel.reinit_buffer_if_null(acc_buf_name_list[gemm_idx]) }}
{%- endfor %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
{%- set tile_X_list=[] %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set tile_X_list = tile_X_list.append( kernel.slice_nd(X_list[gemm_idx], [("m_start", "m_end"), ("k_start", "k_end")]) ) %}
{%- endfor %}
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
{%- set tile_W_3d_list=[] %}
{%- set tile_W_list=[] %}
{%- set acc_slice_list=[] %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set acc_slice_list = acc_slice_list.append(
        kernel.slice_nd(acc_list[gemm_idx], [("0", "m_end - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")])
    ) %}
    {%- set tile_W_3d_list = tile_W_3d_list.append(
        kernel.slice_nd(W_list[gemm_idx], [("nci", "nci + 1"), ("k_start", "k_end"), ()])
    ) %}
{%- endfor %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set tile_W_list = tile_W_list.append(
        kernel.view(tile_W_3d_list[gemm_idx], ["k_end - k_start", micro_gemm.register_blocking.block_n])
    ) %}
{%- endfor %}
                        if (kc == k_block_start) {
                            {%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
                                {{ micro_gemm.codegen_call(
                                    kernel, tile_X_list[gemm_idx], tile_W_list[gemm_idx], acc_slice_list[gemm_idx], accum=False
                                )|indent(28, false) }}
                            {%- endfor %}
                        } else {
                            {%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
                                {{ micro_gemm.codegen_call(
                                    kernel, tile_X_list[gemm_idx], tile_W_list[gemm_idx], acc_slice_list[gemm_idx], accum=True
                                )|indent(28, false) }}
                            {%- endfor %}
                        }
                    }
                }
                {
{%- set tile_acc_list = [] %}
{%- set tile_Y_list = [] %}
{%- for gemm_idx in range(0, gemm_grouped_num, 1) %}
    {%- set tile_acc_list = tile_acc_list.append(
        kernel.slice_nd(acc_list[gemm_idx], [("0", "m_end - m_start"), ("0", "n_end - n_start")])
    ) %}
    {%- set tile_Y_list = tile_Y_list.append(
        kernel.slice_nd(Y_2d_list[gemm_idx], [("m_start", "m_end"), ("n_start", "n_end")])
    ) %}
{%- endfor %}
                    {{ kernel.store_outputs(
                        tile_Y_list,
                        tile_acc_list,
                        GemmOuts,
                        epilogue_nodes,
                        offsets=("m_start", "n_start"),
                        reindexers=reindexers,
                        multi_output_buffers=multi_output_buffers
                    )|indent(20, false)
                    }}
                }
            }
        }
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
}
"""


def get_deduplicated_act(act_mapping: dict[int, ir.IRNode]) -> list[ir.IRNode]:
    act_deduplicated = []
    act_deduplicated_name: OrderedSet[str] = OrderedSet()
    for act_idx in range(len(act_mapping.values())):
        act = act_mapping[act_idx]
        if act.get_name() not in act_deduplicated_name:
            act_deduplicated.append(act)
            act_deduplicated_name.add(act.get_name())
    return act_deduplicated


class CppGroupedGemmTemplate(CppGemmTemplate):
    def __init__(
        self,
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta: int = 1,
        alpha: int = 1,
        has_bias: bool = False,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
        act_mapping: dict[int, ir.IRNode] | None = None,
        gemm_grouped_num: int = 1,
    ) -> None:
        """
        Template for Group of GEMMs:
        * Each GEMM has the same dimensions (m, n, k) and the same leading dimensions (lda, ldb, ldc)
          for their A, B, and C matrices.
        * Each GEMM has distinct or shared activations, has distinct weight, has unique bias or no bias, has distinct epilogues.
        * In the current implementation, the outputs of all GEMMs are accumulated using pointwise epilogues.
          This behavior can be extended in the future if needed.
        """
        super().__init__(
            input_nodes,
            layout,
            num_threads,
            register_blocking,
            beta,
            alpha,
            has_bias,
            epilogue_creator,
        )
        self.act_mapping = act_mapping
        self.gemm_grouped_num = gemm_grouped_num
        # pyrefly: ignore [bad-override]
        self.output_node: list[ir.Buffer] = [
            ir.Buffer(name="buf_out" + str(idx), layout=layout)
            for idx in range(gemm_grouped_num)
        ]

    @classmethod
    # pyrefly: ignore [bad-override]
    def add_choices(
        cls,
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[ir.IRNode],
        beta: int = 1,
        alpha: int = 1,
        has_bias: tuple[bool, ...] = (False, False),
        trans_w: bool = False,
        input_indices: list[int] | None = None,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
        act_mapping: dict[int, ir.IRNode] | None = None,  # gemm idx to its act buf
    ) -> DataProcessorTemplateWrapper:
        # Input nodes order: x, optional[x1], ... w0, w1, ... optional[b0], optional[b1], ...
        gemm_grouped_num = len(has_bias)
        assert act_mapping
        act_deduplicated = get_deduplicated_act(act_mapping)
        wgt_start_idx = len(act_deduplicated)
        bias_start_idx = wgt_start_idx + gemm_grouped_num
        input_indices = list(range(len(input_nodes)))

        _T = TypeVar("_T", ir.IRNode, torch.Tensor)
        _U = TypeVar("_U", ir.Layout, torch.Tensor)

        def reorder_and_filter(
            inputs: list[_T],
            layout_or_out: _U,
        ) -> tuple[list[_T], _U]:
            assert input_indices is not None, "input_indices must be set"
            return [inputs[idx] for idx in input_indices], layout_or_out

        new_inputs, new_layout = reorder_and_filter(input_nodes, layout)

        def maybe_to_dense(
            inputs: list[_T],
            layout_or_out: _U,
        ) -> tuple[list[_T], _U]:
            new_inputs = list(inputs)
            for idx in range(wgt_start_idx, wgt_start_idx + gemm_grouped_num):
                if isinstance(inputs[idx], torch.Tensor):
                    W = inputs[idx]
                    assert isinstance(W, torch.Tensor), "W must be a torch.Tensor"
                    # pyrefly: ignore [unsupported-operation]
                    new_inputs[idx] = W.to_dense() if W.is_mkldnn else W
            return new_inputs, layout_or_out

        def normalize_shapes(
            inputs: list[_T],
            layout_or_out: _U,
        ) -> tuple[list[_T], _U]:
            new_inputs: list[_T] = list(inputs)
            if not trans_w:
                return new_inputs, layout_or_out
            X = new_inputs[0]
            for wgt_idx in range(wgt_start_idx, wgt_start_idx + gemm_grouped_num):
                new_input = new_inputs[wgt_idx]
                new_inputs[wgt_idx] = transpose_w(new_input, trans_w)
            for bias_idx in range(bias_start_idx, len(new_inputs)):
                # pyrefly: ignore [bad-argument-type]
                new_bias = expand_bias(new_inputs[bias_idx], X)
                assert new_bias is not None
                # pyrefly: ignore [unsupported-operation]
                new_inputs[bias_idx] = new_bias
            return new_inputs, layout_or_out

        num_threads = parallel_num_threads()
        new_inputs, _ = normalize_shapes(*maybe_to_dense(new_inputs, new_layout))
        m, n, k, *_ = mm_args(new_inputs[0], new_inputs[wgt_start_idx])
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            new_inputs[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            "micro_gemm",
            m,
            n,
            k,
            input_dtype=new_inputs[0].get_dtype(),
            input2_dtype=new_inputs[wgt_start_idx].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=alpha,
            num_threads=num_threads,
        )
        assert micro_gemm is not None
        _, block_n, _ = micro_gemm.register_blocking
        new_size, padded_n = cls.get_padded_size(
            n, block_n, k, should_block_weight=True
        )
        padding = padded_n - n

        def pack_weight(
            inputs: list[_T],
            layout_or_out: _U,
        ) -> tuple[list[_T], _U]:
            new_W_list = []
            new_inputs = list(inputs)
            W_list = new_inputs[wgt_start_idx : wgt_start_idx + gemm_grouped_num]
            for W in W_list:
                blocked_w = cls.block_weight(W, new_size, padding)
                new_W_list.append(cls.pack_vnni_weight(blocked_w, micro_gemm, new_size))
            new_inputs[wgt_start_idx : wgt_start_idx + gemm_grouped_num] = new_W_list
            return new_inputs, layout_or_out

        def preprocessor(
            inputs: list[_T],
            layout: _U,
        ) -> tuple[list[_T], _U]:
            return pack_weight(
                *normalize_shapes(*maybe_to_dense(*reorder_and_filter(inputs, layout)))
            )

        def postprocessor(output: _T) -> _T:
            if isinstance(output, ir.TensorBox):
                template_buffer = ir.InputsKernel.unwrap_storage_for_input(output)
                assert isinstance(template_buffer, ir.CppTemplateBuffer)
                new_input_nodes, _ = reorder_and_filter(input_nodes, layout)
                W_nodes = new_input_nodes[
                    wgt_start_idx : wgt_start_idx + gemm_grouped_num
                ]
                W_tensor = []
                for W_node in W_nodes:
                    assert W_node.get_name() in V.graph.constants
                    # pyrefly: ignore [bad-argument-type]
                    W_tensor.append(V.graph.constants[W_node.get_name()])
                # pyrefly: ignore [unsupported-operation]
                new_input_nodes[wgt_start_idx : wgt_start_idx + gemm_grouped_num] = (
                    W_tensor  # type: ignore[assignment]
                )
                new_input_nodes, _ = pack_weight(
                    *normalize_shapes(*maybe_to_dense(new_input_nodes, layout))
                )
                # Prune unused tensors
                prune_tensors(input_nodes, new_input_nodes)
                for idx in range(wgt_start_idx, wgt_start_idx + gemm_grouped_num):
                    W_packed = new_input_nodes[idx]
                    assert isinstance(W_packed, torch.Tensor)
                    W_packed_constant = V.graph.add_tensor_constant(W_packed)
                    template_buffer.inputs[idx] = (
                        ir.InputsKernel.unwrap_storage_for_input(W_packed_constant)
                    )

            return output

        template = DataProcessorTemplateWrapper(
            CppGroupedGemmTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            num_threads=num_threads,
            register_blocking=micro_gemm.register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            act_mapping=act_mapping,
            gemm_grouped_num=gemm_grouped_num,
        )
        template.maybe_append_choice(choices)
        return template

    def render(  # type: ignore[override,return,no-untyped-def]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = None,
        flag_template_buffer_has_other_users: bool | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        **kwargs,
    ) -> str:
        assert self.act_mapping
        act_deduplicated = get_deduplicated_act(self.act_mapping)
        wgt_start_idx = len(act_deduplicated)
        bias_start_idx = wgt_start_idx + self.gemm_grouped_num
        X_list = list(self.act_mapping.values())
        W_list = self.input_nodes[wgt_start_idx : wgt_start_idx + self.gemm_grouped_num]
        inp_list = []
        cur_idx = bias_start_idx
        for inp_idx in range(self.gemm_grouped_num):
            inp = None
            # pyrefly: ignore [bad-index, index-error]
            if self.has_bias[inp_idx]:
                inp = self.input_nodes[cur_idx]
                cur_idx += 1
            inp_list.append(inp)

        Y_list = self.output_node
        multi_output_buffers = None
        if template_buffer_node is not None:
            W_list = template_buffer_node.inputs[
                wgt_start_idx : wgt_start_idx + self.gemm_grouped_num
            ]
            assert isinstance(template_buffer_node.outputs, list)
            Y_list = template_buffer_node.outputs
            counters["inductor"]["cpp_grouped_gemm_template"] += 1
            multi_output_buffers = template_buffer_node.outputs

        template_buffer = Y_list[0]
        fake_buffers: list[ir.Buffer] = []
        Y_2d_list = Y_list
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            X_list[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            input_dtype=X_list[0].get_dtype(),
            input2_dtype=W_list[0].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=self.alpha,
            num_threads=self.num_threads,
        )
        assert micro_gemm is not None
        assert self.register_blocking == micro_gemm.register_blocking
        self.log_blockings()
        if isinstance(micro_gemm, CppMicroGemmAMX):
            counters["inductor"]["cpp_micro_gemm_amx_counter"] += 1

        L1_cache_size = torch.cpu.get_capabilities().get(
            "l1d_cache_size", 0
        )  # per core cache size in Bytes
        assert L1_cache_size > 0, f"Expect L1_cache_size > 0 but got {L1_cache_size}"

        L2_cache_size = torch.cpu.get_capabilities().get(
            "l2_cache_size", 0
        )  # per core cache size in Bytes
        assert L2_cache_size > 0, f"Expect L2_cache_size > 0 but got {L2_cache_size}"

        epilogues: list[ir.IRNode] = []
        reindexers: list[Callable[[list[Any]], list[Any]] | None] = []
        gemm_output_buffers: list[ir.Buffer] = []
        for out_buf_idx in range(self.gemm_grouped_num):
            gemm_output_name = f"{template_buffer.get_name()}_GemmOut" + str(
                out_buf_idx
            )
            gemm_output_buffers.append(
                ir.Buffer(name=gemm_output_name, layout=template_buffer.layout)
            )

        assert not self.epilogue_creator, (
            "epilogue_creator is not supported yet in Grouped GEMM Template"
        )

        kernel_args: dict[str, ir.IRNode | None] = {}
        for x_idx in range(wgt_start_idx):
            kernel_args["X" + str(x_idx)] = act_deduplicated[x_idx]
        for w_idx in range(self.gemm_grouped_num):
            kernel_args["W" + str(w_idx)] = W_list[w_idx]
        for inp_idx in range(self.gemm_grouped_num):
            kernel_args["inp" + str(inp_idx)] = inp_list[inp_idx]

        def _bias_add_epilogue(buf: ir.IRNode, inp: ir.IRNode) -> ir.Pointwise:
            return create_epilogue_with_attr(
                buf, "bias_add", other=inp, beta=self.beta, dtype=self.layout.dtype
            )

        for gemm_idx, inp in enumerate(inp_list):
            if inp:
                buffer_name = Y_list[gemm_idx].get_name()
                epilogues.append(
                    ir.ComputedBuffer(
                        name=buffer_name,
                        layout=template_buffer.layout,
                        data=_bias_add_epilogue(gemm_output_buffers[gemm_idx], inp),
                    )
                )
                reindexers.append(None)

        if epilogue_nodes:
            epilogues.extend(epilogue_nodes)
            for epilogue_node in epilogue_nodes:
                Y = cast(ir.Buffer, epilogue_node)
                _, reindexers = gen_2d_view_of_epilogue_buf(
                    Y,
                    template_buffer,
                    [
                        epilogue_node,
                    ],
                    reindexers,
                    default_reindexers=[
                        None,
                    ],
                )

        options = dict(
            N=self.n,
            K=self.k,
            PADDED_N=self.padded_n,
            aliases={},
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            template=self,
            kernel=kernel,
            export_declaration=get_export_declaration(),
            acc_buf_dtype=torch.float,
            DTYPE_TO_CPP=DTYPE_TO_CPP,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            config=config,
            epilogue_nodes=epilogues,
            GemmOuts=gemm_output_buffers,
            reindexers=reindexers,
            kernel_args=kernel_args,
            X_list=X_list,
            W_list=W_list,
            gemm_grouped_num=self.gemm_grouped_num,
            Y_list={"Y" + str(idx): Y for idx, Y in enumerate(Y_list)},
            Y_2d_list=Y_2d_list,
            multi_output_buffers=multi_output_buffers,
            cpu_count=os.cpu_count(),
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(V.graph, "get_dtype", self._fake_get_dtype(fake_buffers))
            )
            return self._template_from_string(GEMM_TEMPLATE).render(**options)
