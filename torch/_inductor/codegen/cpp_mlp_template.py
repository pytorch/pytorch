# mypy: allow-untyped-defs
import contextlib
import logging
from typing import Any, Callable, cast, List, Optional, Set, Union
from unittest.mock import patch

import torch
import torch.utils

from ..._dynamo.utils import counters
from .. import config, ir, lowering as L
from ..kernel.mm_common import mm_args
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import parallel_num_threads
from ..virtualized import V
from .cpp import get_export_declaration
from .cpp_gemm_template import CppPackedGemmTemplate, get_padded_n
from .cpp_micro_gemm import CppMicroGemmAMX, create_micro_gemm, LayoutType
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import (
    DTYPE_TO_CPP,
    GemmBlocking,
    get_gemm_template_output_and_compute_dtype,
)


log = logging.getLogger(__name__)

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
{{micro_gemm.codegen_define(kernel)}}

{%- set kernel_args = {"X": X, "W": W, "W1": W1, "inp": inp, "inp1": inp1} %}

extern "C" {{export_declaration}}
{{kernel.def_kernel(inputs=kernel_args, outputs={"Y": Y}, aliases=aliases)}}
{
    {{kernel.maybe_codegen_profile()}}
    {{ template.codegen_blocks(
        num_threads, N, K, micro_gemm, is_dynamic_M, kernel, GemmOut, config, L1_cache_size, L2_cache_size, X, W
    ) }}

{%- if num_threads > 1 %}
    #pragma omp parallel num_threads({{num_threads}})
    {
        {{ template.codegen_multi_threads_param()|indent(8, false) }}
{%- else %}
    {
        {{ template.codegen_single_thread_param(is_dynamic_M)|indent(8, false) }}
{%- endif %}
        {{ micro_gemm.codegen_init(kernel) }}
{%- if use_local_acc %}
    {%- set acc_buf_name = "local_acc_buf" %}
        {{ kernel.define_buffer(acc_buf_name, ["Mc_blocks*Mr", "Nc_blocks*Nr"], acc_buf_dtype) }}
    {%- set acc_buf2_name = "local_acc_buf2" %}
        {{ kernel.define_buffer(acc_buf2_name, ["Mc_blocks*Mr", "Nc_blocks*Nr"], acc_buf_dtype) }}
{%- endif %}
    if (horizontal_transverse) {
        for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
            const int64_t n_start = nc * Nr;
            const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
            const int64_t n_size = n_end - n_start;
            // NB: assume we pad N, nc_block_end won't exceed padded N here.
            const int64_t nc_block_end = std::min(nc + Nc_blocks, n_block_end);
            for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
                const int64_t my_mc_block_id = (mc_block_id + n_slice_id) % num_Mc_blocks_per_thread;
                const int64_t mc = m_block_start + my_mc_block_id * Mc_blocks;
                const int64_t m_start = mc * Mr;
                const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
                const int64_t m_size = m_end - m_start;
{%- if use_local_acc %}
    {%- set acc = kernel.local_buffers[acc_buf_name] %}
                {{ kernel.reinit_buffer_if_null(acc_buf_name) }}
    {%- set acc2 = kernel.local_buffers[acc_buf2_name] %}
                {{ kernel.reinit_buffer_if_null(acc_buf2_name) }}
{%- else %}
    {%- set acc = kernel.slice_nd(GemmOut, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- endif %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
                    for (int64_t mci = m_start; mci < m_end; mci+=Mr) {
                        const int64_t m_start_i = mci;
                        const int64_t m_end_i = m_start_i + Mr;
{%- set tile_X = kernel.slice_nd(X, [("m_start_i", "m_end_i"), ("k_start", "k_end")]) %}
                        for (int64_t nci = nc; nci < nc_block_end; nci++) {
{%- set acc_slice = kernel.slice_nd(acc, [("m_start_i - m_start", "m_end_i - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")]) %}
{%- set acc2_slice = kernel.slice_nd(
    acc2, [("m_start_i - m_start", "m_end_i - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")]
) %}
{%- set tile_W_3d = kernel.slice_nd(W, [("nci", "nci + 1"), ("k_start", "k_end"), ()]) %}
{%- set tile_W = kernel.view(tile_W_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
{%- set tile_W1_3d = kernel.slice_nd(W1, [("nci", "nci + 1"), ("k_start", "k_end"), ()]) %}
{%- set tile_W1 = kernel.view(tile_W1_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
                            if (kc == k_block_start) {
                                {{ micro_gemm.codegen_call(
                                    kernel, tile_X, tile_W, acc_slice, accum=False, horizontal_transverse=True
                                )|indent(28, false)
                                }}
                                {{ micro_gemm.codegen_call(
                                    kernel, tile_X, tile_W1, acc2_slice, accum=False, horizontal_transverse=True
                                )|indent(28, false)
                                }}
                            } else {
                                {{ micro_gemm.codegen_call(
                                    kernel, tile_X, tile_W, acc_slice, accum=True, horizontal_transverse=True
                                )|indent(28, false)
                                }}
                                {{ micro_gemm.codegen_call(
                                    kernel, tile_X, tile_W1, acc2_slice, accum=True, horizontal_transverse=True
                                )|indent(28, false)
                                }}
                            }
                        }
                    }
                }

                {
{%- set tile_Y = kernel.slice_nd(Y_2d, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- set tile_acc = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("0", "n_end - n_start")]) %}
{%- set tile_acc1 = kernel.slice_nd(acc2, [("0", "m_end - m_start"), ("0", "n_end - n_start")]) %}
{%- if has_gate_bias %}
{%- set tile_inp = kernel.slice_nd(inp, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- else %}
{%- set tile_inp = tile_Y %}
{%- endif %}
{%- if has_up_bias %}
{%- set tile_inp1 = kernel.slice_nd(inp1, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- else %}
{%- set tile_inp1 = tile_Y %}
{%- endif %}
                    // silu-mul epilogues
                    {{ kernel.store_output(
                        tile_Y,
                        (tile_acc, tile_acc1),
                        (GemmOut, GemmOut1),
                        epilogue_nodes,
                        offsets=("m_start", "n_start"),
                        reindexers=reindexers
                    )|indent(20, false)
                    }}
                }
            }
        }
    } else {
        for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
            {{ template.codegen_m_loop_param()|indent(12, false) }}
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                {{ template.codegen_n_loop_param()|indent(16, false) }}
{%- if use_local_acc %}
    {%- set acc = kernel.local_buffers[acc_buf_name] %}
                {{ kernel.reinit_buffer_if_null(acc_buf_name) }}
    {%- set acc2 = kernel.local_buffers[acc_buf2_name] %}
                {{ kernel.reinit_buffer_if_null(acc_buf2_name) }}
{%- else %}
    {%- set acc = kernel.slice_nd(GemmOut, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- endif %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
{%- set tile_X = kernel.slice_nd(X, [("m_start", "m_end"), ("k_start", "k_end")]) %}
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
{%- set acc_slice = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")]) %}
{%- set acc2_slice = kernel.slice_nd(acc2, [("0", "m_end - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")]) %}
{%- set tile_W_3d = kernel.slice_nd(W, [("nci", "nci + 1"), ("k_start", "k_end"), ()]) %}
{%- set tile_W = kernel.view(tile_W_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
{%- set tile_W1_3d = kernel.slice_nd(W1, [("nci", "nci + 1"), ("k_start", "k_end"), ()]) %}
{%- set tile_W1 = kernel.view(tile_W1_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
                        if (kc == k_block_start) {
                            {{ micro_gemm.codegen_call(
                                kernel, tile_X, tile_W, acc_slice, accum=False, horizontal_transverse=False
                            )|indent(28, false)
                            }}
                            {{ micro_gemm.codegen_call(
                                kernel, tile_X, tile_W1, acc2_slice, accum=False, horizontal_transverse=False
                            )|indent(28, false)
                            }}
                        } else {
                            {{ micro_gemm.codegen_call(
                                kernel, tile_X, tile_W, acc_slice, accum=True, horizontal_transverse=False
                            )|indent(28, false)
                            }}
                            {{ micro_gemm.codegen_call(
                                kernel, tile_X, tile_W1, acc2_slice, accum=True, horizontal_transverse=False
                            )|indent(28, false)
                            }}
                        }
                    }
                }

                {
{%- set tile_Y = kernel.slice_nd(Y_2d, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- set tile_acc = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("0", "n_end - n_start")]) %}
{%- set tile_acc1 = kernel.slice_nd(acc2, [("0", "m_end - m_start"), ("0", "n_end - n_start")]) %}
{%- if has_gate_bias %}
{%- set tile_inp = kernel.slice_nd(inp, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- else %}
{%- set tile_inp = tile_Y %}
{%- endif %}
{%- if has_up_bias %}
{%- set tile_inp1 = kernel.slice_nd(inp1, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- else %}
{%- set tile_inp1 = tile_Y %}
{%- endif %}
                    // silu-mul epilogues
                    {{ kernel.store_output(
                        tile_Y,
                        (tile_acc, tile_acc1),
                        (GemmOut, GemmOut1),
                        epilogue_nodes,
                        offsets=("m_start", "n_start"),
                        reindexers=reindexers
                    )|indent(20, false)
                    }}
                }
            }
        }
    }
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
}
"""


class CppPackedMLPTemplate(CppPackedGemmTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ) -> None:
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
        self.silu_mul_fusion = True

    @staticmethod
    def add_choices(
        choices,
        layout,
        input_nodes,
        beta=1,
        alpha=1,
        has_bias=(False, False),
        trans_w=False,
        input_indices=None,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ):
        assert len(input_nodes) >= 3  # x, w0, w1, optional[b0], optional[b1]

        input_indices = list(range(len(input_nodes)))

        def reorder_and_filter(inputs, layout_or_out):
            return [inputs[idx] for idx in input_indices], layout_or_out

        new_inputs, new_layout = reorder_and_filter(input_nodes, layout)

        def maybe_to_dense(inputs, layout_or_out):
            new_inputs = list(inputs)
            for idx in [1, 2]:
                if isinstance(inputs[idx], torch.Tensor):
                    W = inputs[idx]
                    new_inputs[idx] = W.to_dense() if W.is_mkldnn else W
            return new_inputs, layout_or_out

        def normalize_shapes(inputs, layout_or_out):
            new_inputs = list(inputs)

            if not trans_w:
                return new_inputs, layout_or_out
            X = new_inputs[0]
            W0 = new_inputs[1]
            W1 = new_inputs[2]

            B0 = new_inputs[3] if has_bias[0] else None
            B1 = None
            if has_bias[1]:
                B1 = new_inputs[4] if has_bias[0] else new_inputs[3]

            def _transpose_w(W):
                if isinstance(W, ir.IRNode):
                    if trans_w:
                        if not isinstance(W, ir.TensorBox):
                            W = ir.TensorBox(W)
                        W = L.permute(W, [1, 0])
                else:
                    if trans_w:
                        assert isinstance(W, torch.Tensor)
                        W = W.transpose(0, 1)
                return W

            W0 = _transpose_w(W0)
            W1 = _transpose_w(W1)
            new_inputs[1] = W0
            new_inputs[2] = W1

            def _expand_bias(B):
                if B is not None:
                    if isinstance(B, ir.IRNode):
                        if not isinstance(B, ir.TensorBox):
                            B = ir.TensorBox(B)
                        B = L.expand(B, (X.get_size()[0], B.get_size()[-1]))
                    else:
                        assert isinstance(B, torch.Tensor)
                        B = B.expand(X.shape[0], B.shape[-1])
                return B

            B0 = _expand_bias(B0)
            B1 = _expand_bias(B1)
            if B0 is not None:
                new_inputs[3] = B0
            if B1 is not None:
                idx = 4 if B0 is not None else 3
                new_inputs[idx] = B1
            return new_inputs, layout_or_out

        # TODO(jgong5): decide proper number of threads per problem size
        num_threads = parallel_num_threads()
        new_inputs, _ = normalize_shapes(*maybe_to_dense(new_inputs, new_layout))
        m, n, k, *_ = mm_args(new_inputs[0], new_inputs[1])
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            new_inputs[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            "micro_gemm",
            m,
            n,
            k,
            input_dtype=new_inputs[0].get_dtype(),
            input2_dtype=new_inputs[1].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=alpha,
            num_threads=num_threads,
        )
        assert micro_gemm is not None
        _, block_n, _ = micro_gemm.register_blocking
        padded_n = get_padded_n(n, block_n)

        def pack_weight(inputs, layout_or_out):
            W = inputs[1]
            W1 = inputs[2]
            new_inputs = list(inputs)

            def _get_block_w(W):
                blocked_w: Union[ir.IRNode, torch.Tensor] = W
                if isinstance(W, ir.IRNode):
                    new_size = [padded_n // block_n, k, block_n]
                    blocked_w = ir.Buffer(
                        name=W.get_name(),  # Borrow the registered buffer name
                        layout=ir.FixedLayout(
                            W.get_device(),
                            W.get_dtype(),
                            new_size,
                            ir.FlexibleLayout.contiguous_strides(new_size),
                            0,
                        ),
                    )
                else:
                    blocked_w = (
                        torch.nn.functional.pad(W, (0, padded_n - n))
                        .reshape(k, padded_n // block_n, block_n)
                        .transpose(0, 1)
                        .contiguous()
                    )
                    if micro_gemm.get_b_layout() != LayoutType.NORMAL:
                        layout_str = (
                            "VNNI4"
                            if micro_gemm.get_b_layout() == LayoutType.VNNI4
                            else "VNNI2"
                        )
                        assert micro_gemm.get_b_layout() in [
                            LayoutType.VNNI2,
                            LayoutType.VNNI4,
                        ], f"We only support {layout_str} for now"
                        vnni_size = (
                            4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
                        )
                        assert (
                            k % vnni_size == 0
                        ), f"k should be divisible by vnni_size for {layout_str} layout"
                        blocked_w = (
                            blocked_w.view(
                                padded_n // block_n, k // vnni_size, vnni_size, block_n
                            )
                            .transpose(-1, -2)
                            .contiguous()
                            .view(padded_n // block_n, k, block_n)
                        )
                    # normalize stride to be "contiguous_strides" per size
                    # this avoids the problems in L.view during template codegen
                    new_stride = [1]
                    for sz in reversed(blocked_w.shape[1:]):
                        new_stride.insert(0, new_stride[0] * sz)
                    blocked_w = blocked_w.as_strided(blocked_w.shape, new_stride)
                return blocked_w

            new_inputs[1] = _get_block_w(W)
            new_inputs[2] = _get_block_w(W1)
            return new_inputs, layout_or_out

        def preprocessor(inputs, layout):
            return pack_weight(
                *normalize_shapes(*maybe_to_dense(*reorder_and_filter(inputs, layout)))
            )

        def postprocessor(output):
            if isinstance(output, ir.TensorBox):
                # prepack the weight as input to the template buffer
                template_buffer = ir.InputsKernel.unwrap_storage_for_input(output)
                assert isinstance(template_buffer, ir.CppTemplateBuffer)
                new_input_nodes, _ = reorder_and_filter(input_nodes, layout)

                W_node = new_input_nodes[1]
                W_node2 = new_input_nodes[2]
                assert W_node.get_name() in V.graph.constants
                W = V.graph.constants[W_node.get_name()]
                W2 = V.graph.constants[W_node2.get_name()]

                new_input_nodes[1] = W
                new_input_nodes[2] = W2
                new_input_nodes, _ = pack_weight(
                    *normalize_shapes(*maybe_to_dense(new_input_nodes, layout))
                )
                W_packed = new_input_nodes[1]
                W_packed_constant = V.graph.add_tensor_constant(W_packed)
                new_input_nodes[1] = W_packed_constant

                template_buffer.inputs[1] = ir.InputsKernel.unwrap_storage_for_input(
                    W_packed_constant
                )

                W_packed2 = new_input_nodes[2]
                W_packed_constant2 = V.graph.add_tensor_constant(W_packed2)
                new_input_nodes[2] = W_packed_constant2

                template_buffer.inputs[2] = ir.InputsKernel.unwrap_storage_for_input(
                    W_packed_constant2
                )

            return output

        template = DataProcessorTemplateWrapper(
            CppPackedMLPTemplate,
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
        )
        template.maybe_append_choice(choices)
        return template

    def render(  # type: ignore[override,return]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        flag_template_buffer_has_other_users: Optional[bool] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        assert len(self.input_nodes) >= 3

        X, W = self.input_nodes[0], self.input_nodes[1]
        W1 = self.input_nodes[2]
        Y = self.output_node
        has_gate_bias = self.has_bias[0]
        has_up_bias = self.has_bias[1]
        inp = self.input_nodes[3] if has_gate_bias else None
        inp1 = None
        if has_up_bias:
            inp1 = self.input_nodes[4] if has_gate_bias else self.input_nodes[3]

        if template_buffer_node is not None:
            # Use the updated prepacked weight buffer
            W = template_buffer_node.inputs[1]
            W1 = template_buffer_node.inputs[2]
            Y = template_buffer_node
            counters["inductor"]["cpp_mlp_template"] += 1

        template_buffer = Y
        gemm_output_buffer = template_buffer

        fake_buffers: List[ir.Buffer] = []
        Y_aliases: Set[str] = set()

        # Force to use local acc
        use_local_acc = True

        Y_2d: Union[ir.Buffer, ir.ReinterpretView] = Y

        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            X.get_dtype()
        )
        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            input_dtype=X.get_dtype(),
            input2_dtype=W.get_dtype(),
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

        L1_cache_size = torch._C._cpu._L1d_cache_size()  # per core cache size in Bytes
        assert L1_cache_size > 0, f"Expect L1_cache_size > 0 but got {L1_cache_size}"

        L2_cache_size = torch._C._cpu._L2_cache_size()  # per core cache size in Bytes
        assert L2_cache_size > 0, f"Expect L2_cache_size > 0 but got {L2_cache_size}"

        # Hardcode the in-template epilogue
        # Bias-add for each linear if applicable
        # Silu and Mul post op
        epilogues: List[ir.IRNode] = []
        reindexers: List[Optional[Callable[[List[Any]], List[Any]]]] = []

        def epilogue_creator(buf, buf1):
            from ..virtualized import ops

            input_loader = buf.make_loader()
            input_loader1 = buf1.make_loader()
            if has_gate_bias:
                assert inp is not None
                inp_loader = inp.make_loader()
            if has_up_bias:
                assert inp1 is not None
                inp_loader1 = inp1.make_loader()
            dtype = buf.get_dtype()

            def inner_fn(index):
                input = input_loader(index)
                input1 = input_loader1(index)
                if has_gate_bias:
                    input = input + inp_loader(index)
                if has_up_bias:
                    input1 = input1 + inp_loader1(index)
                input = ops.mul(ops.sigmoid(input), input)
                return ops.mul(input, input1)

            return ir.Pointwise(
                device=buf.get_device(),
                dtype=dtype,
                inner_fn=inner_fn,
                ranges=buf.get_size(),
            )

        gemm_output_name = f"{template_buffer.get_name()}_GemmOut"
        gemm_output_buffer = ir.Buffer(
            name=gemm_output_name, layout=template_buffer.layout
        )
        current_input_buffer = gemm_output_buffer

        gemm_output_name1 = f"{template_buffer.get_name()}_GemmOut1"
        gemm_output_buffer1 = ir.Buffer(
            name=gemm_output_name1, layout=template_buffer.layout
        )
        current_input_buffer1 = gemm_output_buffer1

        buffer_name = template_buffer.get_name()
        epilogues.append(
            ir.ComputedBuffer(
                name=buffer_name,
                layout=template_buffer.layout,
                data=epilogue_creator(current_input_buffer, current_input_buffer1),
            )
        )
        reindexers.append(None)

        if epilogue_nodes:
            epilogues.extend(epilogue_nodes)
            assert Y.get_numel() == epilogues[-1].get_numel()
            Y = cast(ir.Buffer, epilogues[-1])
            if (
                Y.get_size() == template_buffer.get_size()
                and Y.get_stride() == template_buffer.get_stride()
            ):
                reindexers.extend([None] * len(epilogue_nodes))
                Y_2d = Y
            else:

                def get_reindexer(epilogue_node):
                    # From template_buffer to epilogue_node_ordered (ordered by stride decreasingly, in dense format), for example:
                    #   template_buffer:
                    #       size (324, 512), stride (512, 1)
                    #   epilogue_node_ordered (ordered by stride decreasingly, in dense format):
                    #       size (1, 18, 18, 512), stride (165888, 9216, 512, 1)
                    stride_order = list(
                        ir.get_stride_order(
                            V.graph.sizevars.size_hints(epilogue_node.get_stride())
                        )
                    )
                    fill_order = ir.stride_order2fill_order(stride_order)
                    reversed_fill_order = list(reversed(fill_order))
                    size_with_stride_ordered_decreasingly = [
                        epilogue_node.get_size()[i] for i in reversed_fill_order
                    ]
                    reshape_reindex = ir.View.dynamic_reshape_indexer(
                        size_with_stride_ordered_decreasingly,
                        template_buffer.get_size(),
                    )

                    # From epilogue_node_ordered (ordered by stride decreasingly, in dense format) to epilogue_node, for example:
                    #   epilogue_node_ordered (ordered by stride decreasingly, in dense format):
                    #       size (1, 18, 18, 512), stride (165888, 9216, 512, 1)
                    #   epilogue_node:
                    #       size (1, 18, 18, 512), stride (165888, 1, 9216, 512)
                    from_stride_ordered_decreasingly_to_epilogue_node_order = [
                        (len(stride_order) - 1) - stride_order[i]
                        for i in range(len(stride_order))
                    ]
                    stride_reindex = ir.same_reorder(
                        from_stride_ordered_decreasingly_to_epilogue_node_order
                    )

                    reindexer = ir.fuse_reindexing(stride_reindex, reshape_reindex)
                    return reindexer

                reindexers.extend([get_reindexer(epilogue_node) for epilogue_node in epilogue_nodes])  # type: ignore[list-item]
                if isinstance(Y, ir.BaseView):
                    storage = ir.StorageBox(Y.unwrap_view())
                else:
                    assert isinstance(Y, ir.Buffer)
                    storage = ir.StorageBox(Y)
                Y_2d = ir.ReinterpretView(
                    data=storage, layout=template_buffer.get_layout()
                )

        options = dict(
            X=X,
            W=W,
            inp=inp,
            Y=Y,
            N=self.n,
            K=self.k,
            PADDED_N=self.padded_n,
            GemmOut=gemm_output_buffer,
            aliases={alias: Y.get_name() for alias in Y_aliases},
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            template=self,
            kernel=kernel,
            export_declaration=get_export_declaration(),
            Y_2d=Y_2d,
            use_local_acc=use_local_acc,
            acc_buf_dtype=torch.float,
            DTYPE_TO_CPP=DTYPE_TO_CPP,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            config=config,
            W1=W1,
            inp1=inp1,
            has_gate_bias=has_gate_bias,
            has_up_bias=has_up_bias,
            epilogue_nodes=epilogues,
            GemmOut1=gemm_output_buffer1,
            reindexers=reindexers,
        )
        with contextlib.ExitStack() as stack:
            for buf in fake_buffers:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            return self._template_from_string(GEMM_TEMPLATE).render(**options)
