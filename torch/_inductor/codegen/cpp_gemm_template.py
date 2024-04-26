from typing import cast, List, Optional

import torch
from .. import ir

from ..ir import Buffer, CppTemplateBuffer, IRNode, Layout
from ..kernel.mm_common import mm_args
from ..lowering import permute, view
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import cache_on_self, parallel_num_threads
from ..virtualized import V
from .cpp_micro_gemm import create_micro_gemm
from .cpp_template import CppTemplate

from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import GemmBlocking

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}

{{micro_gemm.codegen_define()}}

extern "C"
{{kernel.def_kernel(inputs=[X, W, inp], outputs=[Y], names_str="X, W, inp, Y")}}
{
    constexpr int64_t num_threads = {{num_threads}};
    constexpr int64_t N = {{kernel.size(Y, 1)}};
    constexpr int64_t K = {{kernel.size(X, 1)}};
    constexpr int64_t M0 = {{micro_gemm.register_blocking.block_m}};
    constexpr int64_t N0 = {{micro_gemm.register_blocking.block_n}};
    constexpr int64_t K0 = {{micro_gemm.register_blocking.block_k}};
    constexpr int64_t N0_blocks = (N + N0 - 1) / N0;
    constexpr int64_t K0_blocks = (K + K0 - 1) / K0;

    static_assert(N % N0 == 0, "N dimension must be multiple of N0");

    {% if is_dynamic_M %}
    const int64_t M = {{kernel.size(Y, 0)}};
    const int64_t M0_blocks = (M + M0 - 1) / M0;
    // TODO: implement below
    const auto [Mt_blocks, Nt_blocks, Kt_blocks] = mm_get_thread_blocking(M, N, K, M0, N0, K0, num_threads);
    const int64_t M2_blocks = Mt_blocks; // TODO: improve cache blocking
    {% else %}
    constexpr int64_t M = {{kernel.size(Y, 0)}};
    constexpr int64_t M0_blocks = (M + M0 - 1) / M0;
    constexpr int64_t Mt_blocks = {{template.thread_blocking().block_m}};
    constexpr int64_t Nt_blocks = {{template.thread_blocking().block_n}};
    constexpr int64_t Kt_blocks = {{template.thread_blocking().block_k}};
    constexpr int64_t M2_blocks = {{template.cache_blocking().block_m}};
    {% endif %}
    constexpr int64_t K2_blocks = {{template.cache_blocking().block_k}};

    // TODO: support k-slicing
    TORCH_CHECK(Kt_blocks == K0_blocks, "Do not support k slicing yet.");
    // make sure all partitions are assigned
    TORCH_CHECK(
        Mt_blocks * Nt_blocks * Kt_blocks * {{num_threads}} >= M0_blocks * N0_blocks * K0_blocks,
        "Not all partitions are assigned."
    );

    #pragma omp parallel num_threads({{num_threads}})
    {
        int tid = omp_get_thread_num();
        int64_t m_block_start, m_block_end, n_block_start, n_block_end, k_block_start, k_block_end;
        mm_get_thread_blocks(
            tid, M, N, K, Mt_blocks, Nt_blocks, Kt_blocks,
            m_block_start, m_block_end, n_block_start, n_block_end, k_block_start, k_block_end);
        for (int64_t m2 = m_block_start; m2 < m_block_end; m2 += M2_blocks) {
            int64_t m_start = m2 * M0;
            int64_t m_end = std::min((m2 + M2_blocks) * M0, M);
            for (int64_t n2 = n_block_start; n2 < n_block_end; ++n2) {
                int64_t n_start = n2 * N0;
                // TODO: use float32 temporary buffer to support bfloat16/float16 gemm
                {% if inp is not none and beta != 0 %}
                for (int64_t m = m_start; m < m_end; ++m) {
                    #pragma omp simd
                    for (int64_t n = n_start; n < n_start + N0; ++n) {
                        {{kernel.index(Y, "m", "n")}} = beta * {{kernel.index(inp, "m", "n")}};
                    }
                }
                {% endif %}
                for (int64_t k2 = k_block_start; k2 < k_block_end; k2 += K2_blocks) {
                    int64_t k_start = k2 * K0;
                    int64_t k_end = std::min((k2 + K2_blocks) * K0, K);
                    {% set tile_X = kernel.slice_nd(X, [("m_start", "m_end"), ("k_start", "k_end")]) %}
                    {% set tile_W_3d = kernel.slice_nd(W, [("n2", "n2 + 1"), ("k_start", "k_end"), ()]) %}
                    {% set tile_W = kernel.view(tile_W_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
                    {% set tile_Y = kernel.slice_nd(Y, [("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
                    {% if inp is not none and beta != 0 %}
                    {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, tile_Y, accum=True) }}
                    {% else %}
                    if (k2 == k_block_start) {
                        {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, tile_Y, accum=False) }}
                    } else {
                        {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, tile_Y, accum=True) }}
                    }
                    {% endif %}
                }
            }
        }
    }
}
"""


class CppPackedGemmTemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
    ):
        super().__init__("cpp_gemm", input_nodes, layout)
        self.beta = beta
        self.alpha = alpha
        self.num_threads = num_threads
        self.register_blocking = register_blocking
        m, n = layout.size
        _, k = input_nodes[0].get_size()
        self.m, self.n, self.k = m, n, k
        self.is_dynamic_M = len(self.m.free_symbols) > 0

    @cache_on_self
    def thread_blocking(self) -> GemmBlocking:
        # TODO: allow tuning various blocking options
        def get_factors(number):
            factors = []
            # priorize more evenly divided factors
            for i in range(int(number**0.5), 0, -1):
                if number % i == 0:
                    factors.append(number // i)
                    factors.append(i)
            return factors

        def get_blocking(num_threads, factor, m_blocks, n_blocks, k_blocks):
            thread_block_n = (n_blocks + factor - 1) // factor
            cofactor = num_threads // factor
            thread_block_m = (m_blocks + cofactor - 1) // cofactor
            return GemmBlocking(thread_block_m, thread_block_n, k_blocks)

        assert (
            not self.is_dynamic_M
        ), "Unable to determine thread blocking for dynamic M."
        register_blocking = self.register_blocking
        m_blocks = (self.m + register_blocking.block_m - 1) // register_blocking.block_m
        n_blocks = (self.n + register_blocking.block_n - 1) // register_blocking.block_n
        k_blocks = (self.k + register_blocking.block_k - 1) // register_blocking.block_k
        factors = get_factors(self.num_threads)
        assert len(factors) > 0
        for factor in factors:
            if n_blocks % factor == 0 and m_blocks % (self.num_threads // factor) == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
        for factor in factors:
            if n_blocks % factor == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
            cofactor = self.num_threads // factor
            if m_blocks % cofactor == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
        raise AssertionError("Should not reach here.")

    @cache_on_self
    def cache_blocking(self) -> GemmBlocking:
        # TODO: revise me
        assert (
            not self.is_dynamic_M
        ), "Unable to determine cache blocking for dynamic M."
        thread_blocking = self.thread_blocking()
        return GemmBlocking(thread_blocking.block_m, 1, thread_blocking.block_k)

    @staticmethod
    def add_choices(
        choices, layout, input_nodes, beta=1, alpha=1, trans_w=False, input_indices=None
    ):
        if input_indices is None:
            input_indices = list(range(len(input_nodes)))

        def reorder_and_filter(inputs, layout_or_out):
            if len(input_indices) == 2:
                x_idx = input_indices[0]
                w_idx = input_indices[1]
                return [inputs[x_idx], inputs[w_idx]], layout_or_out
            else:
                assert (
                    len(input_indices) == 3
                ), "Cpp Packed GEMM template requires 2 or 3 input nodes."
                # assume the input order is [inp, x, w] and we reorder it to [x, w, inp]
                inp_idx = input_indices[0]
                x_idx = input_indices[1]
                w_idx = input_indices[2]
                return [inputs[x_idx], inputs[w_idx], inputs[inp_idx]], layout_or_out

        def transpose_weight(inputs, layout_or_out):
            if not trans_w:
                return inputs, layout_or_out

            new_inputs = list(inputs)
            W = inputs[1]
            if isinstance(W, ir.IRNode):
                if not isinstance(W, ir.TensorBox):
                    W = ir.TensorBox(W)
                new_inputs[1] = permute(W, [1, 0])
                return new_inputs, layout_or_out
            else:
                assert isinstance(W, torch.Tensor)
                new_inputs[1] = W.transpose(0, 1)
            return new_inputs, layout_or_out

        num_threads = parallel_num_threads()
        new_inputs, _ = transpose_weight(*reorder_and_filter(input_nodes, layout))
        m, n, k, *_ = mm_args(new_inputs[0], new_inputs[1])
        micro_gemm = create_micro_gemm(
            "micro_gemm", m, n, k, layout.dtype, alpha=alpha, num_threads=num_threads
        )
        _, block_n, _ = micro_gemm.register_blocking

        def pack_weight(inputs, layout_or_out):
            W = inputs[1]
            new_inputs = list(inputs)
            if isinstance(W, ir.IRNode):
                if not isinstance(W, ir.TensorBox):
                    W = ir.TensorBox(W)
                k, n = W.get_size()
                assert (
                    n % block_n == 0
                ), f"The last dimension of W must be a multiple of {block_n}."
                blocked_w = permute(
                    view(W, (k, n // block_n, block_n)),
                    [1, 0, 2],
                )
                blocked_w = ir.ExternKernel.require_contiguous(blocked_w)
                blocked_w = ir.ExternKernel.realize_input(blocked_w)
            else:
                k, n = list(W.shape)
                blocked_w = (
                    W.reshape(k, n // block_n, block_n).transpose(0, 1).contiguous()
                )
            new_inputs[1] = blocked_w
            return new_inputs, layout_or_out

        def preprocessor(inputs, layout):
            return pack_weight(*transpose_weight(*reorder_and_filter(inputs, layout)))

        def postprocessor(output):
            if isinstance(output, ir.IRNode):
                # prepack the weight as input to the template buffer
                # TODO: prune the unused constants in V.graph
                # TODO: should we implement it with constant folding in the scheduler instead?
                assert isinstance(output, ir.TensorBox)
                template_buffer = ir.InputsKernel.unwrap_storage_for_input(output)
                assert isinstance(template_buffer, ir.CppTemplateBuffer)
                new_input_nodes, _ = reorder_and_filter(input_nodes, layout)
                W_node = new_input_nodes[1]
                assert W_node.get_name() in V.graph.constants
                W = V.graph.constants[W_node.get_name()]
                new_input_nodes[1] = W
                new_input_nodes, _ = pack_weight(
                    *transpose_weight(new_input_nodes, layout)
                )
                W_packed = new_input_nodes[1]
                W_packed_constant = V.graph.add_tensor_constant(W_packed)
                template_buffer.inputs[1] = ir.InputsKernel.unwrap_storage_for_input(
                    W_packed_constant
                )
            return output

        template = DataProcessorTemplateWrapper(
            CppPackedGemmTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            num_threads=num_threads,
            register_blocking=micro_gemm.register_blocking,
            beta=beta,
            alpha=alpha,
        )
        template.maybe_append_choice(choices)
        return template

    def render(  # type: ignore[override]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        **kwargs,
    ) -> str:
        assert not epilogue_nodes, "Epilogue nodes are not supported for GEMM template."
        assert len(self.input_nodes) >= 2

        X, W = self.input_nodes[0], self.input_nodes[1]
        inp = self.input_nodes[2] if len(self.input_nodes) > 2 else None
        Y = self.output_node

        if template_buffer_node is not None:
            # Use the updated prepacked weight buffer
            W = template_buffer_node.inputs[1]
            Y = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            Y = cast(Buffer, epilogue_nodes[-1])
        assert self.output_node is not None

        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            self.layout.dtype,
            alpha=self.alpha,
            num_threads=self.num_threads,
        )
        assert self.register_blocking == micro_gemm.register_blocking

        options = dict(
            X=X,
            W=W,
            inp=inp,
            Y=Y,
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            template=self,
            kernel=kernel,
            epilogues=epilogue_nodes,
        )
        return self._template_from_string(GEMM_TEMPLATE).render(**options)
