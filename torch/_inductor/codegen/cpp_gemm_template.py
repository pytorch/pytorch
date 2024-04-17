from typing import cast, List, Optional

import torch
from torch._inductor.select_algorithm import DataProcessorTemplateWrapper
from .. import ir

from ..ir import Buffer, CppTemplateBuffer, IRNode, Layout
from ..lowering import permute, view
from .cpp_template import CppTemplate

from .cpp_template_kernel import CppTemplateKernel

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
// TODO: use micro-kernel to replace this naive GEMM implementation below
extern "C"
{{kernel.def_kernel(inputs=[X, W, inp], outputs=[Y], names_str="X, W, inp, Y")}}
{
    // TODO: support >2D tensors
    int64_t M = {{kernel.size(Y, 0)}};
    int64_t N = {{kernel.size(Y, 1)}};
    int64_t K = {{kernel.size(X, 1)}};

    #pragma omp parallel for collapse(2)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t j = 0; j < N/{{n_bs}}; ++j) {
            {{kernel.acc_dtype(Y)}} sum[16];
            for (int64_t ni = 0; ni < {{n_bs}}; ++ni) {
            {% if inp is none %}
                sum[ni] = 0;
            {% else %}
                int64_t n = j * {{n_bs}} + ni;
                sum[ni] = {{beta}} * {{kernel.index(inp, ["m", "n"])}};
            {% endif %}
            }
            for (int64_t k = 0; k < K; ++k) {
                for (int64_t ni = 0; ni < {{n_bs}}; ++ni) {
                    sum[ni] += {{kernel.index(X, ["m", "k"])}} * {{kernel.index(W, ["j", "k", "ni"])}};
                }
            }
            for (int64_t ni = 0; ni < {{n_bs}}; ++ni) {
                int64_t n = j * {{n_bs}} + ni;
                {{kernel.index(Y, ["m", "n"])}} = {{alpha}} * sum[ni];
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
        beta=1,
        alpha=1,
        n_block_size: int = 1,
    ):
        super().__init__("cpp_gemm", input_nodes, layout)
        self.beta = beta
        self.alpha = alpha
        self.n_block_size = n_block_size

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

        n_block_size = 16

        def pack_weight(inputs, layout_or_out):
            W = inputs[1]
            new_inputs = list(inputs)
            if isinstance(W, ir.IRNode):
                if not isinstance(W, ir.TensorBox):
                    W = ir.TensorBox(W)
                k, n = W.get_size()
                assert (
                    n % n_block_size == 0
                ), f"The last dimension of W must be a multiple of {n_block_size}."
                blocked_w = permute(
                    view(W, (k, n // n_block_size, n_block_size)),
                    [1, 0, 2],
                )
                blocked_w = ir.ExternKernel.require_contiguous(blocked_w)
                blocked_w = ir.ExternKernel.realize_input(blocked_w)
            else:
                k, n = list(W.shape)
                blocked_w = (
                    W.reshape(k, n // n_block_size, n_block_size)
                    .transpose(0, 1)
                    .contiguous()
                )
            new_inputs[1] = blocked_w
            return new_inputs, layout_or_out

        def preprocessor(inputs, layout):
            return pack_weight(*transpose_weight(*reorder_and_filter(inputs, layout)))

        template = DataProcessorTemplateWrapper(
            CppPackedGemmTemplate,
            preprocessor,
            None,
            input_nodes=input_nodes,
            layout=layout,
            n_block_size=n_block_size,
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

        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])
        assert self.output_node is not None

        X, W = self.input_nodes[0], self.input_nodes[1]
        inp = self.input_nodes[2] if len(self.input_nodes) > 2 else None
        Y = self.output_node

        options = dict(
            X=X,
            W=W,
            inp=inp,
            Y=Y,
            beta=self.beta,
            alpha=self.alpha,
            n_bs=self.n_block_size,
            template=self,
            kernel=kernel,
            epilogues=epilogue_nodes,
        )
        return self._template_from_string(GEMM_TEMPLATE).render(**options)
