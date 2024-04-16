from typing import cast, List, Optional

from ..ir import Buffer, CppTemplateBuffer, IRNode, Layout
from .cpp_template import CppTemplate

from .cpp_template_kernel import CppTemplateKernel

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
// TODO: use micro-kernel to replace this naive GEMM implementation below
extern "C"
{{kernel.def_kernel(inputs=[X, W], outputs=[Y], names_str="X, W, Y")}}
{
    // TODO: support dynamic shapes
    int64_t M = {{kernel.size(Y, 0)}};
    int64_t N = {{kernel.size(Y, 1)}};
    int64_t K = {{kernel.size(X, 1)}};

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N/{{n_bs}}; ++j) {
            {{kernel.acc_dtype(Y)}} sum[16];
            for (int64_t ni = 0; ni < {{n_bs}}; ++ni) {
                sum[ni] = 0;
            }
            for (int64_t k = 0; k < K; ++k) {
                for (int64_t ni = 0; ni < {{n_bs}}; ++ni) {
                    sum[ni] += {{kernel.index(X, ["i", "k"])}} * {{kernel.index(W, ["j", "k", "ni"])}};
                }
            }
            for (int64_t ni = 0; ni < {{n_bs}}; ++ni) {
                int64_t n = j * {{n_bs}} + ni;
                {{kernel.index(Y, ["i", "n"])}} = sum[ni];
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
        n_block_size: int = 1,
    ):
        super().__init__("cpp_gemm", input_nodes, layout)
        self.n_block_size = n_block_size

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
        Y = self.output_node

        options = dict(
            X=X,
            W=W,
            Y=Y,
            n_bs=self.n_block_size,
            template=self,
            kernel=kernel,
            epilogues=epilogue_nodes,
        )
        return self._template_from_string(GEMM_TEMPLATE).render(**options)
