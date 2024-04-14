from typing import cast, List, Optional

from ..ir import Buffer, CppTemplateBuffer, IRNode, Layout
from .cpp_template import CppTemplate

from .cpp_template_kernel import CppTemplateKernel

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
// TODO: use micro-kernel to replace this naive GEMM implementation below
// TODO: support weight prepack
extern "C"
{{kernel.def_kernel(inputs=[X, W], outputs=[Y], names_str="X, W, Y", input_reorder=input_reorder)}}
{
    // TODO: support dynamic shapes
    int64_t M = {{kernel.size(Y, 0)}};
    int64_t N = {{kernel.size(Y, 1)}};
    int64_t K = {{kernel.size(X, 1)}};

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            {{kernel.acc_dtype(Y)}} sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += {{kernel.index(X, ["i", "k"])}} * {{kernel.index(W, ["k", "j"])}};
            }
            {{kernel.index(Y, ["i", "j"])}} = sum;
        }
    }
}
"""


class CppGemmTemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: Layout,
        input_reorder: Optional[List[int]] = None,
    ):
        super().__init__("cpp_gemm", input_nodes, layout, input_reorder)

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
            template=self,
            kernel=kernel,
            epilogues=epilogue_nodes,
            input_reorder=self.input_reorder,
        )
        return self._template_from_string(GEMM_TEMPLATE).render(**options)
