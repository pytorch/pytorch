import logging
from typing import Any, Callable, Dict

import torch
from torch._inductor import ir
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import Buffer, Layout
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.virtualized import V


log = logging.getLogger(__name__)


class SubgraphChoiceCaller(ir.ChoiceCaller):
    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
        gm: torch.fx.GraphModule,
        example_inputs: list[Any],
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.gm: torch.fx.GraphModule = gm
        self.example_inputs: list[Any] = example_inputs

    def __str__(self) -> str:
        return f"SubgraphCaller({self.name})"

    def benchmark(self, *args: list[Any], out: torch.Tensor):
        import torch._inductor.config as inductor_config
        from torch._inductor.compile_fx import compile_fx_inner

        # Call the subgraph function with the inputs and output
        with V.fake_mode as fake_mode:
            fake_inputs = []

            for arg in self.example_inputs:
                if isinstance(arg, torch.Tensor):
                    fake_inputs.append(fake_mode.from_tensor(arg))
                else:
                    fake_inputs.append(arg)

        # Don't bother autotuning on Triton here
        with inductor_config.patch(
            max_autotune=False,
            max_autotune_gemm=False,
            max_autotune_gemm_backends="ATEN",
        ):
            benchmark_gm = compile_fx_inner(self.gm, fake_inputs)
            context = torch._guards.TracingContext.try_get()

            if context:
                # Reset output strides with nested compilation
                context.output_strides = []

        out.copy_(benchmark_gm([*args])[0])

        return benchmarker.benchmark_gpu(lambda: benchmark_gm([*args]))

    def hash_key(self) -> str:
        return "-".join(
            [
                self.name,
                *[
                    str(arg.shape)
                    for arg in self.example_inputs
                    if isinstance(arg, torch.Tensor)
                ],
            ]
        )

    def output_node(self) -> ir.TensorBox:
        return ir.TensorBox.create(
            ir.SubgraphBuffer(
                layout=self.layout,
                input_nodes=self.input_nodes,
                gm=self.gm,
                example_inputs=self.example_inputs,
            )
        )

    def info_dict(self) -> Dict[str, Any]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "subgraph",
            "kernel_name": self.name,
        }

    def autoheuristic_id(self) -> str:
        return f"subgraph_{self.name}"


class SubgraphTemplate(KernelTemplate):
    """
    A template for subgraph evaluation to be used in autotuning.

    This class allows creating customized subgraphs that can be appended
    as choices during the autotuning process, enabling the selection of
    optimal implementations for complex operations.
    """

    def __init__(
        self,
        name: str,
        make_fx_graph: Callable[..., Any],
    ):
        """
        Initialize a subgraph template.

        Args:
            name: The name of this template
            graph: The FX graph
        """
        self.name = name
        self.make_fx_graph = make_fx_graph

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        example_inputs: list[Any],
        **kwargs: Any,
    ) -> SubgraphChoiceCaller:
        gm = self.make_fx_graph(*example_inputs)

        return SubgraphChoiceCaller(
            name=self.name,
            input_nodes=input_nodes,
            layout=layout,
            description="",
            gm=gm,
            example_inputs=example_inputs,
        )
