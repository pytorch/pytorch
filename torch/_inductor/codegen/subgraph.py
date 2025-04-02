import logging

from typing import Any, Callable

import torch

from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state

from torch._inductor import ir
from torch._inductor.virtualized import V

from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.runtime.benchmarking import benchmarker


log = logging.getLogger(__name__)

class SubgraphChoiceCaller(ir.ChoiceCaller):
    def __init__(self, name, input_nodes, layout, description, gm, real_inputs):
        super().__init__(name, input_nodes, layout, description)
        self.gm = gm
        self.real_inputs = real_inputs

    def __str__(self) -> str:
        return f"SubgraphCaller({self.name})"

    def benchmark(self, *args, out):
        from torch._inductor.compile_fx import compile_fx_inner
        import torch._inductor.config as inductor_config
        # Call the subgraph function with the inputs and output

        with V.fake_mode as fake_mode:
            fake_ten_1, fake_ten_2 = fake_mode.from_tensor(self.real_inputs[0]), fake_mode.from_tensor(self.real_inputs[1])
    
        # Don't bother autotuning on Triton here
        with inductor_config.patch(
            max_autotune=False,
            max_autotune_gemm=False,
            max_autotune_gemm_backends="ATEN",
        ):
            benchmark_gm = compile_fx_inner(self.gm, (fake_ten_1, fake_ten_2, self.real_inputs[2]))
            context = torch._guards.TracingContext.try_get()

            # Reset output strides with nested compilation
            context.output_strides = []

        benchmark_gm([*args, self.real_inputs[2]])

        # inductor benchmarker
        return benchmarker.benchmark_gpu(lambda: benchmark_gm([*args, self.real_inputs[2]]))

    def hash_key(self):
        return "-".join(
            [
                self.name,
                *[str(arg.shape) for arg in self.real_inputs if isinstance(arg, torch.Tensor)]
            ]
        )
    def output_node(self):
        return ir.TensorBox.create(
            ir.SubgraphBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                gm=self.gm,
                real_inputs=self.real_inputs,
            )
        )

    def info_dict(self):
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "subgraph",
            "kernel_name": self.name,
        }

    def autoheuristic_id(self):
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
        graph: Callable,
    ):
        """
        Initialize a subgraph template.

        Args:
            name: The name of this template
            graph: The FX graph
        """
        self.name = name
        self.graph = graph

    def generate(self, input_nodes, layout, **kwargs: Any) -> SubgraphChoiceCaller:
        def generate_inputs(node) -> torch.Tensor:
            size = V.graph.sizevars.size_hints(
                node.get_size(),
            )
            stride = V.graph.sizevars.size_hints(
                node.get_stride(),
            )
            device = node.get_device()
            dtype = node.get_dtype()
            extra_size = node.layout.offset
            allocation_size = V.graph.sizevars.size_hints(
                V.graph.get_allocation_size(node),
            )

            with preserve_rng_state():
                if allocation_size is None or allocation_size == size:
                    return rand_strided(
                        size,
                        stride,
                        device=device,
                        dtype=dtype,
                        extra_size=extra_size,
                    )
                else:
                    return rand_strided(
                        allocation_size,
                        stride,
                        device=device,
                        dtype=dtype,
                        extra_size=extra_size,
                    ).as_strided(size, stride)

        mat_1, mat_2 = input_nodes
        # TODO: torch.compile subgraph for optimal fusions
        # TODO: Tune on kPartitions, by default is 32
        ten_1, ten_2 = generate_inputs(mat_1), generate_inputs(mat_2)

        gm = self.graph(ten_1, ten_2, 256)
        

        return SubgraphChoiceCaller(
            name=self.name,
            input_nodes=input_nodes,
            layout=layout,
            description="",
            gm=gm,
            real_inputs=[ten_1, ten_2, 256],
        )
