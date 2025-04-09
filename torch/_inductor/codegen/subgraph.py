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

class SubgraphChoiceCaller(ir.ChoiceCaller):
    """
    Represents a Subgraph Autotuning choice, and the subgraph can be any arbitrary
    GraphModule. Compiles the Subgraph down to a module for benchmarking.
    """
    def __init__(self, name, input_nodes, layout, description, gm, example_inputs):
        super().__init__(name, input_nodes, layout, description)
        self.gm = gm
        self.example_inputs = example_inputs

    def __str__(self) -> str:
        return f"SubgraphCaller({self.name})"

    def benchmark(self, *args, out):
        # Codegen Subgraph for benchmarking
        # Need GraphLowering instead of SubgraphLowering to generate
        # fully callable module
        from torch._inductor.graph import GraphLowering
        import torch._inductor.config as inductor_config

        bm_graph_lowering = GraphLowering(
            gm=self.gm,
            example_inputs=self.example_inputs,
            shape_env=V.graph._shape_env,
            cpp_wrapper=V.graph.cpp_wrapper,
            aot_mode=V.graph.aot_mode,
            extern_node_serializer=V.graph.extern_node_serializer,
            is_inference=V.graph.is_inference,
            is_backward=V.graph.is_backward,
            name=f"benchmark_{self.name}",
        )

        with V.set_graph_handler(bm_graph_lowering):
            # Don't bother autotuning on Triton here
            with inductor_config.patch(
                max_autotune=False,
                max_autotune_gemm=False,
                max_autotune_gemm_backends="ATEN",
            ):
                bm_graph_lowering.run(*self.example_inputs)
                mod = bm_graph_lowering.compile_to_module()
                bm_func = mod.call
                bm_func([*args])

        out.copy_(bm_func([*args])[0])

        return benchmarker.benchmark_gpu(lambda: bm_func([*args]))

    def hash_key(self):
        return "-".join(
            [
                self.name,
                *[str(arg.shape) for arg in self.example_inputs if isinstance(arg, torch.Tensor)],
                str(self.gm.graph)
            ]
        )

    
    def output_node(self):
        return ir.TensorBox.create(
            ir.SubgraphBuffer(
                layout=self.layout,
                input_nodes=self.input_nodes,
                gm=self.gm,
                example_inputs=self.example_inputs,
                subgraph_name=self.name,
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
        make_fx_graph: Callable,
    ):
        """
        Initialize a subgraph template.

        Args:
            name: The name of this template
            graph: The FX graph
        """
        self.name = name
        self.make_fx_graph = make_fx_graph

    def generate(self, input_nodes, layout, example_inputs, **kwargs: Any) -> SubgraphChoiceCaller:
        """
        Generate a SubgraphChoiceCaller instance for autotuning.

        Args:
            input_nodes: List of input nodes to the subgraph
            layout: Memory layout information for the output
            example_inputs: Example tensor inputs used to trace and benchmark the subgraph
            **kwargs: Additional keyword arguments

        Returns:
            SubgraphChoiceCaller: A callable object that can be used for autotuning
        """
        gm = self.make_fx_graph(*example_inputs)


        return SubgraphChoiceCaller(
            name=self.name,
            input_nodes=input_nodes,
            layout=layout,
            description="",
            gm=gm,
            example_inputs=example_inputs,
        )
