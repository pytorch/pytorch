import itertools
import logging
from typing import Any, Callable, Union

import torch
import torch._inductor.config as config
from torch._inductor import ir
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import (
    Buffer,
    get_free_symbols,
    get_symbolic_inputs,
    gm_original_output_strides,
    ir_node_to_tensor,
    Layout,
)
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.utils import do_bench_using_profiling
from torch._inductor.virtualized import V


log = logging.getLogger(__name__)


class SubgraphChoiceCaller(ir.ChoiceCaller):
    """
    Represents a Subgraph Autotuning choice, and the subgraph can be any arbitrary
    GraphModule. Compiles the Subgraph down to a module for benchmarking.
    """

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
        make_fx_graph: Callable[..., Any],
    ) -> None:
        super().__init__(name, input_nodes, layout, description)

        self.example_inputs = []
        with V.fake_mode:
            for inp in self.input_nodes:
                # Here there will be no unbacked symbols, as SubgraphBuffer does not support them
                assert len(get_free_symbols(inp.get_size(), unbacked_only=True)) == 0
                assert len(get_free_symbols(inp.get_stride(), unbacked_only=True)) == 0

                inp.data.freeze_layout()  # type: ignore[attr-defined]
                self.example_inputs.append(ir_node_to_tensor(inp))

        self.gm = make_fx_graph(*self.example_inputs)
        gm_original_output_strides(self.gm)

        self.sym_inputs = get_symbolic_inputs(self.input_nodes)

    def __str__(self) -> str:
        return f"SubgraphCaller({self.name})"

    def benchmark(self, *args: list[Any], out: torch.Tensor) -> float:
        # Codegen Subgraph for benchmarking
        # Need GraphLowering instead of SubgraphLowering to generate
        # fully callable module
        import torch._inductor.config as inductor_config
        from torch._inductor.graph import GraphLowering

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

        for sym_inp in self.sym_inputs:
            bm_graph_lowering.graph_inputs[sym_inp.name] = sym_inp
            bm_graph_lowering.graph_input_names.append(sym_inp.name)

        sym_inputs = [
            int(V.graph.sizevars.shape_env.size_hint(sym_var))
            for sym_var in self.sym_inputs
        ]

        if len(sym_inputs) == 0:
            # Sanity check that args are same layout as example inputs
            # Only do it if there are no symbolic inputs, otherwise
            # the dynamic dim will be realized to the same size as args
            for ar, example_inp in zip(args, self.example_inputs):
                # Sanity check that args are same layout as example inputs
                if isinstance(ar, torch.Tensor):
                    assert isinstance(example_inp, torch.Tensor)
                    assert ar.shape == example_inp.shape
                    assert ar.stride() == example_inp.stride()

        if len(sym_inputs) == 0:
            # Sanity check that args are same layout as example inputs
            # Only do it if there are no symbolic inputs, otherwise
            # the dynamic dim will be realized to the same size as args
            for ar, example_inp in zip(args, self.example_inputs):
                # Sanity check that args are same layout as example inputs
                if isinstance(ar, torch.Tensor):
                    assert isinstance(example_inp, torch.Tensor)
                    assert ar.shape == example_inp.shape
                    assert ar.stride() == example_inp.stride()

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

                bm_func([*sym_inputs, *args])
        if config.profile_bandwidth_with_do_bench_using_profiling:
            return do_bench_using_profiling(lambda: bm_func([*sym_inputs, *args]))
        return benchmarker.benchmark_gpu(lambda: bm_func([*sym_inputs, *args]))

    def hash_key(self) -> str:
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                *[str(inp.get_size()) for inp in self.input_nodes],
                *[str(inp.get_stride()) for inp in self.input_nodes],
                str(self.gm.graph),
            ]
        )

    def output_node(self) -> Union[ir.TensorBox, ir.ShapeAsConstantBuffer]:
        return ir.TensorBox.create(
            ir.SubgraphBuffer(
                layout=self.layout,
                input_nodes=self.input_nodes,
                gm=self.gm,
                example_inputs=self.example_inputs,
                subgraph_name=self.name,
            )
        )

    def info_dict(self) -> dict[str, Any]:
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

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
    ):
        """
        Initialize a subgraph template.

        Args:
            name: The name of this template
            graph: The FX graph
        """
        super().__init__(name=name)

    def generate(  # type: ignore[override]
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_fx_graph: Callable[..., Any],
        description: str = "",
        **kwargs: Any,
    ) -> SubgraphChoiceCaller:
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

        return SubgraphChoiceCaller(
            name=f"{name}_{next(SubgraphTemplate.index_counter)}",
            input_nodes=input_nodes,
            layout=layout,
            description=description,
            make_fx_graph=make_fx_graph,
        )
