import itertools
import logging
from typing import Any, Callable, Optional, Union

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
            # pyrefly: ignore  # no-matching-overload
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

        # Use appropriate benchmarker based on layout device type
        if self.layout.device.type == "cpu":
            return benchmarker.benchmark_cpu(lambda: bm_func([*sym_inputs, *args]))
        else:
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

    def generate_custom_op_choices(
        self,
        name: str,
        decompositions: list[Callable[..., Any]],
        input_nodes: list[Buffer],
        kwargs: Optional[dict[str, Any]] = None,
        default_impl: Optional[Callable[..., Any]] = None,
    ) -> list[SubgraphChoiceCaller]:
        """
        Generate multiple SubgraphChoiceCaller instances for custom op autotuning.

        This method extends SubgraphTemplate to support custom op decompositions,
        allowing multiple implementations to compete in autotuning.

        Args:
            name: Base name for the choices
            decompositions: List of decomposition functions to compare
            input_nodes: Input nodes for the operation
            kwargs: Additional arguments for decomposition functions
            default_impl: Default implementation for layout inference

        Returns:
            List of SubgraphChoiceCaller instances for autotuning
        """
        if not decompositions:
            return []

        kwargs = kwargs or {}

        # Infer layouts and ensure stride consistency for fair autotuning comparison
        layouts = [
            self._infer_custom_op_layout(input_nodes, [decomp], kwargs, default_impl)
            for decomp in decompositions
        ]

        self._validate_stride_consistency(name, decompositions, layouts)
        layout = layouts[0]  # All layouts have equivalent stride now

        choices = []
        for decomp in decompositions:
            # Create make_fx_graph function for this decomposition
            def make_fx_graph(*args: Any, decomp: Callable[..., Any] = decomp) -> Any:
                import functools

                from torch.fx.experimental.proxy_tensor import make_fx

                # Ensure kwargs is not None for unpacking
                decomp_kwargs = kwargs if kwargs is not None else {}
                return make_fx(functools.partial(decomp, **decomp_kwargs))(*args)

            choice = self.generate(
                name=f"{name}_{decomp.__name__}",
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=make_fx_graph,
                description=f"CustomOp {decomp.__name__}",
            )
            choices.append(choice)

        return choices

    def _validate_stride_consistency(
        self,
        op_name: str,
        decompositions: list[Callable[..., Any]],
        layouts: list[Layout],
    ) -> None:
        """Ensure all decompositions produce compatible strides for fair autotuning."""
        if not layouts:
            return

        strides = [layout.stride for layout in layouts]
        reference = strides[0]
        for i, stride in enumerate(strides[1:]):
            if stride != reference:
                raise AssertionError(
                    f"Stride mismatch in custom op '{op_name}' autotuning: "
                    f"'{decompositions[i].__name__}' produces stride {stride}, "
                    f"but '{decompositions[0].__name__}' produces {reference}. "
                    f"All decompositions must have identical output strides."
                )

    def _infer_custom_op_layout(
        self,
        input_nodes: list[Buffer],
        decompositions: list[Callable[..., Any]],
        kwargs: dict[str, Any],
        default_impl: Optional[Callable[..., Any]] = None,
    ) -> Layout:
        """Infer output layout for custom ops using the default implementation when available."""
        import functools

        from torch._inductor.ir import FixedLayout, ir_node_to_tensor
        from torch._inductor.virtualized import V

        # Use default_impl if available, otherwise use first decomposition
        impl = default_impl if default_impl is not None else decompositions[0]

        with V.fake_mode:
            example_inputs = [ir_node_to_tensor(inp) for inp in input_nodes]
            fn = functools.partial(impl, **kwargs)
            output = fn(*example_inputs)

            return FixedLayout(
                device=output.device,
                dtype=output.dtype,
                size=output.shape,
                stride=output.stride(),
            )
