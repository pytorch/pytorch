import itertools
import logging
from collections.abc import Callable
from typing import Any, Union

import torch
import torch._inductor.config as config
from torch._inductor import ir
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import (
    Buffer,
    FixedLayout,
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


def inline_subgraph_to_ir_nodes(
    gm: torch.fx.GraphModule, inputs: list[Any], name: str
) -> Any:
    """Inline a subgraph by converting its FX operations to individual IR nodes.

    This converts a subgraph to multiple ComputedBuffer nodes (fusable),
    enabling epilogue fusion with subsequent operations.

    Returns:
        TensorBox containing the final operation result as individual IR nodes
    """
    from torch._inductor.lowering import process_subgraph_nodes

    return process_subgraph_nodes(gm, inputs)


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
            # pyrefly: ignore [no-matching-overload]
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
        return benchmarker.benchmark(
            # Shallow clone args since bm_func may clear args
            lambda: bm_func([*sym_inputs, *args]),
            device=benchmarker.infer_device(*sym_inputs, *args),
        )

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
            name: The name for this subgraph choice
            input_nodes: List of input nodes to the subgraph
            layout: Memory layout information for the output
            make_fx_graph: Callable that creates the FX graph for this subgraph
            description: Optional description of this choice
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
        non_tensor_args: list[dict[str, Any]],
        default_impl: Callable[..., Any] | None = None,
    ) -> list[SubgraphChoiceCaller]:
        """
        Generate multiple SubgraphChoiceCaller instances for custom op autotuning.

        This method extends SubgraphTemplate to support custom op decompositions,
        allowing multiple implementations to compete in autotuning.

        Args:
            name: Base name for the choices
            decompositions: List of decomposition functions to compete in autotuning
            input_nodes: List of tensor inputs. All tensor arguments must be passed here.
            non_tensor_args: List of non-tensor kwargs only, one dict per corresponding decomposition.
            default_impl: Default implementation for layout inference

        Returns:
            List of SubgraphChoiceCaller instances for autotuning
        """
        if not decompositions:
            return []

        assert len(decompositions) == len(non_tensor_args), (
            f"decompositions and non_tensor_args must have same length, "
            f"got {len(decompositions)} decompositions and {len(non_tensor_args)} kwargs"
        )

        # Infer layouts and ensure layout consistency for fair autotuning comparison
        layouts = [
            self._infer_custom_op_layout(input_nodes, decomp, kwargs, default_impl)
            for decomp, kwargs in zip(decompositions, non_tensor_args)
        ]

        # Validate all decompositions produce equivalent layouts for fair comparison
        self._validate_layout_equivalence(name, decompositions, layouts)
        layout = layouts[0]  # All layouts are now validated to be equivalent

        choices: list[SubgraphChoiceCaller] = []
        for decomp, decomp_kwargs in zip(decompositions, non_tensor_args):
            # Create make_fx_graph function for this decomposition
            import functools

            def make_fx_graph(
                *args: Any,
                decomp: Callable[..., Any] = decomp,
                decomp_kwargs: dict[str, Any] = decomp_kwargs,
            ) -> Any:
                # decomp_kwargs contains all merged parameters: CustomOpConfig params + runtime kwargs
                from torch.fx.experimental.proxy_tensor import make_fx

                from ..decomposition import select_decomp_table

                decomposition_table = select_decomp_table()

                return make_fx(
                    functools.partial(decomp, **decomp_kwargs),
                    decomposition_table=decomposition_table,
                )(*args)

            # Generate descriptive name for this variant
            variant_name = self._generate_variant_name(decomp, decomp_kwargs)

            choice = self.generate(
                name=f"{name}_{variant_name}",
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=make_fx_graph,
                description=f"CustomOp {decomp.__name__}",
            )
            choices.append(choice)

        return choices

    def _generate_variant_name(
        self, decomp: Callable[..., Any], kwargs: dict[str, Any]
    ) -> str:
        """Generate a descriptive name for a decomposition variant with its parameters."""
        base_name = decomp.__name__
        if not kwargs:
            return base_name
        param_suffix = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
        return f"{base_name}_{param_suffix}"

    def _validate_non_tensor_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate that kwargs contains only non-tensor arguments."""
        for key, value in kwargs.items():
            assert not isinstance(value, (torch.Tensor, Buffer)), (
                f"kwargs['{key}'] contains tensor {type(value)}. "
                f"Tensor arguments should be in input_nodes, not kwargs. "
                f"Only scalar/non-tensor parameters should be in kwargs."
            )

    def _validate_layout_equivalence(
        self,
        op_name: str,
        decompositions: list[Callable[..., Any]],
        layouts: list[Layout],
    ) -> None:
        """Ensure all layouts have consistent stride, device, dtype, and sizes for fair autotuning."""
        if not layouts:
            return

        reference = layouts[0]
        for i, layout in enumerate(layouts[1:], start=1):
            if (layout.device, layout.dtype, layout.size, layout.stride) != (
                reference.device,
                reference.dtype,
                reference.size,
                reference.stride,
            ):
                raise AssertionError(
                    f"Layout mismatch in custom op '{op_name}': "
                    f"decomposition '{decompositions[i].__name__}' produces "
                    f"({layout.device}, {layout.dtype}, {layout.size}, {layout.stride}) "
                    f"but '{decompositions[0].__name__}' produces "
                    f"({reference.device}, {reference.dtype}, {reference.size}, {reference.stride})"
                )

    def _infer_custom_op_layout(
        self,
        input_nodes: list[Buffer],
        function_decomposition: Callable[..., Any],
        kwargs: dict[str, Any],
        default_impl: Callable[..., Any] | None = None,
    ) -> Layout:
        """Infer output layout for custom ops using the default implementation when available.
        Note that the Subgraph assumes custom ops return exactly one tensor output.
        TODO: Add support for multiple output custom ops.
        """
        import functools

        from torch._inductor.virtualized import V

        # Assert kwargs contain only non-tensor arguments
        self._validate_non_tensor_kwargs(kwargs)

        with V.fake_mode:
            example_inputs = []
            for inp in input_nodes:
                raw_shape = inp.get_size()
                concrete_shape = V.graph.sizevars.size_hints(
                    raw_shape, fallback=config.unbacked_symint_fallback
                )
                fake_tensor = torch.empty(
                    concrete_shape, dtype=inp.get_dtype(), device=inp.get_device()
                )
                example_inputs.append(fake_tensor)

            fn = functools.partial(function_decomposition, **kwargs)
            output = fn(*example_inputs)

            # Assert single output
            assert isinstance(output, torch.Tensor), (
                f"Expected single tensor output, got {type(output)}. "
                f"Multi-output custom ops not yet supported in autotuning."
            )

            return FixedLayout(
                device=output.device,
                dtype=output.dtype,
                size=output.shape,
                stride=output.stride(),
            )
