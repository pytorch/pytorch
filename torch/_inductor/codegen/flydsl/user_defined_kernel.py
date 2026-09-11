# mypy: allow-untyped-defs

from __future__ import annotations

from typing import Any, cast
from typing_extensions import override

import sympy

import torch
from torch._inductor import ir
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.virtualized import V


class UserDefinedFlyDSLKernel(ir.ExternKernel):
    """Lower a captured FlyDSL launcher through Python or AOTI codegen."""

    @override
    def codegen(self, wrapper) -> None:
        if V.graph.cpp_wrapper:
            from .flydsl_aot import define_aot_kernel, generate_aot_kernel_call

            artifact = define_aot_kernel(
                wrapper,
                self.launcher_idx,
                self.call_spec_idx,
                self.example_args,
            )
            generate_aot_kernel_call(
                wrapper,
                artifact,
                self.kernel_args,
                device=self.device,
                current_stream_idx=V.graph.scheduler.current_stream_idx,
            )
            return

        kernel_name = define_python_launcher(
            wrapper,
            self.launcher_idx,
            self.call_spec_idx,
        )
        from torch._higher_order_ops.flydsl_kernel_wrap import (
            flydsl_launcher_side_table,
        )

        constant_arg_indices = flydsl_launcher_side_table.get_call_spec(
            self.call_spec_idx
        )
        call_args: list[Any] = []
        arg_types: list[Any] = []
        for arg_idx, arg in enumerate(self.kernel_args):
            if arg_idx in constant_arg_indices:
                continue
            if isinstance(arg, ir.IRNode):
                call_args.append(arg.codegen_reference())
                arg_types.append(arg.get_dtype())
            elif isinstance(arg, (int, float, bool, sympy.Expr)):
                call_args.append(arg)
                arg_types.append(type(arg))
            else:
                raise NotImplementedError(
                    f"Unsupported FlyDSL launcher argument: {type(arg)}: {arg}"
                )

        self.codegen_comment(wrapper, kernel_name)
        wrapper.generate_kernel_call(
            kernel_name,
            call_args,
            arg_types=arg_types,
            triton=True,
            device=self.device,
            original_fxnode_name=self.fx_node.name,
        )

    def __init__(
        self,
        *,
        launcher_idx: int,
        call_spec_idx: int,
        kernel_args: tuple[Any, ...],
        mutated_arg_indices: tuple[int, ...],
    ) -> None:
        inputs: list[ir.IRNode] = []
        lowered_args: list[Any] = []
        input_by_arg_index: dict[int, ir.IRNode] = {}
        for idx, arg in enumerate(kernel_args):
            if isinstance(arg, ir.TensorBox):
                realized = ir.InputsKernel.unwrap_storage_for_input(
                    self.realize_input(arg)
                )
                inputs.append(realized)
                lowered_args.append(realized)
                input_by_arg_index[idx] = realized
            else:
                lowered_args.append(arg)

        if not inputs:
            raise AssertionError(
                "FlyDSL launcher must have at least one tensor argument"
            )
        self.device = inputs[0].get_device()
        super().__init__(
            None,
            ir.NoneLayout(device=self.device),
            inputs,
            tuple(arg for arg in lowered_args if not isinstance(arg, ir.IRNode)),
        )
        self.launcher_idx = launcher_idx
        self.call_spec_idx = call_spec_idx
        self.kernel_args = tuple(lowered_args)
        self.example_args = _example_args(self.kernel_args)
        self.mutation_outputs = [
            ir.MutationOutput(
                ir.NoneLayout(device=self.device),
                input_by_arg_index[idx],
                self,
            )
            for idx in mutated_arg_indices
        ]
        V.graph.register_operation(self)

    def get_outputs(self) -> list[ir.Buffer]:
        return list(self.mutation_outputs)

    def get_device(self):
        return self.device


def _example_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    examples = []
    for arg in args:
        if isinstance(arg, ir.IRNode):
            examples.append(
                cast(Any, torch).empty_strided(
                    V.graph.sizevars.optimization_hints(arg.get_size()),
                    V.graph.sizevars.optimization_hints(arg.get_stride()),
                    dtype=arg.get_dtype(),
                    device="meta",
                )
            )
        elif isinstance(arg, sympy.Expr):
            examples.append(V.graph.sizevars.optimization_hint(arg))
        else:
            examples.append(arg)
    return tuple(examples)


def lower_flydsl_kernel(
    *,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    UserDefinedFlyDSLKernel(
        launcher_idx=launcher_idx,
        call_spec_idx=call_spec_idx,
        kernel_args=args,
        mutated_arg_indices=mutated_arg_indices,
    )


def define_python_launcher(wrapper, launcher_idx: int, call_spec_idx: int) -> str:
    cache_key = ("flydsl", launcher_idx, call_spec_idx)
    cached = wrapper.user_defined_kernel_cache.get(cache_key)
    if cached is not None:
        return cached[0]

    name = f"flydsl_user_kernel_{len(wrapper.user_defined_kernel_cache)}"
    wrapper.add_import_once(
        "from torch._higher_order_ops.flydsl_kernel_wrap import FlyDSLPythonLauncher"
    )
    wrapper.header.writeline(
        f"{name} = FlyDSLPythonLauncher({launcher_idx}, {call_spec_idx})"
    )
    wrapper.user_defined_kernel_cache[cache_key] = (name, None, {})
    return name


def decompose_functional_wrapper(graph) -> None:
    from torch._higher_order_ops.flydsl_kernel_wrap import (
        flydsl_kernel_wrapper_functional,
        flydsl_kernel_wrapper_functional_dense,
    )
    from torch._inductor.fx_passes.post_grad import apply_pass_to_subgraphs

    apply_pass_to_subgraphs(decompose_functional_wrapper, graph)
    if not graph.find_nodes(
        op="call_function",
        target=flydsl_kernel_wrapper_functional,
    ):
        return
    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(flydsl_kernel_wrapper_functional),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        flat_args, spec = torch.utils._pytree.tree_flatten((args, kwargs))

        def decomp(*flat_args):
            args, kwargs = torch.utils._pytree.tree_unflatten(flat_args, spec)
            return (flydsl_kernel_wrapper_functional_dense(*args, **kwargs),)

        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    graph_pass.apply(graph)
    for _ in graph.find_nodes(
        op="call_function",
        target=flydsl_kernel_wrapper_functional,
    ):
        raise AssertionError("flydsl_kernel_wrapper_functional was not removed")
