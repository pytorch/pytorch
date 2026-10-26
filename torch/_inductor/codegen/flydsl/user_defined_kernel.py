# mypy: allow-untyped-defs
from __future__ import annotations

from typing import Any, cast
from typing_extensions import override

import sympy

import torch
from torch._inductor import ir
from torch._higher_order_ops.flydsl_program import FlyDSLProgramSpec
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.virtualized import V


class UserDefinedFlyDSLKernel(ir.ExternKernel):
    """A portable user launcher lowered as one FlyDSL AOT external kernel."""

    @override
    def codegen(self, wrapper) -> None:
        if not V.graph.cpp_wrapper:
            raise NotImplementedError(
                "portable FlyDSL launchers currently require an AOTI C++ wrapper"
            )
        from .flydsl_aot import (
            compile_portable_program,
            FlyDSLAOTBinding,
            generate_aot_kernel_call,
            register_aot_artifact,
        )

        artifact = compile_portable_program(
            self.program_json,
            self.call_spec_json,
            self.example_args,
        )
        kernel_name = (
            "flydsl_portable_" f"{artifact.manifest.specialization_fingerprint[:24]}"
        )
        register_aot_artifact(wrapper, kernel_name, artifact)
        binding = FlyDSLAOTBinding(
            dict(zip(self.program.parameter_names, self.kernel_args))
        )
        generate_aot_kernel_call(
            wrapper,
            kernel_name,
            binding,
            device=self.device,
        )

    def __init__(
        self,
        *,
        program_json: str,
        call_spec_json: str,
        kernel_args: tuple[Any, ...],
        mutated_arg_indices: tuple[int, ...],
    ) -> None:
        inputs: list[ir.IRNode] = []
        lowered_args: list[Any] = []
        input_by_arg_index: dict[int, ir.IRNode] = {}
        for index, arg in enumerate(kernel_args):
            if isinstance(arg, ir.TensorBox):
                realized = ir.InputsKernel.unwrap_storage_for_input(
                    self.realize_input(arg)
                )
                inputs.append(realized)
                lowered_args.append(realized)
                input_by_arg_index[index] = realized
            else:
                lowered_args.append(arg)
        if not inputs:
            raise AssertionError(
                "portable FlyDSL launchers require at least one tensor"
            )
        self.device = inputs[0].get_device()
        super().__init__(
            None,
            ir.NoneLayout(device=self.device),
            inputs,
            tuple(arg for arg in lowered_args if not isinstance(arg, ir.IRNode)),
        )
        self.program_json = program_json
        self.call_spec_json = call_spec_json
        self.program = FlyDSLProgramSpec.from_json(program_json)
        self.kernel_args = tuple(lowered_args)
        self.example_args = _example_args(self.kernel_args)
        self.mutation_outputs = [
            ir.MutationOutput(
                ir.NoneLayout(device=self.device),
                input_by_arg_index[index],
                self,
            )
            for index in mutated_arg_indices
        ]
        V.graph.register_operation(self)

    def get_outputs(self) -> list[ir.Buffer]:
        return list(self.mutation_outputs)

    def get_device(self):
        return self.device


def _example_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    result = []
    for arg in args:
        if isinstance(arg, ir.IRNode):
            result.append(
                cast(Any, torch).empty_strided(
                    V.graph.sizevars.optimization_hints(arg.get_size()),
                    V.graph.sizevars.optimization_hints(arg.get_stride()),
                    dtype=arg.get_dtype(),
                    device="meta",
                )
            )
        elif isinstance(arg, sympy.Expr):
            result.append(V.graph.sizevars.optimization_hint(arg))
        else:
            result.append(arg)
    return tuple(result)


def lower_portable_flydsl(
    *,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    UserDefinedFlyDSLKernel(
        program_json=program_json,
        call_spec_json=call_spec_json,
        kernel_args=args,
        mutated_arg_indices=mutated_arg_indices,
    )


def decompose_portable_functional(graph) -> None:
    from torch._higher_order_ops.flydsl_portable_call import (
        flydsl_portable_functional,
        flydsl_portable_functional_dense,
    )
    from torch._inductor.fx_passes.post_grad import apply_pass_to_subgraphs

    apply_pass_to_subgraphs(decompose_portable_functional, graph)
    if not graph.find_nodes(
        op="call_function",
        target=flydsl_portable_functional,
    ):
        return
    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(flydsl_portable_functional),
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        flat_args, spec = torch.utils._pytree.tree_flatten((args, kwargs))

        def decomp(*flat):
            call_args, call_kwargs = torch.utils._pytree.tree_unflatten(flat, spec)
            return (flydsl_portable_functional_dense(*call_args, **call_kwargs),)

        match.replace_by_example(
            decomp,
            flat_args,
            run_functional_passes=False,
        )

    graph_pass.apply(graph)
