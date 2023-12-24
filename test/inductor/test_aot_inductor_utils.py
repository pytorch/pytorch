# Owner(s): ["module: inductor"]
import tempfile

import torch
import torch._export
import torch._inductor
import torch.fx._pytree as fx_pytree
from torch._inductor.utils import aot_inductor_launcher, cache_dir

from torch.testing._internal.common_utils import IS_FBCODE

from torch.utils import _pytree as pytree


class AOTInductorModelRunner:
    @classmethod
    def compile(
        cls,
        model,
        example_inputs,
        options=None,
        constraints=None,
        disable_constraint_solver=False,
    ):
        # The exact API is subject to change
        so_path = torch._export.aot_compile(
            model,
            example_inputs,
            options=options,
            constraints=constraints,
            remove_runtime_assertions=True,
            disable_constraint_solver=disable_constraint_solver,
        )
        return so_path

    @classmethod
    def load(cls, device, so_path, example_inputs):
        if IS_FBCODE:
            from .fb import test_aot_inductor_model_runner_pybind

            module = test_aot_inductor_model_runner_pybind.Runner(
                so_path, device == "cpu"
            )

            call_spec = module.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])

            def optimized(*args):
                flat_inputs = fx_pytree.tree_flatten_spec((*args, {}), in_spec)
                flat_outputs = module.run(flat_inputs)
                return pytree.tree_unflatten(flat_outputs, out_spec)

        else:
            module = torch.utils.cpp_extension.load_inline(
                name="aot_inductor",
                cpp_sources=[aot_inductor_launcher(so_path, device)],
                # use a unique build directory to avoid test interference
                build_directory=tempfile.mkdtemp(dir=cache_dir()),
                functions=["run", "get_call_spec"],
                with_cuda=(device == "cuda"),
            )

            call_spec = module.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])

            def optimized(*args):
                flat_inputs = fx_pytree.tree_flatten_spec((*args, {}), in_spec)
                flat_outputs = module.run(flat_inputs)
                return pytree.tree_unflatten(flat_outputs, out_spec)

        return optimized

    @classmethod
    def run(
        cls,
        device,
        model,
        example_inputs,
        options=None,
        constraints=None,
        disable_constraint_solver=False,
    ):
        so_path = AOTInductorModelRunner.compile(
            model,
            example_inputs,
            options=options,
            constraints=constraints,
            disable_constraint_solver=disable_constraint_solver,
        )
        optimized = AOTInductorModelRunner.load(device, so_path, example_inputs)
        return optimized(example_inputs)

    @classmethod
    def run_multiple(
        cls,
        device,
        model,
        list_example_inputs,
        options=None,
        constraints=None,
    ):
        so_path = AOTInductorModelRunner.compile(
            model,
            list_example_inputs[0],
            options=options,
            constraints=constraints,
        )
        optimized = AOTInductorModelRunner.load(device, so_path, list_example_inputs[0])
        list_output_tensors = []
        for example_inputs in list_example_inputs:
            list_output_tensors.append(optimized(example_inputs))
        return list_output_tensors
