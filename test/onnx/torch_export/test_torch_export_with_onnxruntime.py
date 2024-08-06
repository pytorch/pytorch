# Owner(s): ["module: onnx"]
from __future__ import annotations

import os
import sys

import torch
import torch.onnx
from torch.testing._internal import common_utils
from torch.utils import _pytree as torch_pytree


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import onnx_test_common


class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    def _compare_onnx_and_torch_exported_program(
        self,
        torch_exported_program,
        onnx_exported_program,
        input_args,
        input_kwargs=None,
        rtol=1e-03,
        atol=1e-07,
    ):
        # avoid mutable default argument
        if input_kwargs is None:
            input_kwargs = {}

        # NOTE: ONNXProgram holds a reference (not copy) to the original ref_model, including its state_dict.
        # Thus, ONNXProgram() must run before ref_model() to prevent ref_model.forward() from changing the state_dict.
        # Otherwise, the ref_model can change buffers on state_dict which would be used by ONNXProgram.__call__()
        onnx_outputs = onnx_exported_program(*input_args, **input_kwargs)
        if isinstance(torch_exported_program, torch.export.ExportedProgram):
            torch_outputs = torch_exported_program.module()(*input_args, **input_kwargs)
        else:
            torch_outputs = torch_exported_program(*input_args, **input_kwargs)
        torch_outputs_onnx_format = onnx_exported_program.adapt_torch_outputs_to_onnx(
            torch_outputs
        )
        if len(torch_outputs_onnx_format) != len(onnx_outputs):
            raise AssertionError(
                f"Expected {len(torch_outputs_onnx_format)} outputs, got {len(onnx_outputs)}"
            )
        for torch_output, onnx_output in zip(torch_outputs_onnx_format, onnx_outputs):
            torch.testing.assert_close(
                torch_output, torch.tensor(onnx_output), rtol=rtol, atol=atol
            )

    def test_exported_program_with_dynamic_input(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        x = torch.randn(2, 3, 4, dtype=torch.float)
        dim0 = torch.export.Dim("dim0")
        exported_program = torch.export.export(
            Model(), (x,), dynamic_shapes={"x": {0: dim0}}
        )
        onnx_program = torch.onnx.dynamo_export(exported_program, x)

        # different dim inputs
        y = torch.randn(3, 3, 4, dtype=torch.float)
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, input_args=(y,)
        )

    def test_exported_program_as_input_from_file(self):
        import tempfile

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        x = torch.randn(1, 1, 2, dtype=torch.float)
        exported_program = torch.export.export(Model(), args=(x,))
        onnx_program = torch.onnx.dynamo_export(exported_program, x)

        with tempfile.NamedTemporaryFile(suffix=".pte") as f:
            torch.export.save(exported_program, f.name)
            del exported_program  # Delete the exported program to ensure that we are loading from file
            loaded_exported_program = torch.export.load(f.name)

        self._compare_onnx_and_torch_exported_program(
            loaded_exported_program, onnx_program, input_args=(x,)
        )

    def test_exported_program_with_specialized_input_during_tracing(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        f = Foo()

        tensor_input = torch.ones(7, 5)
        dim0_x = torch.export.Dim("dim0_x", min=6)
        dynamic_shapes = {"x": {0: dim0_x}, "y": None}
        # specialized input y to 5 during tracing
        exported_program = torch.export.export(
            f, (tensor_input, 5), dynamic_shapes=dynamic_shapes
        )
        onnx_program = torch.onnx.dynamo_export(exported_program, tensor_input, 5)

        # different dim inputs
        additional_tensor_input = torch.ones(8, 5)
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, input_args=(additional_tensor_input, 5)
        )

    def test_onnx_program_supports_retraced_graph(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        tensor_input = torch.ones(5, 5)
        exported_program = torch.export.export(Foo(), (tensor_input,))

        dim0_x = torch.export.Dim("dim0_x")
        # NOTE: If input is ExportedProgram, we need to specify dynamic_shapes
        # as a tuple.
        reexported_program = torch.export.export(
            exported_program.module(), (tensor_input,), dynamic_shapes=({0: dim0_x},)
        )
        reexported_onnx_program = torch.onnx.dynamo_export(
            reexported_program, tensor_input
        )

        additional_tensor_input = torch.ones(7, 5)
        self._compare_onnx_and_torch_exported_program(
            reexported_program,
            reexported_onnx_program,
            input_args=(additional_tensor_input,),
        )

    def test_onnx_program_supports_none_arg_name_in_dynamic(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b):
                return a.sum() + b.sum()

        foo = Foo()

        dim = torch.export.Dim("dim")
        exported_program = torch.export.export(
            foo, (torch.randn(4, 4), torch.randn(4, 4)), dynamic_shapes=(None, {0: dim})
        )
        onnx_program = torch.onnx.dynamo_export(
            exported_program, torch.randn(4, 4), torch.randn(4, 4)
        )

        test_inputs = (
            torch.randn(4, 4),
            torch.randn(7, 4),
        )
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, test_inputs
        )

    def test_onnx_program_suppors_non_arg_name_with_kwarg(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b, kw1, kw2):
                return a.sum() + b.sum() + kw1.sum() - kw2.sum()

        foo = Foo()

        dim = torch.export.Dim("dim")
        dim_for_kw1 = torch.export.Dim("dim_for_kw1")
        exported_program = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            {"kw2": torch.ones(4, 4), "kw1": torch.zeros(4, 4)},
            # We are specifying dynamism on the first kwarg even though user passed in
            # different order
            dynamic_shapes=(None, {0: dim}, {0: dim_for_kw1}, None),
        )
        onnx_program = torch.onnx.dynamo_export(
            exported_program,
            torch.randn(4, 4),
            torch.randn(4, 4),
            kw2=torch.ones(4, 4),
            kw1=torch.zeros(4, 4),
        )

        test_inputs = (torch.randn(4, 4), torch.randn(7, 4))
        test_kwargs = {"kw2": torch.ones(4, 4), "kw1": torch.zeros(9, 4)}
        # This should work even if the kwarg order are flipped.
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, test_inputs, test_kwargs
        )

    def test_exported_program_as_input_lifting_buffers_mutation(self):
        for persistent in (True, False):

            class CustomModule(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.register_buffer(
                        "my_buffer", torch.tensor(4.0), persistent=persistent
                    )

                def forward(self, x, b):
                    output = x + b
                    (
                        self.my_buffer.add_(1.0) + 3.0
                    )  # Mutate buffer through in-place addition
                    return output

            input_x = torch.rand((3, 3), dtype=torch.float32)
            input_b = torch.randn(3, 3)
            model = CustomModule()

            dim = torch.export.Dim("dim")
            exported_program = torch.export.export(
                model,
                (
                    input_x,
                    input_b,
                ),
                dynamic_shapes=({0: dim}, {0: dim}),
            )
            onnx_program = torch.onnx.dynamo_export(exported_program, input_x, input_b)

            # different dim inputs
            additional_inputs_x = torch.rand((4, 3), dtype=torch.float32)
            additional_inputs_b = torch.randn(4, 3)
            self._compare_onnx_and_torch_exported_program(
                exported_program,
                onnx_program,
                (
                    additional_inputs_x,
                    additional_inputs_b,
                ),
            )

    def test_onnx_program_supports_non_arg_name_with_container_type(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b):
                return a[0].sum() + a[1].sum() + b.sum()

        foo = Foo()

        inp_a = (torch.randn(4, 4), torch.randn(4, 4))
        inp_b = torch.randn(4, 4)
        inp = (inp_a, inp_b)

        count = 0

        def dynamify_inp(x):
            # Mark the second input a[1] dynamic
            nonlocal count
            if count == 1:
                dim = torch.export.Dim("dim", min=3)
                count += 1
                return {0: dim}
            count += 1
            return None

        dynamic_shapes = torch_pytree.tree_map(dynamify_inp, inp)
        exported_program = torch.export.export(foo, inp, dynamic_shapes=dynamic_shapes)
        onnx_program = torch.onnx.dynamo_export(exported_program, inp_a, inp_b)

        # NOTE: Careful with the input format. The input format should be
        # consistent with how the model is exported.
        test_inputs = ((torch.randn(4, 4), torch.randn(6, 4)), torch.randn(4, 4))
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, test_inputs
        )

    def test_onnx_program_supports_lazy_module_kwargs(self):
        class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def initialize_parameters(self, *args, **kwargs):
                pass

            def forward(self, x, y):
                return x + y

        m = LazyModule()
        dim = torch.export.Dim("dim")
        dynamic_shapes = ({0: dim}, {0: dim})
        exported_program = torch.export.export(
            m,
            (),
            {"x": torch.randn(3, 3), "y": torch.randn(3, 3)},
            dynamic_shapes=dynamic_shapes,
        )
        onnx_program = torch.onnx.dynamo_export(
            exported_program, x=torch.randn(3, 3), y=torch.randn(3, 3)
        )

        # NOTE: A model should be fed with the input formats that
        # how the model is exported
        inputs = {"x": torch.randn(6, 3), "y": torch.randn(6, 3)}
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, input_args=(), input_kwargs=inputs
        )


if __name__ == "__main__":
    common_utils.run_tests()
