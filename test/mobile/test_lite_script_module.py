# Owner(s): ["oncall: mobile"]

import io

import torch
import torch.utils.bundled_inputs
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLiteScriptQuantizedModule(TestCase):
    def test_bundled_input_with_dynamic_type(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x: dict[int, torch.Tensor],
                y: dict[int, torch.Tensor],
                z: dict[int, torch.Tensor],
            ):
                return x

        model = Model()
        script_module = torch.jit._script.script(model)

        sample_input = {
            script_module.forward: [
                (
                    {0: torch.ones(1)},
                    {1: torch.ones(1)},
                    {2: torch.ones(1)},
                )
            ]
        }

        bundled_model = torch.utils.bundled_inputs.bundle_inputs(
            script_module, sample_input
        )

        buf = bundled_model._save_to_buffer_for_lite_interpreter()
        mobile_module = _load_for_lite_interpreter(io.BytesIO(buf))

        i = mobile_module.run_method("get_all_bundled_inputs")

        self.assertEqual(
            i[0],
            (
                {0: torch.ones(1)},
                {1: torch.ones(1)},
                {2: torch.ones(1)},
            ),
        )


if __name__ == "__main__":
    run_tests()
