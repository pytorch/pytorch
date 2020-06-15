import unittest
import torch
import torch.utils.bundled_inputs

import io

from torch.jit.mobile import _load_for_lite_interpreter

class TestLiteScriptModule(unittest.TestCase):

    def test_load_mobile_module(self):
        class MyTestModule(torch.nn.Module):
            def __init__(self):
                super(MyTestModule, self).__init__()

            def forward(self, x):
                return x + 10

        input = torch.tensor([1])

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        mobile_module_result = mobile_module(input)
        torch.testing.assert_allclose(script_module_result, mobile_module_result)

        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_allclose(script_module_result, mobile_module_forward_result)

        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        torch.testing.assert_allclose(script_module_result, mobile_module_run_method_result)


    def test_find_and_run_method(self):
        class MyTestModule(torch.nn.Module):
            def forward(self, arg):
                return arg

        input = (torch.tensor([1]), )

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(*input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        has_bundled_inputs = mobile_module.find_method("get_all_bundled_inputs")
        self.assertFalse(has_bundled_inputs)

        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            script_module, [input], [])

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        has_bundled_inputs = mobile_module.find_method("get_all_bundled_inputs")
        self.assertTrue(has_bundled_inputs)

        bundled_inputs = mobile_module.run_method("get_all_bundled_inputs")
        mobile_module_result = mobile_module.forward(*bundled_inputs[0])
        torch.testing.assert_allclose(script_module_result, mobile_module_result)



if __name__ == '__main__':
    unittest.main()
