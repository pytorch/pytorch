import torch
import torch.utils.bundled_inputs
import io
from typing import List, NamedTuple

from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import TestCase, run_tests


class TestLiteScriptModule(TestCase):


    def test_namedtupe(self):
        myNamedTuple = NamedTuple('myNamedTuple', [('a', List[torch.Tensor])])
        class MyTestModule(torch.nn.Module):
            def forward(self, a: torch.Tensor):
                p = myNamedTuple([a])
                return p

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(sample_input).a

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True))
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)  # Error here
        mobile_module_result = mobile_module(sample_input).a
        torch.testing.assert_allclose(
            script_module_result,
            mobile_module_result
        )

    def test_namedtuple_custom_classtype(self):
        class Foo(NamedTuple):
            id: torch.Tensor

        class Bar(torch.nn.Module):
            def __init__(self):
                super(Bar, self).__init__()
                self.foo = Foo(torch.tensor(1))

            def forward(self, a: torch.Tensor):
                self.foo = Foo(a)
                return self.foo

        sample_input = torch.Tensor(1)

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(Bar())
        script_module_result = script_module(sample_input)
        buffer_jit = io.BytesIO(script_module.save_to_buffer())
        buffer_jit.seek(0)

        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_allclose(
            script_module_result,
            mobile_module_result
        )

if __name__ == '__main__':
    run_tests()
