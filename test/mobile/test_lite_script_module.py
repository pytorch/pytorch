import torch
import torch.utils.bundled_inputs
from torch.utils.mobile_optimizer import *
import io
from typing import NamedTuple
from collections import namedtuple

from torch.jit.mobile import _load_for_lite_interpreter, _export_operator_list
from torch.testing._internal.common_utils import TestCase, run_tests

class TestLiteScriptModule(TestCase):

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

    def test_save_mobile_module_with_debug_info_with_trace(self):
        class A(torch.nn.Module):
            def __init__(self):
                super(A, self).__init__()

            def forward(self, x):
                return x + 1

        class B(torch.nn.Module):
            def __init__(self):
                super(B, self).__init__()
                self.A0 = A()
                self.A1 = A()

            def forward(self, x):
                return self.A0(x) + self.A1(x)

        input = torch.tensor([5])
        trace_module = torch.jit.trace(B(), input)
        exported_module = trace_module._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True)

        assert(b"mobile_debug.pkl" in exported_module)
        assert(b"module_debug_info" in exported_module)
        assert(b"top(B).forward" in exported_module)
        assert(b"top(B).A0(A).forward" in exported_module)
        assert(b"top(B).A1(A).forward" in exported_module)

    def test_save_mobile_module_with_debug_info_with_script_duplicate_class(self):
        class A(torch.nn.Module):
            def __init__(self):
                super(A, self).__init__()

            def forward(self, x):
                return x + 1

        class B(torch.nn.Module):
            def __init__(self):
                super(B, self).__init__()
                self.A0 = A()
                self.A1 = A()

            def forward(self, x):
                return self.A0(x) + self.A1(x)

        input_data = torch.tensor([5])
        scripted_module = torch.jit.script(B(), input_data)
        exported_module = scripted_module._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True)

        assert(b"mobile_debug.pkl" in exported_module)
        assert(b"module_debug_info" in exported_module)
        assert(b"top(B).forward" in exported_module)
        assert(b"top(B).A0(A).forward" in exported_module)
        assert(b"top(B).A1(A).forward" in exported_module)

    def test_save_mobile_module_with_debug_info_with_script_nested_call(self):
        class A(torch.nn.Module):
            def __init__(self):
                super(A, self).__init__()

            def forward(self, x):
                return x + 1

        class B(torch.nn.Module):
            def __init__(self):
                super(B, self).__init__()

            def forward(self, x):
                return x + 2

        class C(torch.nn.Module):
            def __init__(self):
                super(C, self).__init__()
                self.A0 = A()
                self.B0 = B()

            def forward(self, x):
                return self.A0(self.B0(x)) + 1

        input = torch.tensor([5])
        scripted_module = torch.jit.script(C(), input)

        optimized_scripted_module = optimize_for_mobile(scripted_module)

        exported_module = scripted_module._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True)
        optimized_exported_module = optimized_scripted_module._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True)
        assert(b"mobile_debug.pkl" in exported_module)
        assert(b"module_debug_info" in exported_module)
        assert(b"top(C).forward" in exported_module)
        assert(b"top(C).A0(A).forward" in exported_module)
        assert(b"top(C).B0(B).forward" in exported_module)

        assert(b"mobile_debug.pkl" in optimized_exported_module)
        assert(b"module_debug_info" in optimized_exported_module)
        assert(b"top(C).forward" in optimized_exported_module)
        assert(b"top(C).A0(A).forward" in optimized_exported_module)
        assert(b"top(C).B0(B).forward" in optimized_exported_module)

    def test_load_mobile_module_with_debug_info(self):
        class MyTestModule(torch.nn.Module):
            def __init__(self):
                super(MyTestModule, self).__init__()

            def forward(self, x):
                return x + 5

        input = torch.tensor([3])

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True))
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

    def test_method_calls_with_optional_arg(self):
        class A(torch.nn.Module):
            def __init__(self):
                super(A, self).__init__()

            # opt arg in script-to-script invocation
            def forward(self, x, two: int = 2):
                return x + two

        class B(torch.nn.Module):
            def __init__(self):
                super(B, self).__init__()
                self.A0 = A()

            # opt arg in Python-to-script invocation
            def forward(self, x, one: int = 1):
                return self.A0(x) + one

        script_module = torch.jit.script(B())
        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter()
        )
        mobile_module = _load_for_lite_interpreter(buffer)

        input = torch.tensor([5])
        script_module_forward_result = script_module.forward(input)
        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_allclose(
            script_module_forward_result,
            mobile_module_forward_result
        )

        # change ref only
        script_module_forward_result = script_module.forward(input, 2)
        self.assertFalse(
            (script_module_forward_result == mobile_module_forward_result)
            .all()
            .item()
        )

        # now both match again
        mobile_module_forward_result = mobile_module.forward(input, 2)
        torch.testing.assert_allclose(
            script_module_forward_result,
            mobile_module_forward_result
        )

    def test_unsupported_classtype(self):
        class Foo():
            def __init__(self):
                return

            def func(self, x: int, y: int):
                return x + y

        class MyTestModule(torch.nn.Module):
            def forward(self, arg):
                f = Foo()
                return f.func(1, 2)

        script_module = torch.jit.script(MyTestModule())
        with self.assertRaisesRegex(RuntimeError,
                                    r"Workaround: instead of using arbitrary class type \(class Foo\(\)\), "
                                    r"define a pytorch class \(class Foo\(torch\.nn\.Module\)\)\.$"):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_typing_namedtuple(self):
        myNamedTuple = NamedTuple('myNamedTuple', [('a', torch.Tensor)])

        class MyTestModule(torch.nn.Module):
            def forward(self):
                return myNamedTuple(torch.randn(1))

        script_module = torch.jit.script(MyTestModule())
        with self.assertRaisesRegex(RuntimeError,
                                    r"A named tuple type is not supported in mobile module. "
                                    r"Workaround: instead of using a named tuple type\'s fields, "
                                    r"use a dictionary type\'s key-value pair itmes or "
                                    r"a pytorch class \(class Foo\(torch\.nn\.Module\)\)\'s attributes."):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_collections_namedtuple(self):
        myNamedTuple = namedtuple('myNamedTuple', [('a')])

        class MyTestModule(torch.nn.Module):
            def forward(self):
                return myNamedTuple(torch.randn(1))

        script_module = torch.jit.script(MyTestModule())
        with self.assertRaisesRegex(RuntimeError,
                                    r"A named tuple type is not supported in mobile module. "
                                    r"Workaround: instead of using a named tuple type\'s fields, "
                                    r"use a dictionary type\'s key-value pair itmes or "
                                    r"a pytorch class \(class Foo\(torch\.nn\.Module\)\)\'s attributes."):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_list_with_module_class(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super(Foo, self).__init__()

        class MyTestModuleForListWithModuleClass(torch.nn.Module):
            def __init__(self):
                super(MyTestModuleForListWithModuleClass, self).__init__()
                self.foo = Foo()

            def forward(self):
                my_list: List[Foo] = [self.foo]
                return my_list

        script_module = torch.jit.script(MyTestModuleForListWithModuleClass())
        with self.assertRaisesRegex(RuntimeError,
                                    r"^Returining a list or dictionary with pytorch class type "
                                    r"is not supported in mobile module "
                                    r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(torch\.nn\.Module\)\)\. "
                                    r"Workaround\: instead of using pytorch class as their element type\, "
                                    r"use a combination of list\, dictionary\, and single types\.$"):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_dict_with_module_class(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super(Foo, self).__init__()

        class MyTestModuleForDictWithModuleClass(torch.nn.Module):
            def __init__(self):
                super(MyTestModuleForDictWithModuleClass, self).__init__()
                self.foo = Foo()

            def forward(self):
                my_dict: Dict[int, Foo] = {1: self.foo}
                return my_dict

        script_module = torch.jit.script(MyTestModuleForDictWithModuleClass())
        with self.assertRaisesRegex(RuntimeError,
                                    r"^Returining a list or dictionary with pytorch class type "
                                    r"is not supported in mobile module "
                                    r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(torch\.nn\.Module\)\)\. "
                                    r"Workaround\: instead of using pytorch class as their element type\, "
                                    r"use a combination of list\, dictionary\, and single types\.$"):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_module_export_operator_list(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super(Foo, self).__init__()
                self.weight = torch.ones((20, 1, 5, 5))
                self.bias = torch.ones(20)

            def forward(self, input):
                x1 = torch.zeros(2, 2)
                x2 = torch.empty_like(torch.empty(2, 2))
                x3 = torch._convolution(
                    input,
                    self.weight,
                    self.bias,
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                    False,
                    False,
                    True,
                    True,
                )
                return (x1, x2, x3)

        m = torch.jit.script(Foo())

        buffer = io.BytesIO(m._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        expected_ops = {
            "aten::_convolution",
            "aten::empty.memory_format",
            "aten::empty_like",
            "aten::zeros",
        }
        actual_ops = _export_operator_list(mobile_module)
        self.assertEqual(actual_ops, expected_ops)

if __name__ == '__main__':
    run_tests()
