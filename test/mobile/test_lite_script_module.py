# Owner(s): ["oncall: mobile"]

import torch
import torch.utils.bundled_inputs
import io
from typing import Dict, List
import inspect
from torch.testing import FileCheck

from torch.jit.mobile import _load_for_lite_interpreter, _export_operator_list
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_quantization import (
    AnnotatedSingleLayerLinearModel,
    TwoLayerLinearModel,
    AnnotatedNestedModel
)
from torch.testing._internal.common_quantization import QuantizationLiteTestCase

class TestLiteScriptModule(TestCase):

    def getScriptExportImportCopy(self, m, save_mobile_debug_info=True, also_test_file=False):
        m_scripted = torch.jit.script(m)

        if not also_test_file:
            buffer = io.BytesIO(m_scripted._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=save_mobile_debug_info))
            buffer.seek(0)
            mobile_module = _load_for_lite_interpreter(buffer)
            return mobile_module

        with TemporaryFileName() as fname:
            m_scripted._save_for_lite_interpreter(fname, _save_mobile_debug_info=save_mobile_debug_info)
            mobile_module = _load_for_lite_interpreter(fname)
            return mobile_module

    def test_load_mobile_module(self):
        class MyTestModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        input = torch.tensor([1])

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        mobile_module_result = mobile_module(input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_close(script_module_result, mobile_module_forward_result)

        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        torch.testing.assert_close(script_module_result, mobile_module_run_method_result)

    def test_save_mobile_module_with_debug_info_with_trace(self):
        class A(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.A0 = A()
                self.A1 = A()

            def forward(self, x, y, z):
                return self.A0(x, y) + self.A1(y, z)

        for export_method in ['trace', 'script']:
            x = torch.rand((2, 3))
            y = torch.rand((2, 3))
            z = torch.rand((2, 3))
            if export_method == 'trace':
                trace_module = torch.jit.trace(B(), [x, y, z])
            else:
                trace_module = torch.jit.script(B())
            exported_module = trace_module._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True)
            buffer = io.BytesIO(exported_module)
            buffer.seek(0)

            assert(b"callstack_debug_map.pkl" in exported_module)

            mobile_module = _load_for_lite_interpreter(buffer)
            with self.assertRaisesRegex(RuntimeError, r"Module hierarchy:top\(B\)::<unknown>.A0\(A\)::forward.aten::mul"):
                x = torch.rand((2, 3))
                y = torch.rand((8, 10))
                z = torch.rand((8, 10))
                mobile_module(x, y, z)
            with self.assertRaisesRegex(RuntimeError, r"Module hierarchy:top\(B\)::<unknown>.A1\(A\)::forward.aten::mul"):
                x = torch.rand((2, 3))
                y = torch.rand((2, 3))
                z = torch.rand((8, 10))
                mobile_module(x, y, z)

    def test_load_mobile_module_with_debug_info(self):
        class MyTestModule(torch.nn.Module):
            def forward(self, x):
                return x + 5

        input = torch.tensor([3])

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True))
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        mobile_module_result = mobile_module(input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_close(script_module_result, mobile_module_forward_result)

        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        torch.testing.assert_close(script_module_result, mobile_module_run_method_result)

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
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_method_calls_with_optional_arg(self):
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # opt arg in script-to-script invocation
            def forward(self, x, two: int = 2):
                return x + two

        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
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
        torch.testing.assert_close(
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
        torch.testing.assert_close(
            script_module_forward_result,
            mobile_module_forward_result
        )

    def test_unsupported_classtype(self):
        class Foo:
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
                                    r"define a pytorch class \(class Foo\(torch\.nn\.Module\)\)\. "
                                    r"The problematic type is: "):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_list_with_module_class(self):
        class Foo(torch.nn.Module):
            pass

        class MyTestModuleForListWithModuleClass(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = Foo()

            def forward(self):
                my_list: List[Foo] = [self.foo]
                return my_list

        script_module = torch.jit.script(MyTestModuleForListWithModuleClass())
        with self.assertRaisesRegex(RuntimeError,
                                    r"^Returning a list or dictionary with pytorch class type "
                                    r"is not supported in mobile module "
                                    r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(torch\.nn\.Module\)\)\. "
                                    r"Workaround\: instead of using pytorch class as their element type\, "
                                    r"use a combination of list\, dictionary\, and single types\.$"):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_dict_with_module_class(self):
        class Foo(torch.nn.Module):
            pass

        class MyTestModuleForDictWithModuleClass(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = Foo()

            def forward(self):
                my_dict: Dict[int, Foo] = {1: self.foo}
                return my_dict

        script_module = torch.jit.script(MyTestModuleForDictWithModuleClass())
        with self.assertRaisesRegex(RuntimeError,
                                    r"^Returning a list or dictionary with pytorch class type "
                                    r"is not supported in mobile module "
                                    r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(torch\.nn\.Module\)\)\. "
                                    r"Workaround\: instead of using pytorch class as their element type\, "
                                    r"use a combination of list\, dictionary\, and single types\.$"):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_module_export_operator_list(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
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

    def test_source_range_simple(self):

        class FooTest(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, w):
                return torch.mm(x, w.t())

        ft = FooTest()
        loaded = self.getScriptExportImportCopy(ft)
        _, lineno = inspect.getsourcelines(FooTest)

        with self.assertRaisesRegex(RuntimeError, f'test_lite_script_module.py\", line {lineno + 3}'):
            loaded(torch.rand(3, 4), torch.rand(30, 40))

    def test_source_range_raise_exception(self):

        class FooTest2(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self):
                raise RuntimeError('foo')

        _, lineno = inspect.getsourcelines(FooTest2)

        # In C++ code, the type of exception thrown is torch::jit::JITException
        # which does not extend c10::Error, and hence it isn't possible to add
        # additional context to the exception message and preserve the correct
        #  C++ stack trace for symbolication. i.e. it isn't possible to add
        # the debug handle string to show where in the Python code the exception
        # occured w/o first changing
        # torch::jit::JITException to extend c10::Error.
        with self.assertRaisesRegex(torch.jit.Error, 'foo'):
            ft = FooTest2()
            loaded = self.getScriptExportImportCopy(ft)
            loaded()

    def test_source_range_function_call(self):
        class FooTest3(torch.jit.ScriptModule):
            @torch.jit.script_method
            def add_method(self, x, w):
                return x + w

            @torch.jit.script_method
            def forward(self, x, y, w):
                x = x * y
                x = x + 2
                return self.add_method(x, w)

        ft = FooTest3()
        loaded = self.getScriptExportImportCopy(ft)
        _, lineno = inspect.getsourcelines(FooTest3)

        try:
            loaded(torch.rand(3, 4), torch.rand(3, 4), torch.rand(30, 40))
        except RuntimeError as e:
            error_message = f"{e}"
        self.assertTrue(f'test_lite_script_module.py\", line {lineno + 3}' in error_message)
        self.assertTrue(f'test_lite_script_module.py\", line {lineno + 9}' in error_message)
        self.assertTrue('top(FooTest3)' in error_message)

    def test_source_range_no_debug_info(self):

        class FooTest4(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, w):
                return torch.mm(x, w.t())

        ft = FooTest4()
        loaded = self.getScriptExportImportCopy(ft, save_mobile_debug_info=False)

        try:
            loaded(torch.rand(3, 4), torch.rand(30, 40))
        except RuntimeError as e:
            error_message = f"{e}"
        self.assertTrue("test_lite_script_module.py" not in error_message)

    def test_source_range_raise_exc(self):
        class FooTest5(torch.jit.ScriptModule):
            def __init__(self, val: int):
                super().__init__()
                self.val = val

            @torch.jit.script_method
            def add_method(self, val: int, x, w):
                if (val == self.val):
                    raise RuntimeError('self.val and val are same')
                return x + w

            @torch.jit.script_method
            def forward(self, val: int, x, y, w):
                x = x * y
                x = x + 2
                return self.add_method(val, x, w)

        ft = FooTest5(42)
        loaded = self.getScriptExportImportCopy(ft)
        _, lineno = inspect.getsourcelines(FooTest5)

        try:
            loaded(42, torch.rand(3, 4), torch.rand(3, 4), torch.rand(30, 40))
        except torch.jit.Error as e:
            error_message = f"{e}"

        # In C++ code, the type of exception thrown is torch::jit::JITException
        # which does not extend c10::Error, and hence it isn't possible to add
        # additional context to the exception message and preserve the correct
        #  C++ stack trace for symbolication. i.e. it isn't possible to add
        # the debug handle string to show where in the Python code the exception
        # occured w/o first changing
        # torch::jit::JITException to extend c10::Error.
        self.assertTrue('self.val and val are same' in error_message)

    def test_stacktrace_interface_call(self):
        @torch.jit.interface
        class Forward(torch.nn.Module):
            def forward(self, x) -> torch.Tensor:
                pass

            def forwardError(self, x) -> torch.Tensor:
                pass

        class B(torch.nn.Module):
            def forward(self, x):
                return x

            def forwardError(self, x):
                return self.call() + x

            def call(self):
                return torch.ones(-1)

        class A(torch.nn.Module):
            b : Forward

            def __init__(self):
                super().__init__()
                self.b = B()

            def forward(self):
                self.b.forward(torch.ones(1))
                self.b.forwardError(torch.ones(1))

        a = torch.jit.script(A())
        torch._C._enable_mobile_interface_call_export()
        buffer = io.BytesIO(a._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True))
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)
        try:
            mobile_module()
            self.assertTrue(False)
        except RuntimeError as exp:
            FileCheck().check("Trying to create tensor with negative dimension") \
                .check("Traceback of TorchScript") \
                .check("self.b.forwardError").check_next("~~~~~~~~~~~~~~~~~~~ <--- HERE") \
                .check("return self.call").check_next("~~~~~~~~~ <--- HERE") \
                .check("return torch.ones").check_next("~~~~~~~~~~ <--- HERE").run(str(exp))



class TestLiteScriptQuantizedModule(QuantizationLiteTestCase):

    def test_single_layer(self):
        input = torch.rand(2, 5, dtype=torch.float)
        quantized_model = self._create_quantized_model(model_class=AnnotatedSingleLayerLinearModel, qengine="qnnpack")
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_two_layer(self):
        input = torch.rand(2, 5, dtype=torch.float)
        quantized_model = self._create_quantized_model(model_class=TwoLayerLinearModel)
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_annotated_nested(self):
        input = torch.rand(2, 5, dtype=torch.float)
        quantized_model = self._create_quantized_model(model_class=AnnotatedNestedModel, qengine="qnnpack")
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_quantization_example(self):

        # From the example in Static Quantization section of https://pytorch.org/docs/stable/quantization.html
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model_fp32 = M()

        model_fp32.eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        input_fp32 = torch.randn(4, 1, 4, 4)
        model_fp32_prepared(input_fp32)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        input = torch.randn(4, 1, 4, 4)
        self._compare_script_and_mobile(model=model_int8, input=input)

    def test_bundled_input_with_dynamic_type(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x: Dict[int, torch.Tensor],
                y: Dict[int, torch.Tensor],
                z: Dict[int, torch.Tensor],
            ):
                return x

        model = Model()
        script_module = torch.jit.script(model)

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

if __name__ == '__main__':
    run_tests()
