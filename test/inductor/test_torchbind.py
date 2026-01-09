# Owner(s): ["module: functorch"]
import json
import zipfile
from pathlib import Path

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops.torchbind import CallTorchBind, enable_torchbind_tracing
from torch._inductor import aot_compile, ir
from torch._inductor.codecache import WritableTempFile
from torch._inductor.package import package_aoti
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)


class TestTorchbind(TestCase):
    def setUp(self):
        super().setUp()
        init_torchbind_implementations()

    def get_dummy_exported_model(self):
        """
        Returns the ExportedProgram, example inputs, and result from calling the
        eager model with those inputs
        """

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        ep = torch.export.export(m, inputs, strict=False)

        return ep, inputs, orig_res, m

    def get_exported_model(self):
        """
        Returns the ExportedProgram, example inputs, and result from calling the
        eager model with those inputs
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
                self.b = torch.randn(2, 3)

            def forward(self, x):
                x = x + self.b
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                c = self.attr.add_tensor(x)
                return x + b + c

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        return ep, inputs, orig_res, m

    def test_torchbind_inductor(self):
        ep, inputs, orig_res, _ = self.get_exported_model()
        compiled = torch._inductor.compile(ep.module(), inputs)

        new_res = compiled(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_compile_symint(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(2, 3)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_tensor_return(self.attr, x)
                return a

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)
        new_res = torch.compile(m, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_compile(self):
        _, inputs, orig_res, mod = self.get_exported_model()
        new_res = torch.compile(mod, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_get_buf_bytes(self):
        a = torch.classes._TorchScriptTesting._Foo(10, 20)
        buffer = ir.TorchBindObject(name="a", value=a)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 0)

        t = torch.randn(2, 3)
        b = torch.classes._TorchScriptTesting._ContainsTensor(t)
        buffer = ir.TorchBindObject(name="b", value=b)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 2 * 3 * 4)

        q = _empty_tensor_queue()
        buffer = ir.TorchBindObject(name="q", value=q)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 0)

        q.push(torch.ones(2, 3))
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 2 * 3 * 4)

    def test_torchbind_hop_schema(self):
        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        foo_ir = ir.TorchBindObject(name="foo", value=foo)
        schema = CallTorchBind.schema(foo_ir, "add")
        self.assertEqual(
            str(schema),
            "call_torchbind(__torch__.torch.classes._TorchScriptTesting._Foo _0, str method, int _1) -> int _0",
        )

    def test_torchbind_config_not_generated(self):
        # custom_objs_config.json should not be generated when its empty
        ep, inputs, _, _ = self.get_dummy_exported_model()
        aoti_files = aot_compile(
            ep.module(), inputs, options={"aot_inductor.package": True}
        )
        for file in aoti_files:
            self.assertTrue(not file.endswith("/custom_objs_config.json"))

    def test_torchbind_hop_schema_no_input(self):
        q = _empty_tensor_queue()
        q_ir = ir.TorchBindObject(name="q", value=q)
        schema = CallTorchBind.schema(q_ir, "pop")
        self.assertEqual(
            str(schema),
            "call_torchbind(__torch__.torch.classes._TorchScriptTesting._TensorQueue _0, str method) -> Tensor _0",
        )

    def test_torchbind_hop_schema_no_output(self):
        q = _empty_tensor_queue()
        q_ir = ir.TorchBindObject(name="q", value=q)
        schema = CallTorchBind.schema(q_ir, "push")
        self.assertEqual(
            str(schema),
            "call_torchbind(__torch__.torch.classes._TorchScriptTesting._TensorQueue _0, str method, Tensor _1) -> NoneType _0",
        )

    @skipIfWindows(msg="AOTI is not fully support on Windows")
    def test_torchbind_aot_compile(self):
        ep, inputs, _, _ = self.get_exported_model()
        aoti_files = aot_compile(
            ep.module(), inputs, options={"aot_inductor.package": True}
        )

        custom_objs_config = None
        custom_obj_0 = None
        extern_json = None
        for file in aoti_files:
            if file.endswith("/custom_objs_config.json"):
                custom_objs_config = file
            elif file.endswith("/custom_obj_0"):
                custom_obj_0 = file
            elif file.endswith("wrapper.json") and "metadata" not in file:
                extern_json = file

        self.assertIsNotNone(custom_objs_config)
        self.assertIsNotNone(custom_obj_0)
        self.assertIsNotNone(extern_json)

        with open(custom_objs_config) as file:
            data = json.load(file)
            self.assertEqual(data, {"_torchbind_obj0": "custom_obj_0"})

        with open(extern_json) as file:
            data = json.load(file)
            self.assertEqual(
                data,
                {
                    "nodes": [
                        {
                            "name": "buf1",
                            "node": {
                                "target": "_TorchScriptTesting::takes_foo_tuple_return",
                                "inputs": [
                                    {
                                        "name": "foo",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "x",
                                        "arg": {"as_tensor": {"name": "buf0"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [
                                    {"as_tensor": {"name": "buf2"}},
                                    {"as_tensor": {"name": "buf3"}},
                                ],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                                "name": None,
                            },
                        },
                        {
                            "name": "buf5",
                            "node": {
                                "target": "_TorchScriptTesting::takes_foo",
                                "inputs": [
                                    {
                                        "name": "foo",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "x",
                                        "arg": {"as_tensor": {"name": "buf4"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [{"as_tensor": {"name": "buf6"}}],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                                "name": None,
                            },
                        },
                        {
                            "name": "buf7",
                            "node": {
                                "target": "call_torchbind",
                                "inputs": [
                                    {
                                        "name": "_0",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "method",
                                        "arg": {"as_string": "add_tensor"},
                                        "kind": 1,
                                    },
                                    {
                                        "name": "_1",
                                        "arg": {"as_tensor": {"name": "buf0"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [{"as_tensor": {"name": "buf8"}}],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                                "name": None,
                            },
                        },
                    ]
                },
            )

        # Test that the files are packaged
        with WritableTempFile(suffix=".pt2") as f:
            package_path = package_aoti(f.name, aoti_files)

            with zipfile.ZipFile(package_path, "r") as zip_ref:
                all_files = zip_ref.namelist()
                base_folder = all_files[0].split("/")[0]
                tmp_path_model = Path(base_folder) / "data" / "aotinductor" / "model"
                tmp_path_constants = Path(base_folder) / "data" / "constants"

                self.assertTrue(
                    str(tmp_path_model / "custom_objs_config.json") in all_files
                )
                self.assertTrue(str(tmp_path_constants / "custom_obj_0") in all_files)

    def test_torchbind_aoti(self):
        ep, inputs, orig_res, _ = self.get_exported_model()
        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)

    @torch._inductor.config.patch("aot_inductor.use_runtime_constant_folding", True)
    @skipIfWindows(msg="AOTI is not fully support on Windows")
    def test_torchbind_aot_compile_constant_folding(self):
        ep, inputs, orig_res, _ = self.get_exported_model()
        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)

    def test_torchbind_list_return_aot_compile(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(self.attr, x)
                y = a[0] + a[1] + a[2]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)

    def test_torchbind_queue(self):
        class Foo(torch.nn.Module):
            def __init__(self, tq) -> None:
                super().__init__()
                self.tq = tq

            def forward(self, x):
                self.tq.push(x.cos())
                self.tq.push(x.sin())
                # TODO: int return type in fallback kernel not support yet
                x_cos = self.tq.pop()  # + self.tq.size()
                x_sin = self.tq.pop()  # - self.tq.size()
                return x_sin, x_cos

        inputs = (torch.randn(3, 2),)

        q = _empty_tensor_queue()
        m = Foo(q)
        orig_res = m(*inputs)

        q2 = _empty_tensor_queue()
        m2 = Foo(q2)

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m2, inputs, strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)

    @requires_gpu()
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    @torch._inductor.config.patch("graph_partition", True)
    def test_torchbind_compile_gpu_op_symint_graph_partition(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(2, 3)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_tensor_return(self.attr, x)
                a_cuda = a.to(device=GPU_TYPE)
                return a_cuda + 1

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)
        new_res = torch.compile(m, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_input_aot_compile(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(x, y)
                return a

        m = M()
        inputs = (torch.classes._TorchScriptTesting._Foo(10, 20), torch.ones(2, 3))

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        from torch._dynamo.exc import UserError

        with self.assertRaisesRegex(
            UserError,
            expected_regex="TorchBind object inputs are not supported in AOTInductor",
        ):
            aot_compile(ep.module(), inputs, options={"aot_inductor.package": True})

    def test_aoti_torchbind_name_collision(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._torchbind_obj0 = torch.classes._TorchScriptTesting._Foo(2, 3)

            def forward(self, x):
                a = self._torchbind_obj0.add_tensor(x)
                torchbind = torch.classes._TorchScriptTesting._Foo(4, 5)
                b = torchbind.add_tensor(x)
                return a + b

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)


if __name__ == "__main__":
    run_tests()
