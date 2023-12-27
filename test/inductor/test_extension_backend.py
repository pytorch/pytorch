# Owner(s): ["module: inductor"]
import importlib
import os
import shutil
import sys
import unittest
from typing import Any

import torch
import torch._dynamo
import torch.utils.cpp_extension

try:
    from extension_backends.extension_codegen_backend import (
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .extension_backends.extension_codegen_backend import (
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )

from torch._C import FileCheck
from torch._inductor import codecache, metrics
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
)
from torch.testing._internal.common_utils import IS_FBCODE, IS_MACOS

try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase


@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class ExtensionBackendTests(TestCase):
    module = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Build Extension
        source_file_path = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(
            source_file_path, "extension_backends/extension_device.cpp"
        )

        extension_cache_dir = os.path.join(codecache.cache_dir(), "cpp_extension")
        with open(source_file) as f:
            hash_key = codecache.code_hash(f.read())

        full_cache_dir = os.path.join(extension_cache_dir, hash_key[1:3])
        os.makedirs(full_cache_dir, exist_ok=True)
        dst_cache_file_path = os.path.join(full_cache_dir, hash_key + ".so")

        name = "extension_device"
        if not os.path.exists(dst_cache_file_path):
            cls.module = torch.utils.cpp_extension.load(
                name=name,
                sources=[
                    str(source_file),
                ],
                extra_cflags=["-g"],
                verbose=True,
                is_python_module=True,
            )
            shutil.copy(cls.module.__file__, dst_cache_file_path)
        else:
            spec = importlib.util.spec_from_file_location(name, dst_cache_file_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(module)
            cls.module = module

        torch.utils.rename_privateuse1_backend("extension_device")
        register_backend_for_device(
            "extension_device", ExtensionScheduling, ExtensionWrapperCodegen
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        assert self.module is not None

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    class KernelFunWrapper:
        def __init__(self, op_name, dynamic_shape=True) -> None:
            self.op_name = op_name
            self.dynamic_shape = dynamic_shape

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            with torch._C._SetExcludeDispatchKeyGuard(
                torch._C.DispatchKey.Python, False
            ):
                opt_fn = torch.compile(
                    getattr(torch.ops.aten, self.op_name), dynamic=self.dynamic_shape
                )
                return opt_fn(*args, **kwargs)

    def make_elementwise(self, op_name, dynamic_shape=True):
        return self.KernelFunWrapper(op_name, dynamic_shape)

    def register_ops(
        self,
        op_set,
        namespace_name: str,
        lib_impl: torch.library.Library,
        dispatch_key: str,
    ):
        for _op_name in op_set:
            qualified_op_name = f"{namespace_name}::{_op_name}"
            _, overload_names = torch._C._jit_get_operation(qualified_op_name)
            for overload_name in overload_names:
                try:
                    schema = torch._C._get_schema(qualified_op_name, overload_name)
                    reg_name = schema.name
                    if schema.overload_name:
                        reg_name = f"{reg_name}.{schema.overload_name}"
                    lib_impl.impl(
                        reg_name,
                        self.make_elementwise(_op_name),
                        dispatch_key,
                        compile_mode=True,
                    )
                except Exception:
                    continue

    def test_torch_compile_eager(self):
        namespace_name = "aten"
        dispatch_key = "PrivateUse1"
        torch_compile_op_lib_impl = torch.library.Library("aten", "IMPL")

        op_set = ["add", "mul"]
        self.register_ops(
            op_set, namespace_name, torch_compile_op_lib_impl, dispatch_key
        )

        def fn(a, b, c):
            return a * b + c

        device = self.module.custom_device()
        x = torch.empty(2, 16).to(device=device).fill_(1)
        y = torch.empty(2, 16).to(device=device).fill_(2)
        z = torch.empty(2, 16).to(device=device).fill_(3)
        ref = torch.empty(2, 16).fill_(5)

        torch._dynamo.reset()
        metrics.reset()
        opt_fn = torch.compile(fn)
        ref, code = run_and_get_cpp_code(opt_fn, x, y, z)
        FileCheck().check("void kernel").check("*").check("+").check("loadu").check(
            "extension_device"
        ).run(code)
        self.assertEqual(metrics.generated_kernel_count, 1)

        res = fn(x, y, z)
        self.assertEqual(ref, res.to(device="cpu"))

    def test_open_device_registration(self):
        self.assertTrue(
            get_scheduling_for_device("extension_device") == ExtensionScheduling
        )
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device")
            == ExtensionWrapperCodegen
        )

        self.assertFalse(self.module.custom_op_called())
        device = self.module.custom_device()
        x = torch.empty(2, 16).to(device=device).fill_(1)
        self.assertTrue(self.module.custom_op_called())
        y = torch.empty(2, 16).to(device=device).fill_(2)
        z = torch.empty(2, 16).to(device=device).fill_(3)
        ref = torch.empty(2, 16).fill_(5)

        self.assertTrue(x.device == device)
        self.assertTrue(y.device == device)
        self.assertTrue(z.device == device)

        def fn(a, b, c):
            return a * b + c

        metrics.reset()
        opt_fn = torch.compile()(fn)
        _, code = run_and_get_cpp_code(opt_fn, x, y, z)
        FileCheck().check("void kernel").check("loadu").check("extension_device").run(
            code
        )
        opt_fn(x, y, z)
        res = opt_fn(x, y, z)
        self.assertEqual(ref, res.to(device="cpu"))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    # cpp_extension doesn't work in fbcode right now
    if HAS_CPU and not IS_MACOS and not IS_FBCODE:
        run_tests(needs="filelock")
