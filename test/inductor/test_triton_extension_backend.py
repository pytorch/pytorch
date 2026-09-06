# Owner(s): ["module: inductor"]
import functools
import random
import string
import sys
import unittest
from pathlib import Path
from typing import Any

import torch
import torch._dynamo
import torch.utils.cpp_extension
from torch._inductor import config


try:
    from extension_backends.triton.device_interface import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:device_interface
        DeviceInterface,
    )
    from extension_backends.triton.extension_codegen_backend import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_codegen_backend
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
    from extension_backends.triton.extension_triton_heuristics import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_triton_heuristics
        EXTENSION_TRITON_META_FIELD,
    )
except ImportError:
    from .extension_backends.triton.device_interface import DeviceInterface
    from .extension_backends.triton.extension_codegen_backend import (
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
    from .extension_backends.triton.extension_triton_heuristics import (
        EXTENSION_TRITON_META_FIELD,
    )

import torch._inductor.lowering as inductor_lowering
from torch._C import FileCheck
from torch._dynamo import device_interface
from torch._dynamo.exc import TritonUnavailableError
from torch._inductor import codegen, ir, metrics
from torch._inductor.codegen import common
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    IndentedBuffer,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.utils import get_triton_code, run_and_get_triton_code
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_FBCODE,
    IS_MACOS,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_TRITON, TRITON_HAS_CPU
from torch.utils._triton import has_triton_package


try:
    from .test_extension_backend import BaseExtensionBackendTests
except ImportError:
    from test_extension_backend import BaseExtensionBackendTests

if has_triton_package():
    import triton
    import triton.language as tl


requires_triton_backend = unittest.skipUnless(HAS_TRITON, "Requires Triton backend.")


def mock_triton_hash_with_backend(*args, **kwargs):
    # Generate a random string of length 64. Used to mock the triton_hash_with_backend function
    # since we don't have a triton backend
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=64))


@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class TritonExtensionBackendTestBase(BaseExtensionBackendTests):
    """
    Test creating a backend for inductor with Triton scheduling.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if config.cpp_wrapper:
            raise unittest.SkipTest(
                "Not possible to fix until CppWrapperCpu supports triton for CPU"
            )

        # Store the default backends and reset later
        common.init_backend_registration()

        default_backend_patch = unittest.mock.patch.dict(inductor_lowering.lowerings)
        default_backend_patch.start()
        cls._default_backend_patch = default_backend_patch

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Restore the default backend.
        cls._default_backend_patch.stop()

    def _test_open_device_registration(self):
        torch._register_device_module("privateuseone", self.module)
        register_backend_for_device(
            "privateuseone", ExtensionScheduling, ExtensionWrapperCodegen
        )
        register_device_op_overrides("privateuseone", CPUDeviceOpOverrides())
        device_interface.register_interface_for_device("privateuseone", DeviceInterface)

        self.assertEqual(
            get_scheduling_for_device("privateuseone"), ExtensionScheduling
        )
        self.assertEqual(
            get_wrapper_codegen_for_device("privateuseone"), ExtensionWrapperCodegen
        )
        self.assertEqual(
            device_interface.get_interface_for_device("privateuseone"), DeviceInterface
        )

        device = torch.device("privateuseone")
        x = torch.empty(2, 16).fill_(1).to(device)

        def foo(x):
            return torch.sin(x) + x.min()

        metrics.reset()
        opt_fn = torch.compile(foo)

        # Since we don't have a triton backend, we need to mock the triton_hash_with_backend
        # function
        with unittest.mock.patch(
            "torch.utils._triton.triton_hash_with_backend",
            new=mock_triton_hash_with_backend,
        ):
            code = get_triton_code(opt_fn, x)

        FileCheck().check("import triton").check("@triton.jit").check(
            "tl_math.sin"
        ).check("device_str='privateuseone'").run(code)

    def _register_custom_backend_with_heuristics(self, device):
        path_to_ext_heuristics = str(
            Path(__file__).parent / "extension_backends" / "triton"
        )
        # Add the path to sys.path in the parent process so that the
        # ExtensionCachingAutotuner class (defined in extension_triton_heuristics)
        # can be resolved when the compiled kernel is unpickled from the
        # compile subprocess back into the parent process.
        if path_to_ext_heuristics not in sys.path:
            sys.path.append(path_to_ext_heuristics)
            self.addCleanup(sys.path.remove, path_to_ext_heuristics)

        class ExtensionTritonKernel(codegen.triton.TritonKernel):
            @classmethod
            @functools.lru_cache(None)
            def gen_common_triton_imports(cls) -> str:
                default_imports = super().gen_common_triton_imports()
                custom_imports = IndentedBuffer()
                custom_imports.splice(default_imports)

                custom_imports.splice("""
                    import extension_triton_heuristics as triton_heuristics
                """)
                return custom_imports

            @classmethod
            def triton_meta_common(cls) -> dict[str, Any]:
                triton_meta = super().triton_meta_common()
                triton_meta[EXTENSION_TRITON_META_FIELD] = True
                return triton_meta

        class ExtensionTritonScheduling(codegen.triton.TritonScheduling):
            kernel_type = ExtensionTritonKernel

        class ExtensionPythonWrapperCodegen(PythonWrapperCodegen):
            @classmethod
            def _get_triton_info_kernel_cls(cls) -> type[codegen.triton.TritonKernel]:
                return ExtensionTritonKernel

            @staticmethod
            def create(
                is_subgraph: bool,
                subgraph_name: str | None,
                parent_wrapper: PythonWrapperCodegen | None,
                partition_signatures: ir.GraphPartitionSignature | None = None,
            ):
                if is_subgraph:
                    if subgraph_name is None:
                        raise AssertionError
                    if parent_wrapper is None:
                        raise AssertionError
                    return PythonWrapperCodegen.create(
                        subgraph_name, parent_wrapper, partition_signatures
                    )
                return ExtensionPythonWrapperCodegen()

        register_backend_for_device(
            device, ExtensionTritonScheduling, ExtensionPythonWrapperCodegen
        )

    def _test_codegen_with_custom_heuristics_module(self, device):
        self._register_custom_backend_with_heuristics(device)

        def add(x, y):
            return x + y

        x = torch.zeros((32,), device=device)
        y = x
        compiled_add = torch.compile(add)

        code = run_and_get_triton_code(compiled_add, x, y)
        FileCheck().check("import extension_triton_heuristics").check(
            f"{EXTENSION_TRITON_META_FIELD}"
        ).check("@triton.jit").run(code)

    def _test_codegen_with_custom_heuristics_module_udtk(self, device):
        self._register_custom_backend_with_heuristics(device)

        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        args = [torch.randn(32, device=device) for _ in range(2)]
        code = run_and_get_triton_code(torch.compile(add), *args)

        FileCheck().check("import extension_triton_heuristics").check(
            "@triton.jit"
        ).run(code)


class TritonExtensionBackendGenericTests(TritonExtensionBackendTestBase):
    hw_classification = HardwareClassification.GENERIC

    def test_open_device_registration(self):
        self._test_open_device_registration()


@unittest.skipUnless(TRITON_HAS_CPU, "Requires Triton CPU backend.")
class TritonExtensionBackendCPUTests(TritonExtensionBackendTestBase):
    hw_classification = HardwareClassification.CPU

    def test_codegen_with_custom_heuristics_module(self, device):
        self._test_codegen_with_custom_heuristics_module(device)

    def test_codegen_with_custom_heuristics_module_udtk(self, device):
        self._test_codegen_with_custom_heuristics_module_udtk(device)


@unittest.skipUnless(has_triton_package(), "Requires Triton package.")
@unittest.skipIf(TRITON_HAS_CPU, "Triton CPU backend takes precedence.")
class TritonExtensionBackendAcceleratorTests(TritonExtensionBackendTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        device = cls.get_primary_device()
        if not HAS_TRITON:
            raise unittest.SkipTest(f"triton is required for {device}")
        try:
            interface = device_interface.get_interface_for_device(
                torch.device(device).type
            )
        except NotImplementedError as exc:
            raise unittest.SkipTest(f"requires Triton support for {device}") from exc
        if not interface.is_triton_capable(device):
            raise unittest.SkipTest(f"requires Triton support for {device}")
        try:
            interface.raise_if_triton_unavailable(device)
        except TritonUnavailableError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    @onlyAccelerator
    @requires_triton_backend
    def test_codegen_with_custom_heuristics_module(self, device):
        self._test_codegen_with_custom_heuristics_module(device)

    @onlyAccelerator
    @requires_triton_backend
    def test_codegen_with_custom_heuristics_module_udtk(self, device):
        self._test_codegen_with_custom_heuristics_module_udtk(device)


instantiate_device_type_tests(
    TritonExtensionBackendCPUTests,
    globals(),
    only_for=("cpu",),
)
instantiate_device_type_tests(
    TritonExtensionBackendAcceleratorTests,
    globals(),
    except_for=("cpu", "hpu"),
    allow_xpu=True,
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU and not IS_MACOS:
        run_tests()
