# Owner(s): ["module: inductor"]
import functools
import random
import string
import unittest
from pathlib import Path
from typing import Any, Optional

import torch
import torch._dynamo
import torch.utils.cpp_extension
from torch._inductor import config


try:
    from extension_backends.triton.device_interface import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:device_interface  # noqa: B950
        DeviceInterface,
    )
    from extension_backends.triton.extension_codegen_backend import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_codegen_backend  # noqa: B950
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
    from extension_backends.triton.extension_triton_heuristics import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_triton_heuristics  # noqa: B950
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
from torch.testing._internal.common_utils import IS_FBCODE, IS_MACOS
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU_AND_TRITON,
    TRITON_HAS_CPU,
)


try:
    from .test_extension_backend import BaseExtensionBackendTests
except ImportError:
    from test_extension_backend import BaseExtensionBackendTests

if TRITON_HAS_CPU or HAS_GPU_AND_TRITON:
    import triton
    import triton.language as tl

    if TRITON_HAS_CPU:
        TRITON_DEVICE_TYPE = "cpu"
    else:
        TRITON_DEVICE_TYPE = GPU_TYPE

requires_triton_backend = unittest.skipUnless(
    HAS_GPU_AND_TRITON or TRITON_HAS_CPU, "Requires Triton backend."
)


def mock_triton_hash_with_backend(*args, **kwargs):
    # Generate a random string of length 64. Used to mock the triton_hash_with_backend function
    # since we don't have a triton backend
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=64))


@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class TritonExtensionBackendTests(BaseExtensionBackendTests):
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

    def test_open_device_registration(self):
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
        class ExtensionTritonKernel(codegen.triton.TritonKernel):
            @classmethod
            @functools.lru_cache(None)
            def gen_common_triton_imports(cls) -> str:
                default_imports = super().gen_common_triton_imports()
                custom_imports = IndentedBuffer()
                custom_imports.splice(default_imports)
                path_to_ext_heuristics = (
                    Path(__file__).parent / "extension_backends" / "triton"
                )

                custom_imports.splice(f"""
                    import sys
                    sys.path.append("{path_to_ext_heuristics}")
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
                subgraph_name: Optional[str],
                parent_wrapper: Optional[PythonWrapperCodegen],
                partition_signatures: Optional[ir.GraphPartitionSignature] = None,
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

    @requires_triton_backend
    def test_codegen_with_custom_heuristics_module(self):
        self._register_custom_backend_with_heuristics(TRITON_DEVICE_TYPE)

        def add(x, y):
            return x + y

        x = torch.zeros((32,), device=GPU_TYPE)
        y = x
        compiled_add = torch.compile(add)

        code = run_and_get_triton_code(compiled_add, x, y)
        FileCheck().check("import extension_triton_heuristics").check(
            f"{EXTENSION_TRITON_META_FIELD}"
        ).check("@triton.jit").run(code)

    @requires_triton_backend
    def test_codegen_with_custom_heuristics_module_udtk(self):
        self._register_custom_backend_with_heuristics(TRITON_DEVICE_TYPE)

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

        args = [torch.randn(32, device=GPU_TYPE) for _ in range(2)]
        code = run_and_get_triton_code(torch.compile(add), *args)

        FileCheck().check("import extension_triton_heuristics").check(
            "@triton.jit"
        ).run(code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU and not IS_MACOS:
        run_tests()
