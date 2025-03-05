# Owner(s): ["module: inductor"]
import os
import tempfile
from typing import Any, Callable

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.runtime.triton_compat import tl, triton
from torch._inductor.runtime.triton_heuristics import StaticallyLaunchedCudaKernel
from torch._inductor.test_case import TestCase
from torch.testing._internal.triton_utils import requires_cuda


@requires_cuda
class TestStaticCudaLauncher(TestCase):

    def setUp(self):
        # Create a temporary file to store the cubin.
        # We set delete=False so that the file persists after closing.
        self.tmp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        self.tmp_file.close()  # Close now; we'll open it for writing later.
        super().setUp()

    def tearDown(self):
        super().tearDown()
        # Delete the temporary cubin file.
        if os.path.exists(self.tmp_file.name):
            os.remove(self.tmp_file.name)

    def _make_launcher(
        self, kernel: Callable, args: tuple[Any, ...], grid: tuple[Any, ...] = (1,)
    ) -> StaticallyLaunchedCudaKernel:
        """
        Compiles a Triton kernel with the provided *args,
        writes its cubin to the temporary file, and returns the file path.
        """
        fn = triton.jit(kernel)
        # Launch the kernel to trigger compilation.
        # Forcing a 1-element launch; the returned object contains the compiled info.
        compiled_kernel = fn[grid](*args)
        result = StaticallyLaunchedCudaKernel(compiled_kernel)
        result.write_cubin_to_file(self.tmp_file.name)
        return result

    def test_basic(self):
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        arg1 = 5
        args = (arg0, arg1)

        launcher = self._make_launcher(simple_kernel, args, (1,))
        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_ty_from_signature(), "Oi")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.loadAndRun((1,), stream, (new_arg0, arg1))
        self.assertEqual(new_arg0, arg0)

    def test_basic_1arg(self):
        def simple_kernel_1_arg(arg0):
            x = tl.load(arg0)
            tl.store(arg0, x + 1)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        launcher = self._make_launcher(simple_kernel_1_arg, (arg0,), (1,))
        self.assertEqual(arg0, torch.tensor([1], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_ty_from_signature(), "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.loadAndRun((1,), stream, (new_arg0,))
        self.assertEqual(new_arg0, arg0)

    def test_too_many_args(self):
        def kernel_too_many_args(
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11
        ):
            x = tl.load(arg0)
            y = (
                arg1
                + arg2
                + arg3
                + arg4
                + arg5
                + arg6
                + arg7
                + arg8
                + arg9
                + arg10
                + arg11
            )
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        scalar_args = (1,) * 11
        self.assertRaisesRegex(
            NotImplementedError,
            "No static cuda launcher available",
            lambda: self._make_launcher(kernel_too_many_args, (arg0, *scalar_args)),
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
