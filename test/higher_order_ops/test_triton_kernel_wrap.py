# Owner(s): ["module: higher order operators"]

from unittest import mock

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._higher_order_ops.triton_kernel_wrap import (
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestTritonKernelWrapperAutocast(TestCase):
    @parametrize("functional", [False, True])
    @parametrize("dtype", [torch.int32, torch.float32, torch.bfloat16])
    @parametrize("enabled", [False, True])
    def test_privateuse1_autocast_preserves_arguments(self, functional, dtype, enabled):
        op = (
            triton_kernel_wrapper_functional
            if functional
            else triton_kernel_wrapper_mutation
        )
        x = torch.arange(32, dtype=dtype)
        out = torch.empty_like(x)
        kwargs = {
            "kernel_idx": 0,
            "constant_args_idx": 0,
            "grid": [(1,)],
            "tma_descriptor_metadata": {},
            "kwargs": {"x": x, "out": out},
        }
        if functional:
            kwargs["tensors_to_clone"] = ["out"]

        key = DispatchKey.AutocastPrivateUse1
        keyset = DispatchKeySet(key)
        include = torch._C._dispatch_tls_local_include_set() | keyset
        exclude = torch._C._dispatch_tls_local_exclude_set()
        exclude = exclude - keyset if enabled else exclude | keyset
        device = torch._C._get_privateuse1_backend_name()
        previous = torch.is_autocast_enabled(device)
        result = {"out": out} if functional else None
        launch = mock.Mock(return_value=result)

        # Exercise the real HOP dispatcher with CPU tensors and the PrivateUse1
        # autocast key. Only replace the final launch, so no accelerator is needed.
        with (
            mock.patch.dict(
                op.py_kernels, {DispatchKey.CompositeExplicitAutograd: launch}
            ),
            mock.patch.object(op, "_dispatch_cache", {}),
            torch._C._ForceDispatchKeyGuard(include, exclude),
        ):
            self.assertIs(op(**kwargs), result)
            self.assertEqual(torch.is_autocast_enabled(device), enabled)

        launch.assert_called_once()
        forwarded = launch.call_args.kwargs["kwargs"]
        self.assertIs(forwarded["x"], x)
        self.assertIs(forwarded["out"], out)
        self.assertEqual(forwarded["x"].dtype, dtype)
        self.assertEqual(torch.is_autocast_enabled(device), previous)


if __name__ == "__main__":
    run_tests()
