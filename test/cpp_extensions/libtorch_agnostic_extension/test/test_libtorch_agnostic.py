# Owner(s): ["module: cpp"]

from pathlib import Path

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    IS_WINDOWS,
    run_tests,
    TestCase,
    xfailIfTorchDynamo,
)


# TODO: Fix this error in Windows:
# LINK : error LNK2001: unresolved external symbol PyInit__C
if not IS_WINDOWS:

    class TestLibtorchAgnostic(TestCase):
        @classmethod
        def setUpClass(cls):
            install_cpp_extension(extension_root=Path(__file__).parent.parent)

        @onlyCPU
        def test_slow_sgd(self, device):
            import libtorch_agnostic

            param = torch.rand(5, device=device)
            grad = torch.rand_like(param)
            weight_decay = 0.01
            lr = 0.001
            maximize = False

            new_param = libtorch_agnostic.ops.sgd_out_of_place(
                param, grad, weight_decay, lr, maximize
            )
            torch._fused_sgd_(
                (param,),
                (grad,),
                (),
                weight_decay=weight_decay,
                momentum=0.0,
                lr=lr,
                dampening=0.0,
                nesterov=False,
                maximize=maximize,
                is_first_step=False,
            )
            self.assertEqual(new_param, param)

        @onlyCUDA
        def test_identity_does_not_hog_memory(self, device):
            import libtorch_agnostic

            def _run_identity(prior_mem):
                t = torch.rand(32, 32, device=device)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
                identi_t = libtorch_agnostic.ops.identity(t)
                assert identi_t is t

            init_mem = torch.cuda.memory_allocated(device)

            for _ in range(3):
                _run_identity(init_mem)
                curr_mem = torch.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

        def test_exp_neg_is_leaf(self, device):
            import libtorch_agnostic

            t1 = torch.rand(2, 3, device=device)
            t2 = torch.rand(3, 2, device=device)
            t3 = torch.rand(2, device=device)

            exp, neg, is_leaf = libtorch_agnostic.ops.exp_neg_is_leaf(t1, t2, t3)
            self.assertEqual(exp, torch.exp(t1))
            self.assertEqual(neg, torch.neg(t2))
            self.assertEqual(is_leaf, t3.is_leaf)

        def test_my_abs(self, device):
            import libtorch_agnostic

            t = torch.rand(32, 16, device=device) - 0.5
            cpu_t = libtorch_agnostic.ops.my_abs(t)
            self.assertEqual(cpu_t, torch.abs(t))

            def _make_cuda_tensors(prior_mem):
                cuda_t = libtorch_agnostic.ops.my_abs(t)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
                self.assertEqual(cuda_t, torch.abs(t))

            if t.is_cuda:
                init_mem = torch.cuda.memory_allocated(device)
                for _ in range(3):
                    _make_cuda_tensors(init_mem)
                    curr_mem = torch.cuda.memory_allocated(device)
                    self.assertEqual(curr_mem, init_mem)

        # TODO: Debug this:
        # torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
        # call_function libtorch_agnostic.my_ones_like.default(*(FakeTensor(..., size=(3, 1)), 'cpu'),
        # **{}): got AssertionError("tensor's device must be `meta`, got cpu instead")
        @xfailIfTorchDynamo
        def test_my_ones_like(self, device):
            import libtorch_agnostic

            t = torch.rand(3, 1, device=device) - 0.5
            cpu_t = libtorch_agnostic.ops.my_ones_like(t, "cpu")
            self.assertEqual(cpu_t, torch.ones_like(t, device="cpu"))

            def _make_cuda_tensors(prior_mem):
                cuda_t = libtorch_agnostic.ops.my_ones_like(t, device)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
                self.assertEqual(cuda_t, torch.ones_like(t, device=device))

            if t.is_cuda:
                init_mem = torch.cuda.memory_allocated(device)
                for _ in range(3):
                    _make_cuda_tensors(init_mem)
                    curr_mem = torch.cuda.memory_allocated(device)
                    self.assertEqual(curr_mem, init_mem)

        @onlyCUDA
        def test_z_delete_torch_lib(self, device):
            import libtorch_agnostic

            # Why the z + CUDA? THIS TEST MUST BE RUN LAST
            # We are testing that unloading the library properly deletes the registrations, so running this test
            # earlier will cause all other tests in this file to fail
            lib = libtorch_agnostic.loaded_lib

            # code for unloading a library inspired from
            # https://stackoverflow.com/questions/19547084/can-i-explicitly-close-a-ctypes-cdll
            lib_handle = lib._handle
            lib.dlclose(lib_handle)

            t = torch.tensor([-2.0, 0.5])
            with self.assertRaises(RuntimeError):
                libtorch_agnostic.ops.identity(
                    t
                )  # errors as identity shouldn't be registered anymore

    instantiate_device_type_tests(TestLibtorchAgnostic, globals(), except_for=None)

if __name__ == "__main__":
    run_tests()
