# Owner(s): ["module: cpp"]

import libtorch_agnostic  # noqa: F401

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLibtorchAgnostic(TestCase):
    @onlyCPU
    def test_slow_sgd(self, device):
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

    def test_not_my_abs(self, device):
        t = torch.rand(32, 16, device=device) - 1.0
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
    
    def test_my_ones_like(self, device):
        t = torch.rand(3, 1, device=device) - 1.0
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

    # @onlyCUDA
    # def test_z_delete_torch_lib(self, device):
    #     # Why the z + CUDA? THIS TEST MUST BE RUN LAST
    #     # We are testing that unloading the library properly deletes the registrations, so running this test
    #     # earlier will cause all other tests in this file to fail
    #     lib = libtorch_agnostic.loaded_lib

    #     # code for unloading a library inspired from
    #     # https://stackoverflow.com/questions/19547084/can-i-explicitly-close-a-ctypes-cdll
    #     lib_handle = lib._handle
    #     lib.dlclose(lib_handle)

    #     t = torch.tensor([-2.0, 0.5])
    #     with self.assertRaises(RuntimeError):
    #         libtorch_agnostic.ops.identity(
    #             t
    #         )  # errors as identity shouldn't be registered anymore


instantiate_device_type_tests(TestLibtorchAgnostic, globals(), except_for=None)

if __name__ == "__main__":
    run_tests()
