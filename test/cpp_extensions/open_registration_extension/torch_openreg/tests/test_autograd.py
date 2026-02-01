# Owner(s): ["module: PrivateUse1"]

import os

import psutil
import torch
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfMPS,
    skipIfTorchDynamo,
    skipIfWindows,
    TestCase,
)


class TestAutograd(TestCase):
    # Support MPS and Windows platform later and fix torchdynamo issue
    @skipIfMPS
    @skipIfWindows()
    @skipIfTorchDynamo()
    def test_autograd_init(self):
        """Test autograd initialization and thread creation"""
        # Make sure autograd is initialized
        torch.ones(2, requires_grad=True, device="openreg").sum().backward()

        pid = os.getpid()
        task_path = f"/proc/{pid}/task"
        all_threads = psutil.Process(pid).threads()

        all_thread_names = set()

        for t in all_threads:
            with open(f"{task_path}/{t.id}/comm") as file:
                thread_name = file.read().strip()
            all_thread_names.add(thread_name)

        for i in range(torch.accelerator.device_count()):
            self.assertIn(f"pt_autograd_{i}", all_thread_names)

    def test_autograd_backward(self):
        """Test backward propagation on openreg device"""
        x = torch.randn(2, 3, device="openreg", requires_grad=True)
        y = torch.randn(3, 2, device="openreg", requires_grad=True)
        z = torch.mm(x, y)
        loss = z.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(y.grad.shape, y.shape)

    def test_autograd_grad_fn(self):
        """Test gradient function tracking"""
        x = torch.randn(2, 3, device="openreg", requires_grad=True)
        y = x * 2
        z = y.sum()

        self.assertIsNotNone(y.grad_fn)
        self.assertIsNone(x.grad)

        z.backward()
        self.assertIsNotNone(x.grad)

    def test_autograd_no_grad(self):
        """Test no_grad context manager"""
        x = torch.randn(2, 3, device="openreg", requires_grad=True)

        with torch.no_grad():
            y = x * 2
            self.assertIsNone(y.grad_fn)
            self.assertFalse(y.requires_grad)

        z = x * 2
        self.assertIsNotNone(z.grad_fn)
        self.assertTrue(z.requires_grad)

    def test_autograd_detach(self):
        """Test tensor detach"""
        x = torch.randn(2, 3, device="openreg", requires_grad=True)
        y = x.detach()

        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)

        z = y * 2
        self.assertIsNone(z.grad_fn)

    def test_autograd_retain_grad(self):
        """Test retain_grad for intermediate tensors"""
        x = torch.randn(2, 3, device="openreg", requires_grad=True)
        y = x * 2
        y.retain_grad()
        z = y.sum()
        z.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_autograd_multiple_backward(self):
        """Test multiple backward passes"""
        x = torch.randn(2, 3, device="openreg", requires_grad=True)
        y = x.sum()

        y.backward(retain_graph=True)
        grad1 = x.grad.clone()

        y.backward()
        grad2 = x.grad

        # Second backward should accumulate gradients
        self.assertEqual(grad2, grad1 * 2)

    def test_autograd_cross_device(self):
        """Test autograd with cross-device operations"""
        x = torch.randn(2, 3, requires_grad=True)
        y = x.to("openreg")
        z = y.sum()
        z.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.is_cpu)


if __name__ == "__main__":
    run_tests()
