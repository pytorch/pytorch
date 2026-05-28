#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest


os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"

from helpers.comm_test_helpers import skip_if_torch_compile_not_supported_or_enabled
from integration.helpers.TorchCommTestHelpers import skip_backend, TorchCommTestWrapper

import torch


try:
    import torch.comms
except ImportError:
    pass  # skip test down below will catch this


@skip_if_torch_compile_not_supported_or_enabled()
class FullgraphCompileAutogradTest(unittest.TestCase):
    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        import torch._dynamo
        import torch._inductor.codecache
        import torch._inductor.config
        import torch._inductor.utils

        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear()
        torch._inductor.utils.clear_caches()

        torch._inductor.config.debug = True
        torch._inductor.config.trace.enabled = True

        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def test_wait_tensors_functional_backward(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        y = torch.randn(3, requires_grad=True, device=self.device)

        result = torch.ops.torchcomms.torchcomm_wait_tensors([x, y])

        loss = result[0].sum() + result[1].sum()
        loss.backward()

        torch.testing.assert_close(x.grad.cpu(), torch.ones(3))
        torch.testing.assert_close(y.grad.cpu(), torch.ones(3))

    def test_wait_tensors_inplace_backward(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        y = torch.randn(3, requires_grad=True, device=self.device)

        x_clone = x.clone()
        y_clone = y.clone()

        result = torch.ops.torchcomms.torchcomm_wait_tensors_([x_clone, y_clone])

        loss = result[0].sum() + result[1].sum()
        loss.backward()

        torch.testing.assert_close(x.grad.cpu(), torch.ones(3))
        torch.testing.assert_close(y.grad.cpu(), torch.ones(3))

    def test_wait_tensors_grad_chain(self):
        x = torch.randn(3, requires_grad=True, device=self.device)

        y = x * 2
        waited = torch.ops.torchcomms.torchcomm_wait_tensors([y])
        z = waited[0] + 1
        loss = z.sum()
        loss.backward()

        torch.testing.assert_close(x.grad.cpu(), torch.full((3,), 2.0))

    def test_async_tensor_grad_fn_delegation(self):
        from torch.comms.functional.async_tensor import TorchCommsAsyncTensor

        x = torch.randn(3, requires_grad=True, device=self.device)
        y = x * 2

        async_tensor = TorchCommsAsyncTensor(y, None)

        self.assertEqual(async_tensor.grad_fn, y.grad_fn)

    def test_async_tensor_backward(self):
        from torch.comms.functional.async_tensor import TorchCommsAsyncTensor

        x = torch.randn(3, requires_grad=True, device=self.device)
        y = x * 2

        async_tensor = TorchCommsAsyncTensor(y, None)

        loss = async_tensor.sum()
        loss.backward()

        torch.testing.assert_close(x.grad.cpu(), torch.full((3,), 2.0))

    def test_async_tensor_is_leaf(self):
        from torch.comms.functional.async_tensor import TorchCommsAsyncTensor

        x = torch.randn(3, requires_grad=True, device=self.device)
        async_x = TorchCommsAsyncTensor(x, None)
        self.assertTrue(async_x.is_leaf)

        y = x * 2
        async_y = TorchCommsAsyncTensor(y, None)
        self.assertFalse(async_y.is_leaf)

    def test_async_tensor_grad_setter(self):
        from torch.comms.functional.async_tensor import TorchCommsAsyncTensor

        x = torch.randn(3, requires_grad=True, device=self.device)
        async_x = TorchCommsAsyncTensor(x, None)

        grad = torch.ones(3, device=self.device)
        async_x.grad = grad

        torch.testing.assert_close(x.grad.cpu(), grad.cpu())
        torch.testing.assert_close(async_x.grad.cpu(), grad.cpu())

    def test_wait_tensors_gradient_accumulation(self):
        x = torch.randn(3, requires_grad=True, device=self.device)

        y1 = torch.ops.torchcomms.torchcomm_wait_tensors([x.clone()])[0]
        loss1 = y1.sum()
        loss1.backward()
        grad1 = x.grad.clone()

        y2 = torch.ops.torchcomms.torchcomm_wait_tensors([x.clone()])[0]
        loss2 = y2.sum()
        loss2.backward()

        torch.testing.assert_close(x.grad.cpu(), (grad1 * 2).cpu())

    def test_wait_tensors_requires_grad_true(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        result = torch.ops.torchcomms.torchcomm_wait_tensors([x])
        self.assertTrue(result[0].requires_grad)

    def test_wait_tensors_requires_grad_false(self):
        x = torch.randn(3, requires_grad=False, device=self.device)
        result = torch.ops.torchcomms.torchcomm_wait_tensors([x])
        self.assertFalse(result[0].requires_grad)

    def test_wait_tensors_mixed_requires_grad(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        y = torch.randn(3, requires_grad=False, device=self.device)
        result = torch.ops.torchcomms.torchcomm_wait_tensors([x, y])

        self.assertTrue(len(result) == 2)
        for t in result:
            self.assertTrue(t.requires_grad)

    def test_wait_tensors_retain_graph(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        y = torch.ops.torchcomms.torchcomm_wait_tensors([x * 2])[0]
        loss = y.sum()

        loss.backward(retain_graph=True)
        grad1 = x.grad.clone()

        x.grad = None
        loss.backward()
        grad2 = x.grad

        torch.testing.assert_close(grad1.cpu(), grad2.cpu())

    def test_wait_tensors_create_graph(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        y = torch.ops.torchcomms.torchcomm_wait_tensors([x * x])[0]
        loss = y.sum()

        (grad,) = torch.autograd.grad(loss, x, create_graph=True)

        self.assertTrue(grad.requires_grad)

        grad_sum = grad.sum()
        (grad2,) = torch.autograd.grad(grad_sum, x)

        torch.testing.assert_close(grad2.cpu(), torch.full((3,), 2.0))

    @skip_backend("gloo")
    def test_all_reduce_backward_eager(self):
        from torch.comms import ReduceOp

        x = torch.randn(4, requires_grad=True, device=self.device)

        b = x * 1.0
        b = self.torchcomm.all_reduce(b, ReduceOp.SUM, async_op=True)
        loss = b.sum()
        loss.backward()

        expected_grad = torch.full((4,), float(self.num_ranks), device=self.device)
        torch.testing.assert_close(x.grad.cpu(), expected_grad.cpu())

    @skip_backend("gloo")
    def test_all_gather_backward_eager(self):
        x = torch.randn(4, requires_grad=True, device=self.device)

        b = x * 1.0
        output = [torch.zeros(4, device=self.device) for _ in range(self.num_ranks)]
        output = self.torchcomm.all_gather(output, b, async_op=True)
        loss = sum(t.sum() for t in output)
        loss.backward()

        expected_grad = torch.ones(4) * self.num_ranks
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    @skip_backend("gloo")
    def test_reduce_scatter_backward_eager(self):
        from torch.comms import ReduceOp

        chunk_size = 4
        x = torch.randn(
            chunk_size * self.num_ranks, requires_grad=True, device=self.device
        )

        b = x * 1.0
        input_chunks = list(b.chunk(self.num_ranks))
        output = torch.zeros(chunk_size, device=self.device)
        output = self.torchcomm.reduce_scatter(
            output, input_chunks, ReduceOp.SUM, async_op=True
        )
        loss = output.sum()
        loss.backward()

        expected_grad = torch.ones(chunk_size * self.num_ranks)
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    @skip_backend("gloo")
    def test_all_gather_single_backward_eager(self):
        chunk_size = 4
        x = torch.randn(chunk_size, requires_grad=True, device=self.device)

        b = x * 1.0
        output = torch.zeros(chunk_size * self.num_ranks, device=self.device)
        output = self.torchcomm.all_gather_single(output, b, async_op=True)
        loss = output.sum()
        loss.backward()

        expected_grad = torch.ones(chunk_size) * self.num_ranks
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    @skip_backend("gloo")
    def test_reduce_scatter_single_backward_eager(self):
        from torch.comms import ReduceOp

        chunk_size = 4
        x = torch.randn(
            chunk_size * self.num_ranks, requires_grad=True, device=self.device
        )

        b = x * 1.0
        output = torch.zeros(chunk_size, device=self.device)
        output = self.torchcomm.reduce_scatter_single(
            output, b, ReduceOp.SUM, async_op=True
        )
        loss = output.sum()
        loss.backward()

        expected_grad = torch.ones(chunk_size * self.num_ranks)
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    @skip_backend("gloo")
    def test_all_to_all_single_backward_eager(self):
        chunk_size = 4
        x = torch.randn(
            chunk_size * self.num_ranks, requires_grad=True, device=self.device
        )

        b = x * 1.0
        output = torch.zeros(chunk_size * self.num_ranks, device=self.device)
        output = self.torchcomm.all_to_all_single(output, b, async_op=True)
        loss = output.sum()
        loss.backward()

        expected_grad = torch.ones(chunk_size * self.num_ranks)
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    @skip_backend("gloo")
    def test_all_to_all_v_single_backward_eager(self):
        # amount sent to rank j = j + 1, amount received from rank j = self.rank + 1
        # e.g.,
        # 0: sends [1,2,3,4], receives [1,1,1,1]
        # 1: sends [1,2,3,4], receives [2,2,2,2]
        # 2: sends [1,2,3,4], receives [3,3,3,3]
        # 3: sends [1,2,3,4], receives [4,4,4,4]

        input_split_sizes = [j + 1 for j in range(self.num_ranks)]
        output_split_sizes = [self.rank + 1] * self.num_ranks

        total_input = sum(input_split_sizes)
        total_output = sum(output_split_sizes)

        x = torch.randn(total_input, requires_grad=True, device=self.device)

        b = x * 1.0
        output = torch.zeros(total_output, device=self.device)
        output = self.torchcomm.all_to_all_v_single(
            output, b, output_split_sizes, input_split_sizes, async_op=True
        )
        loss = output.sum()
        loss.backward()

        expected_grad = torch.ones(total_input)
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    @skip_backend("gloo")
    def test_all_reduce_async_backward_eager(self):
        from torch.comms import ReduceOp

        x = torch.randn(4, requires_grad=True, device=self.device)

        b = x * 1.0
        b = self.torchcomm.all_reduce(b, ReduceOp.SUM, async_op=True)
        loss = b.sum()
        loss.backward()

        expected_grad = torch.full((4,), float(self.num_ranks), device=self.device)
        torch.testing.assert_close(x.grad.cpu(), expected_grad.cpu())

    def test_wait_tensors_functional_backward_compiled(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        y = torch.randn(3, requires_grad=True, device=self.device)

        def fn(a, b):
            result = torch.ops.torchcomms.torchcomm_wait_tensors([a, b])
            loss = result[0].sum() + result[1].sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x, y)
        loss.backward()

        torch.testing.assert_close(x.grad.cpu(), torch.ones(3))
        torch.testing.assert_close(y.grad.cpu(), torch.ones(3))

    def test_wait_tensors_inplace_backward_compiled(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        y = torch.randn(3, requires_grad=True, device=self.device)

        def fn(a, b):
            a_clone = a.clone()
            b_clone = b.clone()
            result = torch.ops.torchcomms.torchcomm_wait_tensors_([a_clone, b_clone])
            loss = result[0].sum() + result[1].sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x, y)
        loss.backward()

        torch.testing.assert_close(x.grad.cpu(), torch.ones(3))
        torch.testing.assert_close(y.grad.cpu(), torch.ones(3))

    def test_wait_tensors_grad_chain_compiled(self):
        x = torch.randn(3, requires_grad=True, device=self.device)

        def fn(a):
            b = a * 2
            waited = torch.ops.torchcomms.torchcomm_wait_tensors([b])
            c = waited[0] + 1
            loss = c.sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        torch.testing.assert_close(x.grad.cpu(), torch.full((3,), 2.0))

    def test_wait_tensors_requires_grad_propagation_compiled(self):
        x = torch.randn(3, requires_grad=True, device=self.device)
        y = torch.randn(3, requires_grad=False, device=self.device)

        def fn(a, b):
            return torch.ops.torchcomms.torchcomm_wait_tensors([a, b])

        compiled_fn = torch.compile(fn, fullgraph=True)
        output = compiled_fn(x, y)

        self.assertTrue(len(output) == 2)
        for t in output:
            self.assertTrue(t.requires_grad)

    def test_gradient_accumulation_compiled(self):
        x = torch.randn(3, requires_grad=True, device=self.device)

        def fn(a):
            b = torch.ops.torchcomms.torchcomm_wait_tensors([a.clone()])[0]
            loss = b.sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)

        loss1 = compiled_fn(x)
        loss1.backward()
        grad1 = x.grad.clone()

        loss2 = compiled_fn(x)
        loss2.backward()

        torch.testing.assert_close(x.grad.cpu(), (grad1 * 2).cpu())

    def test_retain_graph_compiled(self):
        x = torch.randn(3, requires_grad=True, device=self.device)

        def fn(a):
            b = a * 2
            waited = torch.ops.torchcomms.torchcomm_wait_tensors([b])
            loss = waited[0].sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)

        loss.backward(retain_graph=True)
        grad1 = x.grad.clone()

        x.grad = None
        loss.backward()
        grad2 = x.grad

        torch.testing.assert_close(grad1.cpu(), grad2.cpu())

    def test_all_reduce_backward_compiled(self):
        from torch.comms import ReduceOp

        x = torch.randn(4, requires_grad=True, device=self.device)

        def fn(a):
            b = a * 1.0
            self.torchcomm.all_reduce(b, ReduceOp.SUM, async_op=False)
            loss = b.sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        expected_grad = torch.full((4,), float(self.num_ranks), device=self.device)
        torch.testing.assert_close(x.grad.cpu(), expected_grad.cpu())

    def test_all_gather_backward_compiled(self):
        x = torch.randn(4, requires_grad=True, device=self.device)

        def fn(a):
            b = a * 1.0
            output = [torch.zeros(4, device=self.device) for _ in range(self.num_ranks)]
            self.torchcomm.all_gather(output, b, async_op=False)
            loss = sum(t.sum() for t in output)
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        expected_grad = torch.ones(4) * self.num_ranks
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    def test_reduce_scatter_backward_compiled(self):
        from torch.comms import ReduceOp

        chunk_size = 4
        x = torch.randn(
            chunk_size * self.num_ranks, requires_grad=True, device=self.device
        )

        def fn(a):
            b = a * 1.0
            input_chunks = list(b.chunk(self.num_ranks))
            output = torch.zeros(chunk_size, device=self.device)
            self.torchcomm.reduce_scatter(
                output, input_chunks, ReduceOp.SUM, async_op=False
            )
            loss = output.sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        expected_grad = torch.ones(chunk_size * self.num_ranks)
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    def test_all_gather_single_backward_compiled(self):
        chunk_size = 4
        x = torch.randn(chunk_size, requires_grad=True, device=self.device)

        def fn(a):
            b = a * 1.0
            output = torch.zeros(chunk_size * self.num_ranks, device=self.device)
            self.torchcomm.all_gather_single(output, b, async_op=False)
            loss = output.sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        expected_grad = torch.ones(chunk_size) * self.num_ranks
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    def test_reduce_scatter_single_backward_compiled(self):
        from torch.comms import ReduceOp

        chunk_size = 4
        x = torch.randn(
            chunk_size * self.num_ranks, requires_grad=True, device=self.device
        )

        def fn(a):
            b = a * 1.0
            output = torch.zeros(chunk_size, device=self.device)
            self.torchcomm.reduce_scatter_single(
                output, b, ReduceOp.SUM, async_op=False
            )
            loss = output.sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        expected_grad = torch.ones(chunk_size * self.num_ranks)
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    def test_all_to_all_single_backward_compiled(self):
        chunk_size = 4
        x = torch.randn(
            chunk_size * self.num_ranks, requires_grad=True, device=self.device
        )

        def fn(a):
            b = a * 1.0
            output = torch.zeros(chunk_size * self.num_ranks, device=self.device)
            self.torchcomm.all_to_all_single(output, b, async_op=False)
            loss = output.sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        expected_grad = torch.ones(chunk_size * self.num_ranks)
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    def test_all_to_all_v_single_backward_compiled(self):
        # amount sent to rank j = j + 1, amount received from rank j = self.rank + 1
        # e.g.,
        # 0: sends [1,2,3,4], receives [1,1,1,1]
        # 1: sends [1,2,3,4], receives [2,2,2,2]
        # 2: sends [1,2,3,4], receives [3,3,3,3]
        # 3: sends [1,2,3,4], receives [4,4,4,4]

        input_split_sizes = [j + 1 for j in range(self.num_ranks)]
        output_split_sizes = [self.rank + 1] * self.num_ranks

        total_input = sum(input_split_sizes)
        total_output = sum(output_split_sizes)

        x = torch.randn(total_input, requires_grad=True, device=self.device)

        def fn(a):
            b = a * 1.0
            output = torch.zeros(total_output, device=self.device)
            self.torchcomm.all_to_all_v_single(
                output, b, output_split_sizes, input_split_sizes, async_op=False
            )
            loss = output.sum()
            return loss

        compiled_fn = torch.compile(fn, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        expected_grad = torch.ones(total_input)
        torch.testing.assert_close(x.grad.cpu(), expected_grad)


if __name__ == "__main__":
    unittest.main()
