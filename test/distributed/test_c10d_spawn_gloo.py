# Owner(s): ["oncall: distributed"]

import copy
import os
import tempfile

from test_c10d_spawn import TestDistributedNNFunctions

import torch
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import (
    requires_gloo,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


class DistributedDataParallelSingleProcessTest(TestCase):
    def setUp(self):
        super().setUp()
        self.rank = 0
        self.world_size = 1
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f

    def tearDown(self):
        try:
            os.remove(self.file.name)
        except OSError:
            pass

    def _test_base(self, net, inp, check_allclose=True):
        store = c10d.FileStore(self.file.name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        if inp[0].is_cuda:
            device_ids = [torch.cuda.current_device()]
        else:
            device_ids = None

        ddp = nn.parallel.DistributedDataParallel(
            copy.deepcopy(net), device_ids=device_ids, process_group=process_group
        )

        net_opt = torch.optim.Adam(net.parameters(), lr=0.001)
        ddp_opt = torch.optim.Adam(ddp.parameters(), lr=0.001)

        for i, j in zip(ddp.parameters(), net.parameters()):
            self.assertTrue(i.allclose(j))

        for _ in range(10):
            net_out = net(*inp)
            ddp_out = ddp(*inp)

            net_out.sum().backward()
            ddp_out.sum().backward()

            net_opt.step()
            ddp_opt.step()

        if check_allclose:
            for i, j in zip(ddp.parameters(), net.parameters()):
                self.assertTrue(i.allclose(j))

    @requires_gloo()
    def test_cpu(self):
        self._test_base(nn.Linear(2, 2), [torch.randn(30, 2)])

    @requires_gloo()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "At least 1 CUDA GPUS needed")
    def test_cuda(self):
        self._test_base(nn.Linear(2, 2).to(0), [torch.randn(30, 2).to(0)])

    @requires_gloo()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "At least 1 CUDA GPUS needed")
    def test_rnn(self):
        # This test is inspired by the bug reported in
        # https://github.com/pytorch/pytorch/issues/36268
        BATCH_SIZE = 12  # Divisible by 2, 3, 4
        INPUT_DIM = 256
        OUTPUT_DIM = 256
        HIDDEN_DIM = 256
        N_LAYERS = 3
        SEQ_LEN = 100

        class Net(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.hidden_layers = hidden_layers

                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, hidden_layers, batch_first=True
                )
                self.h2o = nn.Linear(hidden_dim, output_dim)

            def forward(self, x, y):
                self.lstm.flatten_parameters()
                h_t, _ = self.lstm(x)
                output = self.h2o(h_t)
                loss = nn.functional.mse_loss(output, y)
                return loss

        net = Net(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS).to(0)
        inp = [
            torch.randn((BATCH_SIZE, SEQ_LEN, INPUT_DIM)).to(0),
            torch.rand((BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)).to(0),
        ]

        # Not checking result allclose as the parameter inconsistency exist
        # prior to this change. See #37079
        self._test_base(net, inp, check_allclose=False)


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsGloo(TestDistributedNNFunctions):
        BACKEND = "gloo"

        def setUp(self):
            if not c10d.is_gloo_available():
                self.skipTest("c10d was not compiled with the Gloo backend")
            if torch.cuda.is_available() and torch.cuda.device_count() < 2:
                self.skipTest("Need at least 2 CUDA GPUs")
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            super().setUp()

        @requires_gloo()
        @skip_but_pass_in_sandcastle_if(
            not c10d.is_gloo_available(), "Gloo not available"
        )
        def test_broadcast_subgroup_with_nonzero_global_src(self):
            self._init_process_group()
            group = c10d.new_group([1])

            if self.rank == 1:
                x = torch.ones(5, 5, requires_grad=True)
                y = torch.distributed.nn.broadcast(x, 1, group=group)
                y.sum().backward()
                self.assertEqual(x.grad, torch.ones_like(x))

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        def test_gather(self):
            self._init_process_group()
            device = self._get_device()
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            tensors = torch.distributed.nn.gather(x, 1)
            if self.rank == 1:
                for i, t in enumerate(tensors):
                    self.assertEqual(t, torch.ones(5, 5, device=device) + i)
            elif self.rank == 0:
                for t in tensors:
                    zeros = torch.zeros(5, 5, device=device)
                    self.assertEqual(t, zeros)
            y = torch.sum(torch.stack(tensors), axis=0)
            z = y.sin().sum()
            z.backward()

            x_s = 3 * torch.ones(5, 5, device=device)
            self.assertEqual(x.grad, x_s.cos())

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        def test_scatter(self):
            self._init_process_group()
            device = self._get_device()
            x0 = torch.ones(5, 5, device=device)
            x1 = torch.ones(5, 5, device=device) + 1
            x0.requires_grad = True
            x1.requires_grad = True

            y = torch.distributed.nn.scatter([x0, x1], 1)
            if self.rank == 1:
                self.assertEqual(y, 1 + torch.ones(5, 5, device=device))
            elif self.rank == 0:
                self.assertEqual(y, torch.ones(5, 5, device=device))
            z = y.sin().sum()
            z.backward()

            if self.rank == 1:
                x0_s = torch.ones(5, 5, device=device).cos()
                x1_s = (2 * torch.ones(5, 5, device=device)).cos()
                self.assertEqual(x0.grad, x0_s)
                self.assertEqual(x1.grad, x1_s)
            if self.rank == 0:
                self.assertEqual(x0.grad, torch.zeros(5, 5, device=device))


if __name__ == "__main__":
    run_tests()
