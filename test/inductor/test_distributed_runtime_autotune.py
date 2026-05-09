# Owner(s): ["module: inductor"]

import os
import sys
import time

import torch
import torch.distributed as dist


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch._inductor import config
from torch._inductor.runtime import distributed_runtime_autotune as dra
from torch._inductor.test_case import run_tests
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)


class TestDistributedRuntimeAutotune(MultiProcessTestCase):
    """
    Test distributed runtime autotuning across 2 ranks.
    """

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        dra._AUTOTUNE_PG = None
        dra._COORDINATORS.clear()
        os.environ["LOCAL_WORLD_SIZE"] = str(self.world_size)
        self._spawn_processes()

    @skip_if_lt_x_gpu(1)
    @config.patch(
        coordinate_descent_tuning=True,
        distributed_runtime_autotune=True,
        distributed_runtime_autotune_max_wait_seconds=5,
        distributed_runtime_autotune_min_kernels=2,
        force_disable_caches=True,
    )
    def test_distributed_autotune_single_gpu(self):
        """
        Functional test if we have only one GPU: Two ranks share a single GPU
        and collaborate to perform distributed autotune using the gloo backend.
        """
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        os.environ["LOCAL_RANK"] = str(self.rank)

        device = "cuda:0"
        torch.cuda.set_device(device)

        class Model(torch.nn.Module):
            def forward(self, x):
                x = x - x.amax(dim=-1, keepdim=True)
                x = x.exp()
                x = x / x.sum(dim=-1, keepdim=True)
                x = torch.mm(x, x)
                x = x - x.mean(dim=-1, keepdim=True)
                x = torch.mm(x, x)
                return x.sum()

        model = Model().to(device)
        torch.manual_seed(42)
        x = torch.randn(256, 256, device=device, requires_grad=True)

        # Eager
        out_eager = model(x)
        out_eager.backward()
        grad_eager = x.grad.clone()
        x.grad = None

        # Compiled
        compiled = torch.compile(model)
        out_compiled = compiled(x)
        self.assertEqual(out_eager, out_compiled)

        out_compiled.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(grad_eager, x.grad)

        # Verify distributed autotuning succeeded with full results
        fwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/fwd")]
        bwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/bwd")]
        self.assertGreater(len(fwd_coord), 0)
        self.assertGreater(len(bwd_coord), 0)

        for coordinator in fwd_coord + bwd_coord:
            self.assertTrue(coordinator._autotuning_done)
            self.assertTrue(coordinator._autotuning_attempted)
            self.assertTrue(coordinator._autotuning_full_match)

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @config.patch(
        coordinate_descent_tuning=True,
        distributed_runtime_autotune=True,
        distributed_runtime_autotune_max_wait_seconds=5,
        distributed_runtime_autotune_min_kernels=2,
        force_disable_caches=True,
    )
    def test_distributed_autotune_multi_kernel(self):
        """
        Both ranks compile the same model that produces multiple kernels and
        then collaborate to perform distributed autotune.
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        os.environ["LOCAL_RANK"] = str(self.rank)

        device = f"cuda:{self.rank}"
        torch.cuda.set_device(device)

        class Model(torch.nn.Module):
            def forward(self, x):
                x = x - x.amax(dim=-1, keepdim=True)
                x = x.exp()
                x = x / x.sum(dim=-1, keepdim=True)
                x = torch.mm(x, x)
                x = x - x.mean(dim=-1, keepdim=True)
                x = torch.mm(x, x)
                return x.sum()

        model = Model().to(device)
        torch.manual_seed(42)
        x = torch.randn(256, 256, device=device, requires_grad=True)
        dist.broadcast(x, src=0)

        # Eager
        out_eager = model(x)
        out_eager.backward()
        grad_eager = x.grad.clone()
        x.grad = None

        # Compiled
        compiled = torch.compile(model)
        out_compiled = compiled(x)
        self.assertEqual(out_eager, out_compiled)

        out_compiled.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(grad_eager, x.grad)

        # Verify distributed autotuning succeeded with full results
        fwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/fwd")]
        bwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/bwd")]
        self.assertGreater(len(fwd_coord), 0)
        self.assertGreater(len(bwd_coord), 0)

        for coordinator in fwd_coord + bwd_coord:
            self.assertTrue(coordinator._autotuning_done)
            self.assertTrue(coordinator._autotuning_attempted)
            self.assertTrue(coordinator._autotuning_full_match)

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @config.patch(
        coordinate_descent_tuning=True,
        distributed_runtime_autotune=True,
        distributed_runtime_autotune_max_wait_seconds=5,
        distributed_runtime_autotune_min_kernels=1,
        force_disable_caches=True,
    )
    def test_distributed_autotune_single_kernel(self):
        """
        Both ranks compile a function that produces a single reduction Triton
        kernel. One rank autotunes it and the other receives the result.
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        os.environ["LOCAL_RANK"] = str(self.rank)

        device = f"cuda:{self.rank}"
        torch.cuda.set_device(device)

        def f(x):
            return x.sum(dim=-1)

        torch.manual_seed(42)
        x = torch.randn(256, 256, device=device, requires_grad=True)
        dist.broadcast(x, src=0)

        # Eager
        out_eager = f(x)
        loss_eager = out_eager.sum()
        loss_eager.backward()
        grad_eager = x.grad.clone()
        x.grad = None

        # Compiled
        compiled = torch.compile(f)
        out_compiled = compiled(x)
        self.assertEqual(out_eager, out_compiled)

        loss_compiled = out_compiled.sum()
        loss_compiled.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(grad_eager, x.grad)

        # Verify distributed autotuning succeeded with full results
        fwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/fwd")]
        self.assertGreater(len(fwd_coord), 0)

        for coordinator in fwd_coord:
            self.assertTrue(coordinator._autotuning_done)
            self.assertTrue(coordinator._autotuning_attempted)
            self.assertTrue(coordinator._autotuning_full_match)

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @config.patch(
        coordinate_descent_tuning=True,
        distributed_runtime_autotune=True,
        distributed_runtime_autotune_max_wait_seconds=5,
        distributed_runtime_autotune_min_kernels=1000,
        force_disable_caches=True,
    )
    def test_distributed_autotune_too_few_kernels(self):
        """
        Both ranks compile the same function, but the min_kernels threshold
        is set high enough that the leader rejects distributed autotuning.
        Both ranks should fall back to local autotuning.
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        os.environ["LOCAL_RANK"] = str(self.rank)

        device = f"cuda:{self.rank}"
        torch.cuda.set_device(device)

        def f(x):
            return x.sum(dim=-1)

        torch.manual_seed(42)
        x = torch.randn(256, 256, device=device, requires_grad=True)
        dist.broadcast(x, src=0)

        # Eager
        out_eager = f(x)
        loss_eager = out_eager.sum()
        loss_eager.backward()
        grad_eager = x.grad.clone()
        x.grad = None

        # Compiled
        compiled = torch.compile(f)
        out_compiled = compiled(x)
        self.assertEqual(out_eager, out_compiled)

        loss_compiled = out_compiled.sum()
        loss_compiled.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(grad_eager, x.grad)

        # Verify distributed autotuning was not attempted
        fwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/fwd")]
        self.assertGreater(len(fwd_coord), 0)

        for coordinator in fwd_coord:
            self.assertTrue(coordinator._autotuning_done)
            self.assertFalse(coordinator._autotuning_attempted)
            self.assertFalse(coordinator._autotuning_full_match)

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @config.patch(
        coordinate_descent_tuning=True,
        distributed_runtime_autotune=True,
        distributed_runtime_autotune_max_wait_seconds=5,
        distributed_runtime_autotune_min_kernels=1,
        force_disable_caches=True,
    )
    def test_distributed_autotune_kernel_mismatch(self):
        """
        The ranks compile different functions, causing kernel list mismatch.
        Distributed autotuning still proceeds; each rank autotunes its
        round-robin subset and uses whatever matching results come back.
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        os.environ["LOCAL_RANK"] = str(self.rank)

        device = f"cuda:{self.rank}"
        torch.cuda.set_device(device)

        # Two functions producing different reduction kernels.
        def fn_sum(x):
            return x.sum(dim=-1)

        def fn_amax(x):
            return x.amax(dim=-1)

        fn = fn_sum if self.rank == 0 else fn_amax
        torch.manual_seed(42)
        x = torch.randn(256, 256, device=device)

        # Eager
        out_eager = fn(x)

        # Compiled
        compiled = torch.compile(fn)
        out_compiled = compiled(x)
        self.assertEqual(out_eager, out_compiled)

        # Distributed autotuning was attempted despite mismatched kernel lists.
        fwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/fwd")]
        self.assertGreater(len(fwd_coord), 0)

        for coordinator in fwd_coord:
            self.assertTrue(coordinator._autotuning_done)
            self.assertTrue(coordinator._autotuning_attempted)

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @config.patch(
        coordinate_descent_tuning=True,
        distributed_runtime_autotune=True,
        distributed_runtime_autotune_max_wait_seconds=5,
        distributed_runtime_autotune_min_kernels=1,
        force_disable_caches=True,
    )
    def test_distributed_autotune_one_rank_compiles(self):
        """
        Only one rank compiles a reduction kernel. It should attempt distributed
        autotuning, but fall back to local when the other rank doesn't signal
        participation.
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        os.environ["LOCAL_RANK"] = str(self.rank)

        device = f"cuda:{self.rank}"
        torch.cuda.set_device(device)

        # Rank 0 compiles a reduction; rank 1 compiles pointwise (no reduction
        # kernels, so no coordinator is created and rank 1 never votes).
        def fn_reduction(x):
            return x.sum(dim=-1)

        def fn_pointwise(x):
            return x * 2 + 1

        fn = fn_reduction if self.rank == 0 else fn_pointwise
        torch.manual_seed(42)
        x = torch.randn(256, 256, device=device)

        # Eager
        out_eager = fn(x)

        # Compiled
        compiled = torch.compile(fn)
        out_compiled = compiled(x)
        self.assertEqual(out_eager, out_compiled)

        if self.rank == 0:
            # Verify coordinator was created but not attempted (other rank
            # never voted, so participation check failed)
            fwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/fwd")]
            self.assertGreater(len(fwd_coord), 0)
            for coordinator in fwd_coord:
                self.assertTrue(coordinator._autotuning_done)
                self.assertFalse(coordinator._autotuning_attempted)
                self.assertFalse(coordinator._autotuning_full_match)
        else:
            self.assertEqual(len(dra._COORDINATORS), 0)

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @config.patch(
        coordinate_descent_tuning=True,
        distributed_runtime_autotune=True,
        distributed_runtime_autotune_max_wait_seconds=0.1,
        distributed_runtime_autotune_min_kernels=1,
        force_disable_caches=True,
    )
    def test_distributed_autotune_late_participation(self):
        """
        One rank compiles immediately while the other delays. The leader should
        time out waiting for the late rank's vote. Both ranks should then fall
        back to local autotuning.
        """
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        os.environ["LOCAL_RANK"] = str(self.rank)

        device = f"cuda:{self.rank}"
        torch.cuda.set_device(device)

        def f(x):
            return x.sum(dim=-1)

        torch.manual_seed(42)
        x = torch.randn(256, 256, device=device, requires_grad=True)
        dist.broadcast(x, src=0)

        # Eager
        out_eager = f(x)
        loss_eager = out_eager.sum()
        loss_eager.backward()
        grad_eager = x.grad.clone()
        x.grad = None

        # Rank 1 sleeps so the leader (rank 0) times out waiting for its vote
        if self.rank == 1:
            time.sleep(3)

        # Compiled — both ranks fall back to local autotuning
        compiled = torch.compile(f)
        out_compiled = compiled(x)
        self.assertEqual(out_eager, out_compiled)

        loss_compiled = out_compiled.sum()
        loss_compiled.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(grad_eager, x.grad)

        # Verify distributed autotuning was not attempted (leader timed out)
        fwd_coord = [v for k, v in dra._COORDINATORS.items() if k.endswith("/fwd")]
        self.assertGreater(len(fwd_coord), 0)

        for coordinator in fwd_coord:
            self.assertTrue(coordinator._autotuning_done)
            self.assertFalse(coordinator._autotuning_attempted)
            self.assertFalse(coordinator._autotuning_full_match)

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
