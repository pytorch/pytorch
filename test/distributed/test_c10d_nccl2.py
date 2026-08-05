# Owner(s): ["oncall: distributed"]
#
# Tests specific to the in-tree torchcomms NCCL backends.

import ctypes
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReconfigureOptions
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    skip_if_rocm_ver_atleast_multiprocess,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


class ProcessGroupNCCL2Test(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_watchdog_does_not_release_python_backed_tensor(self) -> None:
        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.ones(4, device=self.device).as_subclass(TensorSubclass)
        outputs = [torch.empty(4, device=self.device) for _ in range(self.world_size)]
        work = dist.all_gather(outputs, tensor, async_op=True)
        del tensor
        del work

        torch.cuda.synchronize()
        time.sleep(2)
        dist.barrier()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_shared_options_type(self) -> None:
        self.assertIs(dist.ProcessGroupNCCL2.Options, dist.ProcessGroupNCCL.Options)
        opts = dist.ProcessGroupNCCL2.Options()
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 4
        self.assertEqual(opts.config.cga_cluster_size, 2)
        self.assertEqual(opts.config.max_ctas, 4)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduction_semantics(self) -> None:
        tensor = torch.ones(4, dtype=torch.bool, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.assertEqual(
            tensor.view(torch.uint8),
            torch.ones(4, dtype=torch.uint8, device=self.device),
        )

        with self.assertRaisesRegex(TypeError, "ReduceOp.AVG"):
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

        for dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
            tensor = torch.ones(4, device=self.device).to(dtype)
            with self.assertRaisesRegex(RuntimeError, "Unsupported Float8"):
                dist.all_reduce(tensor)

        tensor = torch.empty(4, dtype=torch.float4_e2m1fn_x2, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "Unsupported Float4"):
            dist.all_reduce(tensor)

    @requires_nccl()
    @requires_nccl_version((2, 24), "Need NCCL 2.24+ for Float8")
    @skip_if_lt_x_gpu(2)
    def test_float8_reduction(self) -> None:
        if torch.cuda.get_device_capability(self.device) < (9, 0):
            self.skipTest("Float8 reductions require sm90 or newer")
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            tensor = torch.ones(4, device=self.device).to(dtype)
            dist.all_reduce(tensor)
            self.assertEqual(tensor, torch.full_like(tensor, self.world_size))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_float4_transport(self) -> None:
        tensor = torch.full(
            (4,), self.rank + 1, dtype=torch.uint8, device=self.device
        ).view(torch.float4_e2m1fn_x2)
        dist.broadcast(tensor, src=0)
        self.assertEqual(
            tensor.view(torch.uint8),
            torch.ones(4, dtype=torch.uint8, device=self.device),
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ephemeral_timeout(self) -> None:
        dist.set_timeout(timedelta(seconds=3))

        existing_work = dist.all_reduce(
            torch.ones(4, device=self.device), async_op=True
        )
        dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(timedelta(seconds=10))
        self.assertEqual(existing_work.timeout, timedelta(seconds=3))

        tensor = torch.ones(4, device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        self.assertEqual(work.timeout, timedelta(seconds=13))
        existing_work.wait()
        work.wait()
        torch.cuda.synchronize(self.device)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            work = dist.all_reduce(tensor, async_op=True)
            if work.timeout == timedelta(seconds=3):
                work.wait()
                return
            work.wait()
            time.sleep(0.1)
        self.fail("ephemeral timeout was not reset after collective completion")


class _ProcessGroupNCCL2OptionsTest(MultiProcContinuousTest):
    """Base for groups initialized with backend specific options."""

    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    def _check_all_reduce(self) -> None:
        t = torch.full((4,), float(self.rank), device=self.device)
        dist.all_reduce(t)
        expected = float(sum(range(self.world_size)))
        self.assertEqual(t, torch.full((4,), expected, device=self.device))


class ProcessGroupNCCL2ShrinkTest(_ProcessGroupNCCL2OptionsTest):
    @requires_nccl()
    @requires_nccl_version((2, 27), "Need NCCL 2.27+ for communicator shrink")
    @skip_if_lt_x_gpu(2)
    def test_shrink_group(self) -> None:
        tensor = torch.ones(4, device=self.device)
        group = dist.new_group(device_id=self.device)
        dist.barrier(group=group)
        excluded = list(range(1, self.world_size))

        if self.rank in excluded:
            dist.destroy_process_group(group)
            return

        shrunk = dist.shrink_group(excluded, group=group)
        self.assertEqual(shrunk.size(), 1)
        dist.all_reduce(tensor, group=shrunk)
        self.assertEqual(tensor, torch.ones_like(tensor))
        dist.destroy_process_group(shrunk)


class ProcessGroupNCCL2ConfigTest(_ProcessGroupNCCL2OptionsTest):
    @classmethod
    def opts(cls, high_priority_stream=False):
        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 4
        return opts

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_with_config(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertEqual(backend.options.config.cga_cluster_size, 2)
        self.assertEqual(backend.options.config.max_ctas, 4)
        self.assertTrue(backend.options.is_high_priority_stream)
        self._check_all_reduce()


class ProcessGroupNCCL2NonblockingTest(_ProcessGroupNCCL2OptionsTest):
    @classmethod
    def opts(cls, high_priority_stream=False):
        opts = dist.ProcessGroupNCCL2.Options()
        opts.config.blocking = 0
        return opts

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collectives_with_nonblocking_communicator(self) -> None:
        self._check_all_reduce()

        tensor = torch.full((4,), float(self.rank), device=self.device)
        gathered = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        dist.gather(tensor, gathered, dst=0)
        if self.rank == 0:
            for rank, output in enumerate(gathered):
                self.assertEqual(output, torch.full_like(output, rank))

        output = torch.empty_like(tensor)
        scatter_list = (
            [torch.full_like(tensor, float(rank)) for rank in range(self.world_size)]
            if self.rank == 0
            else None
        )
        dist.scatter(output, scatter_list, src=0)
        self.assertEqual(output, tensor)


class ProcessGroupNCCLLegacyNonblockingTest(ProcessGroupNCCL2NonblockingTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-legacy"


class ProcessGroupNCCL2ScalableInitTest(_ProcessGroupNCCL2OptionsTest):
    ranks_per_root = 1

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = str(cls.ranks_per_root)
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_with_scalable_init(self) -> None:
        self._check_all_reduce()


class ProcessGroupNCCL2UnevenScalableInitTest(ProcessGroupNCCL2ScalableInitTest):
    world_size = 3
    ranks_per_root = 2

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_collective_with_scalable_init(self) -> None:
        self._check_all_reduce()


class ProcessGroupNCCL2ExpandableSegmentsTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_large_in_place_all_gather(self) -> None:
        numel = 16 * 1024 * 1024
        output = torch.empty(
            self.world_size * numel, dtype=torch.bfloat16, device=self.device
        )
        input = output.narrow(0, self.rank * numel, numel)
        input.fill_(self.rank)
        self.assertTrue(
            any(segment["is_expandable"] for segment in torch.cuda.memory_snapshot())
        )

        dist.all_gather_single(output, input)

        for rank, chunk in enumerate(output.chunk(self.world_size)):
            self.assertEqual(chunk, torch.full_like(chunk, rank))


class ProcessGroupNCCLLazyTest(ProcessGroupNCCL2Test):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-lazy"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reconfigure_not_supported(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertFalse(backend.supports_reconfigure)
        with self.assertRaisesRegex(
            RuntimeError, "does not support get_reconfigure_handle"
        ):
            backend.get_reconfigure_handle()
        with self.assertRaisesRegex(RuntimeError, "does not support reconfigure"):
            backend.reconfigure(ReconfigureOptions())

        opts = dist.ProcessGroupNCCL.Options()
        opts.enable_reconfigure = True
        with self.assertRaisesRegex(
            RuntimeError, "nccl-lazy does not support enable_reconfigure"
        ):
            dist.ProcessGroupNCCLLazy(
                dist.HashStore(), self.rank, self.world_size, opts
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_lazy_pair_channels(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        before_collective = backend._num_active_channels()
        t = torch.full((4,), 1.0, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertEqual(backend._num_active_channels(), before_collective)

        send_t = torch.full((4,), float(self.rank), device=self.device)
        recv_t = torch.empty((4,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(send_t, nxt)
            dist.recv(recv_t, prev)
        else:
            dist.recv(recv_t, prev)
            dist.send(send_t, nxt)
        torch.cuda.synchronize()
        self.assertEqual(recv_t, torch.full((4,), float(prev), device=self.device))

        expected = 1 if nxt == prev else 2
        self.assertGreaterEqual(backend._num_active_channels(), expected)


class ProcessGroupNCCLLazyNonblockingTest(ProcessGroupNCCL2NonblockingTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-lazy"


def _live_env(name: str) -> str | None:
    # os.environ is a snapshot taken at interpreter startup and does NOT observe
    # values set later by C++ via setenv (which is how the nccl2 backend forces
    # NCCL_ALGO under deterministic mode), so read the live environ via libc.
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    val = libc.getenv(name.encode())
    return val.decode() if val is not None else None


class _DeterministicNVLSTest(MultiProcContinuousTest):
    """Base for the nccl2 deterministic-mode NVLS-disable hook.

    Under torch deterministic mode, if the user has not set NCCL_ALGO the nccl2
    backend forces NCCL_ALGO=^NVLS (NVLS can give non-deterministic reductions),
    mirroring the stock ProcessGroupNCCL. The hook runs during PG init and the
    env var persists once set, so each scenario is its own subclass: every class
    gets a freshly spawned process pool (clean deterministic flag / NCCL_ALGO)
    and configures the env in _init_pg before the PG (and thus the hook) runs.
    """

    deterministic: bool = False
    preset_nccl_algo: str | None = None

    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        if cls.preset_nccl_algo is not None:
            os.environ["NCCL_ALGO"] = cls.preset_nccl_algo
        else:
            os.environ.pop("NCCL_ALGO", None)
        if cls.deterministic:
            torch.use_deterministic_algorithms(True)
        super()._init_pg(rank, world_size, rdvz_file)

    def _all_reduce_and_read_algo(self) -> str | None:
        # Force the (possibly lazy) nccl2 comm to initialize so the hook runs,
        # and confirm the reduction is numerically correct under the chosen algo.
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertEqual(t, torch.full_like(t, self.world_size))
        return _live_env("NCCL_ALGO")


class ProcessGroupNCCL2DeterministicNVLSDisabledTest(_DeterministicNVLSTest):
    deterministic = True

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_deterministic_disables_nvls(self) -> None:
        self.assertEqual(self._all_reduce_and_read_algo(), "^NVLS")


class ProcessGroupNCCL2DeterministicNVLSPresetTest(_DeterministicNVLSTest):
    deterministic = True
    preset_nccl_algo = "Ring"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_deterministic_respects_user_nccl_algo(self) -> None:
        self.assertEqual(self._all_reduce_and_read_algo(), "Ring")


class ProcessGroupNCCL2NonDeterministicNVLSTest(_DeterministicNVLSTest):
    deterministic = False

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_non_deterministic_does_not_force_nccl_algo(self) -> None:
        self.assertIsNone(self._all_reduce_and_read_algo())


if __name__ == "__main__":
    if TEST_CUDA:
        run_tests()
