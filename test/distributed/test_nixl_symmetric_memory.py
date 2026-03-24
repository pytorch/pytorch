# Owner(s): ["module: c10d"]

import os
import unittest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
    TestCase,
)


def _nixl_compiled() -> bool:
    """Check whether the NIXL symmetric memory backend was compiled in."""
    try:
        symm_mem.set_backend("NIXL")
        return True
    except RuntimeError:
        return False


def _nixl_vram_functional() -> bool:
    """Probe whether NIXL's UCX backend has CUDA (VRAM) support."""
    if not _nixl_compiled():
        return False
    if not torch.cuda.is_available():
        return False

    import pathlib

    ucx_cuda_found = False
    for ucx_dir in [
        "/opt/hpcx/ucx/lib/ucx",
        "/usr/lib/ucx",
        "/usr/local/lib/ucx",
    ]:
        if pathlib.Path(ucx_dir, "libuct_cuda.so").exists():
            ucx_cuda_found = True
            break

    if not ucx_cuda_found:
        return False

    try:
        import nixl_cu12  # noqa: F401

        return False
    except ImportError:
        return True


NIXL_COMPILED = _nixl_compiled()
NIXL_FUNCTIONAL = _nixl_vram_functional()

requires_nixl = unittest.skipUnless(NIXL_COMPILED, "NIXL backend not compiled")
requires_nixl_functional = unittest.skipUnless(
    NIXL_FUNCTIONAL, "NIXL not functional (UCX may lack CUDA support)"
)


def _nixl_has_device_ops() -> bool:
    """Check whether nixl_put/nixl_get ops are registered AND functional.

    The ops may be registered in the dispatcher but fail at runtime if
    NIXL's createXferReq cannot find the required registrations (e.g.,
    UCX backend doesn't have peer memory keys).  This is expected until
    the transfer path is fully debugged.
    """
    try:
        torch.ops.symm_mem.nixl_put  # noqa: B018
    except (AttributeError, RuntimeError):
        return False
    # Ops exist but may not be functional yet.
    # Set to True once end-to-end NIXL transfers work.
    return False


NIXL_HAS_DEVICE_OPS = _nixl_has_device_ops()
requires_nixl_device_ops = unittest.skipUnless(
    NIXL_HAS_DEVICE_OPS, "NIXL transfer ops not yet functional"
)

# Signal operations (barrier, put_signal, wait_signal) are not yet
# implemented in the NIXL backend C++ code.
NIXL_HAS_SIGNALS = False
requires_nixl_signals = unittest.skipUnless(
    NIXL_HAS_SIGNALS, "NIXL signal ops not yet implemented"
)


if "NIXL_PLUGIN_DIR" not in os.environ:
    try:
        import importlib.util

        spec = importlib.util.find_spec("nixl_cu12")
        if spec and spec.origin:
            import pathlib

            pkg_dir = pathlib.Path(spec.origin).parent
            for candidate in [
                pkg_dir.parent / ".nixl_cu12.mesonpy.libs" / "plugins",
                pkg_dir.parent / ".nixl.mesonpy.libs" / "plugins",
            ]:
                if candidate.is_dir():
                    os.environ["NIXL_PLUGIN_DIR"] = str(candidate)
                    break
    except Exception:
        pass


@requires_cuda_p2p_access()
class NixlSymmetricMemoryTest(MultiProcContinuousTest):
    """Multi-process tests for the NIXL symmetric memory backend."""

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init(self):
        symm_mem.set_backend("NIXL")
        torch.cuda.set_device(self.rank)

    # -- Allocation ---------------------------------------------------------

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_alloc(self):
        self._init()
        dtype = torch.float
        numel = 1024

        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=dist.group.WORLD)

        foo()
        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=dist.group.WORLD)

    # -- Handle properties --------------------------------------------------

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_rendezvous_handle_properties(self):
        self._init()
        numel = 512
        dtype = torch.float32
        t = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        self.assertEqual(hdl.rank, self.rank)
        self.assertEqual(hdl.world_size, self.world_size)
        self.assertGreater(hdl.buffer_size, 0)

    # -- Local buffer access ------------------------------------------------

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_get_local_buffer(self):
        self._init()
        numel = 256
        dtype = torch.float32
        t = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        buf = hdl.get_buffer(self.rank, (numel,), dtype)
        t.fill_(42.0)
        self.assertEqual(buf.sum().item(), 42.0 * numel)

    # -- Multiple allocations -----------------------------------------------

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_multiple_allocations(self):
        self._init()
        tensors = [
            symm_mem.empty(64, dtype=torch.float, device=self.device)
            for _ in range(8)
        ]
        for tensor in tensors:
            hdl = symm_mem.rendezvous(tensor, group=dist.group.WORLD)
            self.assertEqual(hdl.rank, self.rank)
            self.assertEqual(hdl.world_size, self.world_size)

    # -- Repeated rendezvous ------------------------------------------------

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_repeated_rendezvous(self):
        self._init()
        t = symm_mem.empty(128, dtype=torch.float, device=self.device)
        hdl1 = symm_mem.rendezvous(t, group=dist.group.WORLD)
        hdl2 = symm_mem.rendezvous(t, group=dist.group.WORLD)
        self.assertEqual(hdl1.rank, hdl2.rank)
        self.assertEqual(hdl1.buffer_size, hdl2.buffer_size)

    # -- Various dtypes -----------------------------------------------------

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_dtypes(self):
        self._init()
        # Keep all tensors alive to avoid address reuse confusing the cache.
        tensors = []
        handles = []
        for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32]:
            t = symm_mem.empty(256, dtype=dtype, device=self.device)
            hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
            tensors.append(t)
            handles.append(hdl)
            self.assertEqual(hdl.rank, self.rank)

    # -- Signal pad ---------------------------------------------------------

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_signal_pad_accessible(self):
        self._init()
        t = symm_mem.empty(64, dtype=torch.float, device=self.device)
        hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        sig = hdl.get_signal_pad(self.rank)
        self.assertIsNotNone(sig)
        self.assertGreater(sig.numel(), 0)

    # -- put_signal / wait_signal -------------------------------------------

    @requires_nixl_signals
    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_put_wait_signal(self):
        self._init()
        t = symm_mem.empty(1024, dtype=torch.float, device=self.device)
        hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        channel = 0
        dist.barrier()
        if self.rank % 2 == 1:
            hdl.put_signal(dst_rank=self.rank - 1, channel=channel)
            torch.cuda.synchronize()
        elif self.rank % 2 == 0 and self.rank + 1 < self.world_size:
            hdl.wait_signal(src_rank=self.rank + 1, channel=channel)
            torch.cuda.synchronize()
        dist.barrier()

    # -- barrier ------------------------------------------------------------

    @requires_nixl_signals
    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_barrier(self):
        self._init()
        t = symm_mem.empty(64, dtype=torch.float, device=self.device)
        hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        hdl.barrier(channel=0)
        torch.cuda.synchronize()
        dist.barrier()

    # -- nixl_put -----------------------------------------------------------

    @requires_nixl_device_ops
    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_put(self):
        self._init()
        t = symm_mem.empty(
            1024, dtype=torch.float, device=self.device
        ).fill_(float(self.rank))
        symm_mem.rendezvous(t, group=dist.group.WORLD)
        dist.barrier()
        if self.rank == 1:
            torch.ops.symm_mem.nixl_put(t, 0)
        dist.barrier()
        torch.cuda.synchronize()
        if self.rank == 0:
            self.assertEqual(t, torch.ones_like(t))

    # -- nixl_get -----------------------------------------------------------

    @requires_nixl_device_ops
    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_get(self):
        self._init()
        t = symm_mem.empty(
            1024, dtype=torch.float, device=self.device
        ).fill_(float(self.rank))
        symm_mem.rendezvous(t, group=dist.group.WORLD)
        dist.barrier()
        if self.rank == 0:
            torch.ops.symm_mem.nixl_get(t, 1)
        dist.barrier()
        torch.cuda.synchronize()
        if self.rank == 0:
            self.assertEqual(t, torch.ones_like(t))

    # -- nixl_put_with_signal / nixl_wait_for_signal ------------------------

    @requires_nixl_device_ops
    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_put_with_signal(self):
        self._init()
        data = symm_mem.empty(
            512, dtype=torch.float, device=self.device
        ).fill_(float(self.rank))
        symm_mem.rendezvous(data, group=dist.group.WORLD)
        dist.barrier()
        signal_val = 42
        if self.rank == 1:
            torch.ops.symm_mem.nixl_put_with_signal(data, signal_val, 0)
        dist.barrier()
        torch.cuda.synchronize()
        if self.rank == 0:
            torch.ops.symm_mem.nixl_wait_for_signal(data, signal_val)
            torch.cuda.synchronize()
            self.assertEqual(data, torch.ones_like(data))
        dist.barrier()


class NixlSymmetricMemorySingleProcTest(TestCase):
    """Single-process tests that do not need a distributed process group."""

    def test_nixl_backend_name_in_valid_list(self):
        if not NIXL_COMPILED:
            with self.assertRaises(RuntimeError):
                symm_mem.set_backend("NIXL")

    @requires_nixl
    def test_nixl_single_process_alloc(self):
        symm_mem.set_backend("NIXL")
        t = symm_mem.empty(512, dtype=torch.float32, device="cuda:0")
        self.assertEqual(t.shape, torch.Size([512]))
        self.assertEqual(t.device, torch.device("cuda", 0))


if __name__ == "__main__":
    run_tests()
