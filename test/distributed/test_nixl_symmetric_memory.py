# Owner(s): ["module: c10d"]

import os
import pathlib
import unittest
import importlib.util

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


_NIXL_WHEEL_PLUGIN_DIRS = (
    ("nixl_cu13", ".nixl-cu13.mesonpy.libs"),
    ("nixl_cu12", ".nixl-cu12.mesonpy.libs"),
    ("nixl", ".nixl.mesonpy.libs"),
)
_COMMON_UCX_PLUGIN_DIRS = (
    pathlib.Path("/opt/hpcx/ucx/lib"),
    pathlib.Path("/opt/hpcx/ucx/lib/ucx"),
    pathlib.Path("/usr/lib/ucx"),
    pathlib.Path("/usr/local/lib/ucx"),
)
_COMMON_NIXL_PLUGIN_DIRS = (
    pathlib.Path("/usr/local/nixl/lib/x86_64-linux-gnu/plugins"),
    pathlib.Path("/usr/local/nixl/lib/aarch64-linux-gnu/plugins"),
)


def _maybe_set_nixl_plugin_dir_from_wheel() -> None:
    if "NIXL_PLUGIN_DIR" in os.environ:
        return
    for package, lib_dir in _NIXL_WHEEL_PLUGIN_DIRS:
        spec = importlib.util.find_spec(package)
        if not spec or not spec.origin:
            continue
        candidate = pathlib.Path(spec.origin).parent.parent / lib_dir / "plugins"
        if candidate.is_dir():
            os.environ["NIXL_PLUGIN_DIR"] = str(candidate)
            return


def _path_entries(env_var: str) -> list[pathlib.Path]:
    return [pathlib.Path(p) for p in os.environ.get(env_var, "").split(":") if p]


def _has_file_in_dirs(filename: str, dirs: list[pathlib.Path]) -> bool:
    return any((d / filename).exists() for d in dirs)


def _nixl_vram_functional() -> bool:
    """Check if NIXL can register VRAM.

    The NIXL backend needs the UCX plugin and CUDA-aware UCX transports at
    runtime. The CUDA DL / DLFW containers provide NIXL, but source builds
    must point LD_LIBRARY_PATH and NIXL_PLUGIN_DIR at matching UCX/NIXL
    prefixes.
    """
    if not symm_mem.is_nixl_available() or not torch.cuda.is_available():
        return False

    ucx_dirs = _path_entries("LD_LIBRARY_PATH") + list(_COMMON_UCX_PLUGIN_DIRS)
    if not _has_file_in_dirs("libuct_cuda.so", ucx_dirs):
        return False

    plugin_dirs = _path_entries("NIXL_PLUGIN_DIR") + [
        d / "plugins" for d in _path_entries("LD_LIBRARY_PATH")
    ] + list(_COMMON_NIXL_PLUGIN_DIRS)
    return _has_file_in_dirs("libplugin_UCX.so", plugin_dirs)


_maybe_set_nixl_plugin_dir_from_wheel()
NIXL_COMPILED = symm_mem.is_nixl_available()
NIXL_FUNCTIONAL = _nixl_vram_functional()

requires_nixl = unittest.skipUnless(NIXL_COMPILED, "NIXL backend not compiled")
requires_nixl_functional = unittest.skipUnless(
    NIXL_FUNCTIONAL, "NIXL not functional (UCX may lack CUDA support)"
)

@requires_cuda_p2p_access()
class NixlSymmetricMemoryTest(MultiProcContinuousTest):
    """Multi-process tests for the NIXL symmetric memory backend.

    TODO: Revisit once NIXL's repeated metadata-load semantics are documented
    across releases. Continuous workers reuse one rendezvoused tensor so these
    tests do not depend on repeated remote metadata reloads after address reuse.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init(self):
        symm_mem.set_backend("NIXL")
        torch.cuda.set_device(self.rank)
        # MultiProcContinuousTest keeps worker processes alive across methods.
        # Reuse one allocation/rendezvous per worker to keep tests deterministic.
        if not hasattr(self, "_shared_t"):
            self._shared_t = symm_mem.empty(
                1024, dtype=torch.float32, device=self.device
            )
            self._shared_hdl = symm_mem.rendezvous(
                self._shared_t, group=dist.group.WORLD
            )

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_alloc_and_rendezvous(self):
        """Allocation + rendezvous succeeds."""
        self._init()
        self.assertIsNotNone(self._shared_hdl)

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_handle_properties(self):
        """Handle reports correct rank, world_size, and positive buffer_size."""
        self._init()
        hdl = self._shared_hdl
        self.assertEqual(hdl.rank, self.rank)
        self.assertEqual(hdl.world_size, self.world_size)
        self.assertGreater(hdl.buffer_size, 0)

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_get_local_buffer(self):
        """Local buffer aliases the original tensor."""
        self._init()
        t = self._shared_t
        numel = t.numel()
        buf = self._shared_hdl.get_buffer(self.rank, (numel,), t.dtype)
        t.fill_(42.0)
        self.assertEqual(buf.sum().item(), 42.0 * numel)

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_repeated_rendezvous(self):
        """Second rendezvous on same tensor returns cached handle."""
        self._init()
        hdl2 = symm_mem.rendezvous(self._shared_t, group=dist.group.WORLD)
        self.assertEqual(self._shared_hdl.rank, hdl2.rank)
        self.assertEqual(self._shared_hdl.buffer_size, hdl2.buffer_size)

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_signal_pad_accessible(self):
        """Local signal pad is non-null and has elements."""
        self._init()
        sig = self._shared_hdl.get_signal_pad(self.rank)
        self.assertIsNotNone(sig)
        self.assertGreater(sig.numel(), 0)


@requires_cuda_p2p_access()
class NixlPairSymmetricMemoryTest(NixlSymmetricMemoryTest):
    world_size = 2

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_put_pair(self):
        self._init()
        self._shared_t.fill_(float(self.rank + 1))
        torch.cuda.synchronize(self.device)
        dist.barrier()

        if self.rank == 0:
            torch.ops.symm_mem.nixl_put(self._shared_t, 1)
        dist.barrier()
        self.assertEqual(self._shared_t, torch.full_like(self._shared_t, 1.0))

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_get_pair(self):
        self._init()
        self._shared_t.fill_(float(self.rank + 1))
        torch.cuda.synchronize(self.device)
        dist.barrier()

        if self.rank == 0:
            torch.ops.symm_mem.nixl_get(self._shared_t, 1)
            self.assertEqual(self._shared_t, torch.full_like(self._shared_t, 2.0))
        else:
            self.assertEqual(self._shared_t, torch.full_like(self._shared_t, 2.0))
        dist.barrier()

    @requires_nixl_functional
    @skip_if_lt_x_gpu(2)
    def test_nixl_put_with_signal_pair(self):
        self._init()
        self._shared_hdl.get_signal_pad(self.rank, (1,), torch.uint64).zero_()
        self._shared_t.fill_(float(self.rank + 1))
        torch.cuda.synchronize(self.device)
        dist.barrier()

        if self.rank == 0:
            torch.ops.symm_mem.nixl_put_with_signal(self._shared_t, 123, 1)
        else:
            torch.ops.symm_mem.nixl_wait_for_signal(self._shared_t, 123)
            self.assertEqual(self._shared_t, torch.full_like(self._shared_t, 1.0))
        dist.barrier()


class NixlSingleProcTest(TestCase):
    def test_nixl_backend_name_in_valid_list(self):
        if not NIXL_COMPILED:
            with self.assertRaises(RuntimeError):
                symm_mem.set_backend("NIXL")
        else:
            self.assertTrue(symm_mem.is_nixl_available())

    @requires_nixl
    def test_nixl_single_process_alloc(self):
        symm_mem.set_backend("NIXL")
        t = symm_mem.empty(512, dtype=torch.float32, device="cuda:0")
        self.assertEqual(t.shape, torch.Size([512]))
        self.assertEqual(t.device, torch.device("cuda", 0))


if __name__ == "__main__":
    run_tests()
