# Owner(s): ["oncall: distributed"]

"""
Regression tests for: new_group() must not route a 'fake' subgroup through the
TorchComms split_group delegation.

split_group can only split the parent's existing communicator, so it cannot
produce a child whose backend differs from the parent's. A "fake" subgroup of a
real parent is exactly that case -- which is how DeviceMesh creates disabled /
unflattened mesh dimensions (with use_local_synchronization=True for hashed PG
names). Before the fix that delegation raised NotImplementedError. The fix routes
a fake subgroup of a non-fake parent through the normal path, which builds a
FakeProcessGroup directly and still produces a hashed name.

These tests use a real (gloo) parent on a single CPU process: no GPU needed.
"""

import os

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import run_tests, TestCase


class FakeBackendNewGroupTest(TestCase):
    def setUp(self):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        # Real (non-fake) parent so a fake child differs from the parent backend.
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _check_fake_subgroup(self):
        g = dist.new_group(ranks=[0], backend="fake", use_local_synchronization=True)
        self.assertIsNotNone(g)
        self.assertEqual(dist.get_backend(g), "fake")
        # Hashed-name path preserved: sha1 hexdigest (40 chars), not a counter int.
        name = g.group_name
        self.assertFalse(name.isdigit(), f"expected hashed name, got int {name!r}")
        self.assertEqual(len(name), 40, f"expected 40-char sha1 hash, got {name!r}")

    def test_fake_subgroup_of_real_parent_builds_fake_pg(self):
        """With TorchComms 'enabled', new_group(backend='fake',
        use_local_synchronization=True) on a real parent builds a
        FakeProcessGroup (no NotImplementedError) and keeps the hashed name."""
        # Simulate TorchComms enabled for just the new_group call, so the test
        # needs neither the torchcomms package nor a GPU.
        orig = c10d._use_torchcomms_enabled
        c10d._use_torchcomms_enabled = lambda: True
        try:
            self._check_fake_subgroup()
        finally:
            c10d._use_torchcomms_enabled = orig

    def test_fake_subgroup_real_torchcomms_flag(self):
        """Same as above but driving the real dist_config.use_torchcomms flag."""
        import torch.distributed.config as dist_config

        prev = dist_config.use_torchcomms
        dist_config.use_torchcomms = True
        try:
            # With the flag set, _use_torchcomms_enabled() reduces to whether the
            # torchcomms package is available; skip when it isn't.
            if not c10d._use_torchcomms_enabled():
                self.skipTest("torchcomms not available")
            self._check_fake_subgroup()
        finally:
            dist_config.use_torchcomms = prev


if __name__ == "__main__":
    run_tests()
