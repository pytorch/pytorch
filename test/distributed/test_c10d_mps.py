# Owner(s): ["module: mps"]
"""Smoke tests for the MPS distributed backend (ProcessGroupMPS / JACCL).

These verify the Python bindings on a single machine. They do not exercise
end-to-end multi-rank collectives over Thunderbolt RDMA — that requires two
Macs cabled together; see ``docs/source/distributed.mps.md``.

The file `sys.exit(0)`s cleanly on systems where the MPS distributed backend
isn't built (Linux/Windows CI, USE_C10D_MPS=0), so it is safe to include in
the default distributed test suite.
"""

import sys

import torch.distributed as c10d


if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if not c10d.is_mps_backend_available():
    print(
        "c10d MPS backend not built (USE_C10D_MPS=0), skipping tests",
        file=sys.stderr,
    )
    sys.exit(0)


from torch._C._distributed_c10d import ProcessGroup
from torch.distributed.distributed_c10d import ProcessGroupMPS
from torch.testing._internal.common_utils import run_tests, TestCase


class TestProcessGroupMPSBindings(TestCase):
    def test_backend_enum_has_mps(self):
        self.assertEqual(c10d.Backend.MPS, "mps")
        self.assertIn(c10d.Backend.MPS, c10d.Backend.backend_list)

    def test_backend_type_enum_has_mps(self):
        self.assertTrue(hasattr(ProcessGroup.BackendType, "MPS"))

    def test_processgroupmps_options_attr(self):
        self.assertTrue(hasattr(ProcessGroupMPS, "Options"))

    def test_options_default_construct(self):
        opts = ProcessGroupMPS.Options()
        self.assertEqual(opts.backend, "mps")

    def test_world_size_1_construction_throws(self):
        """Constructing the backend with no peers must fail eagerly.

        Two failure paths converge here, both surfaced as RuntimeError:
          * machine without an RDMA-capable Thunderbolt device — fails the
            ``ibv_alloc_pd`` probe on every ``rdma_en*`` and the constructor
            throws "requires Apple Thunderbolt RDMA on every rank".
          * machine with a working RDMA device but world_size=1 — passes the
            probe, then JACCL ``init`` rejects the config with
            "configuration should define a valid mesh or a valid ring".

        We only assert the type, not the message, since which path runs
        depends on the host.
        """
        store = c10d.HashStore()
        opts = ProcessGroupMPS.Options()
        with self.assertRaises(RuntimeError):
            ProcessGroupMPS(store, 0, 1, opts)


if __name__ == "__main__":
    run_tests()
