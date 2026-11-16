# Owner(s): ["oncall: distributed"]

import argparse
import gc
import os
import pickle
import tempfile

from torch.distributed.flight_recorder.components.loader import read_dir
from torch.testing._internal.common_utils import run_tests, TestCase


class FlightRecorderLoaderTest(TestCase):
    def setUp(self):
        super().setUp()
        restore_gc = gc.enable if gc.isenabled() else gc.disable
        self.addCleanup(restore_gc)

    def _write_dump(self, trace_dir):
        path = os.path.join(trace_dir, "dump0")
        with open(path, "wb") as output:
            pickle.dump(
                {
                    "entries": [],
                    "version": "2.8",
                    "pg_config": {},
                },
                output,
            )

    def test_read_dir_restores_enabled_gc_after_success(self):
        with tempfile.TemporaryDirectory() as trace_dir:
            self._write_dump(trace_dir)
            gc.enable()

            read_dir(argparse.Namespace(trace_dir=trace_dir, prefix="dump"))

            self.assertTrue(gc.isenabled())

    def test_read_dir_restores_enabled_gc_after_failure(self):
        gc.enable()

        with self.assertRaisesRegex(AssertionError, "does not exist"):
            read_dir(
                argparse.Namespace(
                    trace_dir="missing-flight-recorder-dir",
                    prefix=None,
                )
            )

        self.assertTrue(gc.isenabled())

    def test_read_dir_preserves_disabled_gc(self):
        with tempfile.TemporaryDirectory() as trace_dir:
            self._write_dump(trace_dir)
            gc.disable()

            read_dir(argparse.Namespace(trace_dir=trace_dir, prefix="dump"))

            self.assertFalse(gc.isenabled())


if __name__ == "__main__":
    run_tests()
