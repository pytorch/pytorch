# Owner(s): ["oncall: distributed"]

import logging
import os
import subprocess
import sys
import textwrap
import unittest

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor.debug import _clear_sharding_prop_cache
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_SANDCASTLE,
    run_concurrently,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestDTensorLogging(TestCase):
    """Test DTensor logging."""

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        _clear_sharding_prop_cache()
        self.world_size = 2
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )
        acc = torch.accelerator.current_accelerator(check_available=True)
        self.device_type = acc.type if acc else "cpu"

    def test_sharding_prop_cache_logging(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        log_records = []
        handler = logging.Handler()
        handler.emit = lambda record: log_records.append(record)

        def log_string():
            return "\n".join([r.getMessage() for r in log_records])

        # Test C++ shard prop cache
        dispatch_logger = logging.getLogger("torch.distributed.tensor._dispatch")
        dispatch_logger.setLevel(logging.DEBUG)
        dispatch_logger.addHandler(handler)

        # Test simple miss/hit
        x_dt = DTensor.from_local(torch.randn(2, 4), mesh, [Shard(0)], run_check=False)
        x_dt + x_dt  # miss
        x_dt + x_dt  # hit

        # Test miss with different placements
        x_dt1 = x_dt.redistribute(placements=[Replicate()])
        x_dt1 + x_dt1

        # Test miss with different shapes
        x_dt2 = DTensor.from_local(torch.randn(4, 4), mesh, [Shard(0)], run_check=False)
        x_dt2 + x_dt2

        self.assertExpectedInline(
            log_string(),
            f"""\
sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0)))) on DeviceMesh((2,), '{self.device_type}', stride=(1,))) -> Spec(f32[4, 4](S(0)))
sharding_prop HIT (C++ fast path): aten::add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0))), 4822678189205111) -> Spec(f32[4, 4](S(0)))
sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[4, 4](R)), Spec(f32[4, 4](R))) on DeviceMesh((2,), '{self.device_type}', stride=(1,))) -> Spec(f32[4, 4](R))
sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[8, 4](S(0))), Spec(f32[8, 4](S(0)))) on DeviceMesh((2,), '{self.device_type}', stride=(1,))) -> Spec(f32[8, 4](S(0)))""",
        )

        # Test Python LRU cache, directly with ShardingPropagator
        log_records.clear()
        prop_logger = logging.getLogger("torch.distributed.tensor._sharding_prop")
        prop_logger.setLevel(logging.DEBUG)
        prop_logger.addHandler(handler)
        propagator = DTensor._op_dispatcher.sharding_propagator

        spec = DTensorSpec(
            mesh=mesh,
            placements=(Shard(0),),
            tensor_meta=TensorMeta(
                shape=torch.Size([4, 4]), stride=(4, 1), dtype=torch.float32
            ),
        )
        op_schema = OpSchema(
            op=torch.ops.aten.add.Tensor, args_schema=(spec, spec), kwargs_schema={}
        )
        propagator.propagate_op_sharding(op_schema)  # Python cache miss
        propagator.propagate_op_sharding(op_schema)  # Python cache hit
        self.assertExpectedInline(
            log_string(),
            f"""\
sharding_prop python cache MISS: aten.add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0)))) on DeviceMesh((2,), '{self.device_type}', stride=(1,))) -> Spec(f32[4, 4](S(0)))
sharding_prop python cache HIT: aten.add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0)))) on DeviceMesh((2,), '{self.device_type}', stride=(1,))) -> Spec(f32[4, 4](S(0)))""",
        )

    def test_logging_level_change_resets_cpp_cache(self):
        """setLevel on the dispatch logger resets the C++ cached logging flag."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        dispatch_logger = logging.getLogger("torch.distributed.tensor._dispatch")

        log_records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = lambda record: log_records.append(record)
        dispatch_logger.addHandler(handler)

        x_dt = DTensor.from_local(torch.randn(2, 4), mesh, [Shard(0)], run_check=False)

        # With logging off, the C++ hit path should produce no records.
        dispatch_logger.setLevel(logging.WARNING)
        x_dt + x_dt  # miss (not logged)
        x_dt + x_dt  # hit (not logged)
        self.assertEqual(len(log_records), 0)

        # After enabling DEBUG, the C++ side should pick it up
        # automatically (via the setLevel hook) without any manual reset.
        _clear_sharding_prop_cache()
        log_records.clear()
        dispatch_logger.setLevel(logging.DEBUG)
        x_dt + x_dt  # miss (logged)
        x_dt + x_dt  # hit (logged)
        self.assertEqual(len(log_records), 2)
        self.assertIn("MISS", log_records[0].getMessage())
        self.assertIn("HIT", log_records[1].getMessage())

    def test_sharding_prop_cache_hit_in_worker_thread(self):
        """A worker thread hitting the C++ sharding prop cache must exit cleanly."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        dispatch_logger = logging.getLogger("torch.distributed.tensor._dispatch")

        log_records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = lambda record: log_records.append(record)
        dispatch_logger.addHandler(handler)
        self.addCleanup(dispatch_logger.removeHandler, handler)
        self.addCleanup(dispatch_logger.setLevel, dispatch_logger.level)

        x_dt = DTensor.from_local(torch.randn(2, 4), mesh, [Shard(0)], run_check=False)

        def worker():
            x_dt + x_dt  # miss (populates thread-local C++ cache)
            x_dt + x_dt  # hit (invokes log_sharding_prop_cache_hit on worker thread)
            return torch._C._get_DTensor_sharding_propagator_cache_stats()

        for level, expected_records in ((logging.WARNING, 0), (logging.DEBUG, 2)):
            log_records.clear()
            dispatch_logger.setLevel(level)
            hits, _ = run_concurrently(worker, num_threads=1)[0]
            self.assertGreaterEqual(hits, 1)
            self.assertEqual(len(log_records), expected_records)
        self.assertIn("MISS", log_records[0].getMessage())
        self.assertIn("HIT", log_records[1].getMessage())

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    def test_sharding_prop_cache_hit_worker_thread_exit_subprocess(self):
        """A worker thread hitting an unreferenced logger in C++ must exit cleanly."""
        # Run the worker-thread C++ cache-hit scenario in a fresh subprocess and
        # drop Python references to the logger before the worker thread exits.
        # This ensures that if C++ stored the logger in a thread_local py::object
        # (instead of process-wide immortal storage), its destructor at thread
        # exit after PyThreadState_DeleteCurrent() releases the GIL would
        # deallocate the logger and crash the subprocess, without polluting the
        # parent process's cached logger.
        script = textwrap.dedent(
            f"""
            import logging
            import torch
            import torch.distributed as dist
            import torch.distributed.tensor._dispatch as _dispatch
            from torch.distributed.tensor import DeviceMesh, DTensor, Shard
            from torch.testing._internal.common_utils import run_concurrently
            from torch.testing._internal.distributed.fake_pg import FakeStore

            dist.init_process_group(
                backend="fake", rank=0, world_size=2, store=FakeStore()
            )
            try:
                messages = []
                handler = logging.Handler()
                handler.emit = lambda record: messages.append(record.getMessage())

                logger = logging.Logger("torch.distributed.tensor._dispatch")
                logger.setLevel(logging.DEBUG)
                logger.addHandler(handler)
                _dispatch.logger = logger
                logging.Logger.manager.loggerDict[
                    "torch.distributed.tensor._dispatch"
                ] = logger
                del logger

                mesh = DeviceMesh("{self.device_type}", [0, 1])
                x_dt = DTensor.from_local(
                    torch.randn(2, 4), mesh, [Shard(0)], run_check=False
                )

                def worker():
                    x_dt + x_dt  # miss (populates thread-local C++ cache)
                    x_dt + x_dt  # hit (invokes log_sharding_prop_cache_hit)
                    hits, _ = (
                        torch._C._get_DTensor_sharding_propagator_cache_stats()
                    )
                    assert hits >= 1
                    assert any(
                        "HIT (C++ fast path)" in m for m in messages
                    ), messages
                    _dispatch.logger = None
                    logging.Logger.manager.loggerDict.pop(
                        "torch.distributed.tensor._dispatch", None
                    )

                run_concurrently(worker, num_threads=1)
            finally:
                dist.destroy_process_group()
            """
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"Worker thread exit after C++ cache hit failed:\n{proc.stderr}",
        )


if __name__ == "__main__":
    run_tests()
