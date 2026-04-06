# Owner(s): ["oncall: distributed"]

import logging

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor.debug import _clear_sharding_prop_cache
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


@requires_cuda
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
        self.device_type = "cuda"

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
            """\
sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0)))) on DeviceMesh((2,), 'cuda', stride=(1,))) -> Spec(f32[4, 4](S(0)))
sharding_prop HIT (C++ fast path): aten::add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0))), 4822678189205111) -> Spec(f32[4, 4](S(0)))
sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[4, 4](R)), Spec(f32[4, 4](R))) on DeviceMesh((2,), 'cuda', stride=(1,))) -> Spec(f32[4, 4](R))
sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[8, 4](S(0))), Spec(f32[8, 4](S(0)))) on DeviceMesh((2,), 'cuda', stride=(1,))) -> Spec(f32[8, 4](S(0)))""",  # noqa: B950
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
            """\
sharding_prop python cache MISS: aten.add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0)))) on DeviceMesh((2,), 'cuda', stride=(1,))) -> Spec(f32[4, 4](S(0)))
sharding_prop python cache HIT: aten.add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0)))) on DeviceMesh((2,), 'cuda', stride=(1,))) -> Spec(f32[4, 4](S(0)))""",  # noqa: B950
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


if __name__ == "__main__":
    run_tests()
