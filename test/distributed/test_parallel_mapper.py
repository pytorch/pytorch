# Owner(s): ["oncall: distributed"]

"""Tests for torch.distributed.parallel and _collective_interceptor."""

import os
import threading

import torch
import torch.distributed as dist
import torch.distributed.parallel._parallel_mapper as pm
from torch.distributed.parallel import (
    parallel_map,
    parallel_multi_apply,
    parallel_starmap,
    sync_wrap,
)
from torch.distributed._collective_interceptor import (
    call_collective_interceptor,
    clear_worker_id,
    get_collective_interceptor,
    get_worker_id,
    set_collective_interceptor,
    set_worker_id,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _make_mock_collective():
    """Return (mock_op, call_order) where mock_op honours the interceptor."""
    call_order: list = []
    lock = threading.Lock()

    def _impl(tensor):
        with lock:
            call_order.append(tensor)
        return tensor

    def mock_op(tensor):
        interceptor = get_collective_interceptor()
        if interceptor is not None:
            return interceptor("mock_op", _impl, tensor)
        return _impl(tensor)

    return mock_op, call_order


class TestCollectiveInterceptor(TestCase):
    """Thread-local interceptor primitives."""

    def tearDown(self):
        set_collective_interceptor(None)
        clear_worker_id()
        super().tearDown()

    def test_set_get_clear(self):
        self.assertIsNone(get_collective_interceptor())
        fn = lambda *a, **kw: None  # noqa: E731
        set_collective_interceptor(fn)
        self.assertIs(get_collective_interceptor(), fn)
        set_collective_interceptor(None)
        self.assertIsNone(get_collective_interceptor())

    def test_thread_isolation(self):
        set_collective_interceptor(lambda *a, **kw: None)
        set_worker_id(7)
        seen = {}

        def probe():
            seen["interceptor"] = get_collective_interceptor()
            seen["worker_id"] = get_worker_id()

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        self.assertIsNone(seen["interceptor"])
        self.assertIsNone(seen["worker_id"])

    def test_reentrant_guard(self):
        inner_value = []

        def interceptor(op_name, fn, *args, **kwargs):
            inner_value.append(get_collective_interceptor())
            return fn(*args, **kwargs)

        set_collective_interceptor(interceptor)
        call_collective_interceptor(interceptor, "op", lambda: 42)
        self.assertIsNone(inner_value[0])
        self.assertIsNotNone(get_collective_interceptor())


class TestParallelMap(TestCase):
    """Core parallel_map / starmap / multi_apply behaviour."""

    def test_map_basic(self):
        self.assertEqual(
            parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5], max_workers=3),
            [2, 4, 6, 8, 10],
        )

    def test_starmap_basic(self):
        self.assertEqual(
            parallel_starmap(lambda a, b: a + b, [(1, 10), (2, 20)], max_workers=2),
            [11, 22],
        )

    def test_multi_apply_basic(self):
        sums, prods = parallel_multi_apply(
            lambda a, b: (a + b, a * b), [1, 2, 3], [10, 20, 30], max_workers=3
        )
        self.assertEqual(sums, [11, 22, 33])
        self.assertEqual(prods, [10, 40, 90])

    def test_multi_apply_with_kwargs(self):
        def fn(a, scale=1):
            return a * scale, a + scale

        scaled, added = parallel_multi_apply(fn, [1, 2, 3], max_workers=3, scale=10)
        self.assertEqual(scaled, [10, 20, 30])
        self.assertEqual(added, [11, 12, 13])

    def test_multi_apply_fewer_workers_than_items(self):
        """multi_apply batches correctly when max_workers < len(items)."""
        mock_op, order = _make_mock_collective()

        def fn(a, b):
            mock_op(a)
            return a + b, a * b

        sums, prods = parallel_multi_apply(
            fn, [1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60], max_workers=2
        )
        self.assertEqual(sums, [11, 22, 33, 44, 55, 66])
        self.assertEqual(prods, [10, 40, 90, 160, 250, 360])
        self.assertEqual(order, [1, 2, 3, 4, 5, 6])

    def test_empty_and_single(self):
        self.assertEqual(parallel_map(lambda x: x, [], max_workers=4), [])
        self.assertEqual(parallel_map(lambda x: x * 3, [7], max_workers=4), [21])

    def test_batching(self):
        results = parallel_map(lambda x: x + 1, list(range(10)), max_workers=3)
        self.assertEqual(results, list(range(1, 11)))

    def test_return_order_matches_input(self):
        def slow_first(x):
            if x == 0:
                s = 0
                for i in range(200_000):
                    s += i
            return x

        self.assertEqual(
            parallel_map(slow_first, [0, 1, 2, 3], max_workers=4), [0, 1, 2, 3]
        )


class TestOrdering(TestCase):
    """Collective ops are serialized in strict worker_id order."""

    def test_single_round(self):
        mock_op, order = _make_mock_collective()

        results = parallel_map(
            lambda item: (mock_op(item * 10), item * 10)[1],
            [0, 1, 2, 3],
            max_workers=4,
        )
        self.assertEqual(results, [0, 10, 20, 30])
        self.assertEqual(order, [0, 10, 20, 30])

    def test_multiple_rounds(self):
        mock_op, order = _make_mock_collective()

        def fn(item):
            mock_op(f"r1_w{item}")
            mock_op(f"r2_w{item}")
            return item

        parallel_map(fn, [0, 1, 2], max_workers=3)
        self.assertEqual(
            order, ["r1_w0", "r1_w1", "r1_w2", "r2_w0", "r2_w1", "r2_w2"]
        )

    def test_computation_between_syncs(self):
        mock_op, order = _make_mock_collective()

        def fn(item):
            s = 0
            for i in range(item * 50_000):
                s += i
            mock_op(f"w{item}")
            return item

        parallel_map(fn, [0, 1, 2, 3], max_workers=4)
        self.assertEqual(order, ["w0", "w1", "w2", "w3"])

    def test_fewer_workers_than_items_with_sync(self):
        """Ordering stays correct across batches when max_workers < len(items)."""
        mock_op, order = _make_mock_collective()

        def fn(item):
            mock_op(item)
            return item

        # 6 items, 2 workers => 3 batches
        results = parallel_map(fn, [0, 1, 2, 3, 4, 5], max_workers=2)
        self.assertEqual(results, [0, 1, 2, 3, 4, 5])
        self.assertEqual(order, [0, 1, 2, 3, 4, 5])


class TestSyncWrap(TestCase):

    def test_ordering_and_passthrough(self):
        order: list = []
        lock = threading.Lock()

        def custom_sync(data):
            with lock:
                order.append(data)
            return data

        wrapped = sync_wrap(custom_sync)

        # Inside parallel_map: serialized in worker_id order
        def fn(item):
            wrapped(f"w{item}")
            return item

        parallel_map(fn, [0, 1, 2], max_workers=3)
        self.assertEqual(order, ["w0", "w1", "w2"])

        # Outside parallel_map: passthrough
        order.clear()
        self.assertEqual(wrapped(42), 42)
        self.assertEqual(order, [42])

    def test_preserves_name(self):
        def my_function(x):
            return x

        wrapped = sync_wrap(my_function)
        self.assertEqual(wrapped.__name__, "my_function")
        self.assertIs(wrapped.__wrapped__, my_function)


class TestErrorHandling(TestCase):

    def test_exception_propagation(self):
        def fn(item):
            if item == 2:
                raise ValueError("boom")
            return item

        with self.assertRaisesRegex(ValueError, "boom"):
            parallel_map(fn, [0, 1, 2, 3], max_workers=4)

    def test_exception_before_sync_no_deadlock(self):
        mock_op, _ = _make_mock_collective()

        def fn(item):
            if item == 1:
                raise RuntimeError("early fail")
            mock_op(item)
            return item

        with self.assertRaisesRegex(RuntimeError, "early fail"):
            parallel_map(fn, [0, 1, 2], max_workers=3)

    def test_pool_recovers_after_error(self):
        with self.assertRaises(ZeroDivisionError):
            parallel_map(lambda x: 1 // 0 if x == 1 else x, [0, 1, 2], max_workers=3)

        self.assertEqual(
            parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=3), [2, 4, 6]
        )


class TestWorkerId(TestCase):

    def test_worker_id_in_fn_and_cleared_after(self):
        def fn(item):
            return (item, get_worker_id())

        results = parallel_map(fn, [10, 20, 30], max_workers=3)
        self.assertEqual([r[0] for r in results], [10, 20, 30])
        self.assertEqual(set(r[1] for r in results), {0, 1, 2})
        self.assertIsNone(get_worker_id())


class TestTimeout(TestCase):

    def test_inconsistent_sync_triggers_timeout(self):
        mock_op, _ = _make_mock_collective()
        old_timeout = pm._SYNC_TIMEOUT
        pm._SYNC_TIMEOUT = 1.0
        try:
            def fn(item):
                if item == 1:
                    mock_op(item)
                return item

            with self.assertRaises(TimeoutError):
                parallel_map(fn, [0, 1], max_workers=2)
        finally:
            pm._SYNC_TIMEOUT = old_timeout


class TestStress(TestCase):

    def test_repeated_calls_with_sync(self):
        mock_op, order = _make_mock_collective()
        for _ in range(50):
            order.clear()
            results = parallel_map(
                lambda item: (mock_op(item), item)[1], [0, 1, 2, 3], max_workers=4
            )
            self.assertEqual(results, [0, 1, 2, 3])
            self.assertEqual(order, [0, 1, 2, 3])


class TestParallelMapNCCL(MultiProcessTestCase):
    """Multi-GPU tests with real NCCL collective operations."""

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_in_parallel_map(self):
        """parallel_map correctly serializes real dist.all_reduce calls."""
        self._init_pg()
        device = f"cuda:{self.rank}"

        def fn(item):
            t = torch.tensor([item + self.rank], dtype=torch.float32, device=device)
            dist.all_reduce(t)
            return t.item()

        results = parallel_map(fn, [10.0, 20.0, 30.0], max_workers=3)
        # world_size=2, so all_reduce sums rank 0 and rank 1 values
        # For item=10: rank0 has 10+0=10, rank1 has 10+1=11, sum=21
        # For item=20: 20+21=41, For item=30: 30+31=61
        expected = [
            (item + 0) + (item + 1) for item in [10.0, 20.0, 30.0]
        ]
        self.assertEqual(results, expected)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_in_parallel_map(self):
        """parallel_map correctly serializes real dist.broadcast calls."""
        self._init_pg()
        device = f"cuda:{self.rank}"

        def fn(item):
            t = torch.tensor([item * (self.rank + 1)], dtype=torch.float32, device=device)
            dist.broadcast(t, src=0)
            return t.item()

        results = parallel_map(fn, [1.0, 2.0, 3.0], max_workers=3)
        # broadcast from rank 0: rank 0 has item*1, after broadcast all ranks have rank 0's value
        expected = [1.0, 2.0, 3.0]
        self.assertEqual(results, expected)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_collectives_in_parallel_map(self):
        """Multiple real collective ops per worker stay correctly ordered."""
        self._init_pg()
        device = f"cuda:{self.rank}"

        def fn(item):
            t = torch.full((4,), float(item), device=device)
            dist.all_reduce(t)  # sum across ranks: item * world_size
            dist.broadcast(t, src=0)  # both ranks get rank 0's result
            return t[0].item()

        results = parallel_map(fn, [1.0, 2.0, 3.0, 4.0], max_workers=4)
        expected = [item * self.world_size for item in [1.0, 2.0, 3.0, 4.0]]
        self.assertEqual(results, expected)


if __name__ == "__main__":
    run_tests()
