# Owner(s): ["oncall: distributed"]

import os
from datetime import timedelta

import torch
from torch._C._distributed_c10d import FakeProcessGroup
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


class RejectCoalescing(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.coalesce.default, torch.ops.aten._coalesce.default):
            raise AssertionError("FakeWork must preserve uncoalesced COO storage")
        return func(*args, **(kwargs or {}))


class TestFakeWorkStreams(TestCase):
    def _make_results(self, device, result_kind):
        tensors = [torch.ones(2), torch.empty(2, device="meta")]
        parts = []
        initial = []
        if result_kind in ("dense", "mixed"):
            dense_initial = torch.arange(1, 5, dtype=torch.float32)
            dense = dense_initial.to(device)
            tensors.append(dense)
            parts.append(dense)
            initial.append(dense_initial)
        if result_kind in ("coo", "mixed"):
            indices = torch.tensor([[0, 0, 1]])
            values = torch.tensor([1.0, 2.0, 3.0])
            sparse = torch.sparse_coo_tensor(
                indices.to(device),
                values.to(device),
                size=(4,),
                is_coalesced=False,
            )
            tensors.append(sparse)
            parts.extend((sparse._indices(), sparse._values()))
            initial.extend((indices, values))
        if result_kind == "csr":
            dense_initial = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
            csr = dense_initial.to(device).to_sparse_csr()
            tensors.append(csr)
            parts.append(csr.values())
            initial.append(dense_initial.to_sparse_csr().values())
        return tensors, parts, initial

    def _make_pg(self):
        if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
            self.skipTest("CUDA_LAUNCH_BLOCKING removes the cross-stream window")
        options = FakeProcessGroup.Options()
        options.simulate_uniform_ranks = True
        return FakeProcessGroup._create_internal(0, 2, options)

    def _wait(self, work, future, wait_method):
        if wait_method == "future_wait":
            self.assertTrue(future.done())
            return future.wait()
        if wait_method == "work_wait":
            self.assertTrue(work.wait(timeout=timedelta(seconds=10)))
        else:
            self.assertIsNone(work.block_current_stream())
        return work.result()

    def _assert_in_stale_window(self, attempt):
        # attempt(cycles) runs one trial and returns (window, check): window is a
        # tuple of flags that are all true iff a stale-data window opened, and
        # check() asserts the waited-on snapshots.
        windows = []
        for cycles in (50_000_000, 200_000_000, 500_000_000):
            window, check = attempt(cycles)
            windows.append((cycles, *window))
            if all(window):
                check()
                return
        self.fail(
            "INCONCLUSIVE: no stale-data window after three bounded delays: "
            + repr(windows)
        )

    @parametrize("wait_method", ["work_wait", "future_wait", "block_current_stream"])
    @parametrize("result_kind", ["dense", "coo", "mixed", "csr"])
    def test_result_work_orders_current_stream(self, device, wait_method, result_kind):
        pg = self._make_pg()
        producer = torch.cuda.Stream(device=device)
        consumer = torch.cuda.Stream(device=device)

        def attempt(cycles):
            ready = torch.cuda.Event()
            delay_done = torch.cuda.Event()
            producer_done = torch.cuda.Event()
            negative_done = torch.cuda.Event()
            consumer_done = torch.cuda.Event()
            with torch.cuda.stream(producer):
                tensors, parts, initial = self._make_results(device, result_kind)
                identities = [tensor._cdata for tensor in tensors]
                pointers = [part.data_ptr() for part in parts]
                sparse_pointers = [
                    (tensor, tensor._indices().data_ptr(), tensor._values().data_ptr())
                    for tensor in tensors
                    if tensor.is_sparse
                ]
                ready.record()
                torch.cuda._sleep(cycles)
                delay_done.record()
                for part in parts:
                    part.mul_(2)
                with RejectCoalescing():
                    work = pg.broadcast(tensors)
                reported_completed = work.is_completed()
                producer_done.record()

            with torch.cuda.stream(consumer):
                consumer.wait_event(ready)
                negative = [part.clone() for part in parts]
                negative_done.record()
                returns = []
                snapshots = []
                with RejectCoalescing():
                    # The Future is first requested away from the issuing stream.
                    future = work.get_future() if wait_method == "future_wait" else None
                    for _ in range(2):
                        returns.append(self._wait(work, future, wait_method))
                        snapshots.append([part.clone() for part in parts])
                    if future is None:
                        future = work.get_future()
                consumer.wait_event(producer_done)
                positive = [part.clone() for part in parts]
                consumer_done.record()

            # Every snapshot is queued before any host wait can drain the producer.
            negative_done.synchronize()
            delay_pending = not delay_done.query()
            consumer_done.synchronize()
            expected = [part * 2 for part in initial]
            self.assertEqual(positive, expected, atol=0, rtol=0)
            self.assertEqual(parts, expected, atol=0, rtol=0)
            self.assertTrue(reported_completed)
            self.assertTrue(work.is_completed())
            self.assertTrue(future.done())
            for result in [work.result(), future.value(), *returns]:
                self.assertEqual([tensor._cdata for tensor in result], identities)
            self.assertEqual([part.data_ptr() for part in parts], pointers)
            for tensor, indices_ptr, values_ptr in sparse_pointers:
                self.assertFalse(tensor.is_coalesced())
                self.assertEqual(tensor._indices().data_ptr(), indices_ptr)
                self.assertEqual(tensor._values().data_ptr(), values_ptr)

            negative_is_initial = all(
                torch.equal(snapshot.cpu(), before)
                for snapshot, before in zip(negative, initial)
            )

            def check():
                for snapshot in snapshots:
                    self.assertEqual(snapshot, expected, atol=0, rtol=0)

            return (delay_pending, negative_is_initial), check

        self._assert_in_stale_window(attempt)

    @deviceCountAtLeast(2)
    def test_result_work_orders_current_stream_per_device(self, devices):
        pg = self._make_pg()
        devices = [torch.device(d) for d in devices[:2]]
        producers = [torch.cuda.Stream(device=d) for d in devices]
        consumers = [torch.cuda.Stream(device=d) for d in devices]
        initial = torch.arange(1, 5, dtype=torch.float32)
        for d in devices:
            # Load the kernels first so lazy module loading cannot close the window.
            with torch.cuda.device(d):
                torch.cuda._sleep(1)
                initial.to(d).mul_(1).clone()
            torch.cuda.synchronize(d)

        def attempt(cycles):
            results = [initial.to(d) for d in devices]
            for d in devices:
                torch.cuda.synchronize(d)
            ready = []
            with torch.cuda.stream(producers[0]), torch.cuda.stream(producers[1]):
                for d, result in zip(devices, results):
                    with torch.cuda.device(d):
                        ready.append(torch.cuda.Event())
                        ready[-1].record()
                        torch.cuda._sleep(cycles)
                        result.mul_(2)
                work = pg.broadcast(results)

            with torch.cuda.stream(consumers[0]), torch.cuda.stream(consumers[1]):
                negative = []
                for consumer, event, result in zip(consumers, ready, results):
                    consumer.wait_event(event)
                    negative.append(result.clone())
                self.assertTrue(work.wait())
                snapshots = [result.clone() for result in results]
            for d in devices:
                torch.cuda.synchronize(d)

            def check():
                self.assertEqual([s.cpu() for s in snapshots], [initial * 2] * 2)

            return tuple(torch.equal(n.cpu(), initial) for n in negative), check

        self._assert_in_stale_window(attempt)


instantiate_device_type_tests(TestFakeWorkStreams, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
