import threading
import time
import torch
from torch.futures import Future
from torch.testing._internal.common_utils import TestCase, TemporaryFileName


def slow_set_future(fut, value):
    time.sleep(0.5)
    fut.set_result(value)


def add_one(fut):
    return fut.wait() + 1


class TestFuture(TestCase):
    def test_wait(self):
        f = Future()
        f.set_result(torch.ones(2, 2))

        self.assertEqual(f.wait(), torch.ones(2, 2))

    def test_wait_multi_thread(self):
        f = Future()

        t = threading.Thread(target=slow_set_future, args=(f, torch.ones(2, 2)))
        t.daemon = True
        t.start()

        self.assertEqual(f.wait(), torch.ones(2, 2))

    def test_mark_future_twice(self):
        fut = Future()
        fut.set_result(1)
        with self.assertRaisesRegex(
            RuntimeError,
            "Future can only be marked completed once"
        ):
            fut.set_result(1)

    def test_pickle_future(self):
        fut = Future()
        errMsg = "Can not pickle torch.futures.Future"
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                torch.save(fut, fname)

    def test_then(self):
        fut = Future()
        then_fut = fut.then(lambda x: x.wait() + 1)

        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        self.assertEqual(then_fut.wait(), torch.ones(2, 2) + 1)

    def test_chained_then(self):
        fut = Future()
        futs = []
        last_fut = fut
        for _ in range(20):
            last_fut = last_fut.then(add_one)
            futs.append(last_fut)

        fut.set_result(torch.ones(2, 2))

        for i in range(len(futs)):
            self.assertEqual(futs[i].wait(), torch.ones(2, 2) + i + 1)
