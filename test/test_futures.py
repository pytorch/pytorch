import threading
import time
import torch
from torch.futures import Future
from torch.testing._internal.common_utils import TestCase, TemporaryFileName


def add_one(fut):
    return fut.wait() + 1


class TestFuture(TestCase):
    def test_wait(self):
        f = Future()
        f.set_result(torch.ones(2, 2))

        self.assertEqual(f.wait(), torch.ones(2, 2))

    def test_wait_multi_thread(self):

        def slow_set_future(fut, value):
            time.sleep(0.5)
            fut.set_result(value)

        f = Future()

        t = threading.Thread(target=slow_set_future, args=(f, torch.ones(2, 2)))
        t.start()

        self.assertEqual(f.wait(), torch.ones(2, 2))
        t.join()

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

    def _test_error(self, cb, errMsg):
        fut = Future()
        then_fut = fut.then(cb)

        fut.set_result(5)
        self.assertEqual(5, fut.wait())
        with self.assertRaisesRegex(RuntimeError, errMsg):
            then_fut.wait()

    def test_then_wrong_arg(self):

        def wrong_arg(tensor):
            return tensor + 1

        self._test_error(wrong_arg, "unsupported operand type.*Future.*int")

    def test_then_no_arg(self):

        def no_arg():
            return True

        self._test_error(no_arg, "takes 0 positional arguments but 1 was given")

    def test_then_raise(self):

        def raise_value_error(fut):
            raise ValueError("Expected error")

        self._test_error(raise_value_error, "Expected error")
