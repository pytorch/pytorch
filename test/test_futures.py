import threading
import time
import torch
import unittest
from torch.futures import Future
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase, TemporaryFileName, run_tests
from typing import TypeVar

T = TypeVar("T")


def add_one(fut):
    return fut.wait() + 1


class TestFuture(TestCase):
    def test_set_exception(self) -> None:
        # This test is to ensure errors can propagate across futures.
        error_msg = "Intentional Value Error"
        value_error = ValueError(error_msg)

        f = Future[T]()
        # Set exception
        f.set_exception(value_error)  # type: ignore
        # Exception should throw on wait
        with self.assertRaisesRegex(ValueError, "Intentional"):
            f.wait()

        # Exception should also throw on value
        f = Future()
        f.set_exception(value_error)  # type: ignore
        with self.assertRaisesRegex(ValueError, "Intentional"):
            f.value()  # type: ignore

        def cb(fut):
            fut.value()  # type: ignore

        f = Future()
        f.set_exception(value_error)  # type: ignore

        with self.assertRaisesRegex(RuntimeError, "Got the following error"):
            cb_fut = f.then(cb)
            cb_fut.wait()

    def test_set_exception_multithreading(self) -> None:
        # Ensure errors can propagate when one thread waits on future result
        # and the other sets it with an error.
        error_msg = "Intentional Value Error"
        value_error = ValueError(error_msg)

        def wait_future(f):
            with self.assertRaisesRegex(ValueError, "Intentional"):
                f.wait()

        f = Future[T]()
        t = threading.Thread(target=wait_future, args=(f, ))
        t.start()
        f.set_exception(value_error)  # type: ignore
        t.join()

        def cb(fut):
            fut.value()  # type: ignore

        def then_future(f):
            fut = f.then(cb)
            with self.assertRaisesRegex(RuntimeError, "Got the following error"):
                fut.wait()

        f = Future[T]()
        t = threading.Thread(target=then_future, args=(f, ))
        t.start()
        f.set_exception(value_error)  # type: ignore
        t.join()

    def test_done(self) -> None:
        f = Future[torch.Tensor]()
        self.assertFalse(f.done())

        f.set_result(torch.ones(2, 2))
        self.assertTrue(f.done())

    def test_done_exception(self) -> None:
        err_msg = "Intentional Value Error"

        def raise_exception(unused_future):
            raise RuntimeError(err_msg)

        f1 = Future[torch.Tensor]()
        self.assertFalse(f1.done())
        f1.set_result(torch.ones(2, 2))
        self.assertTrue(f1.done())

        f2 = f1.then(raise_exception)
        self.assertTrue(f2.done())
        with self.assertRaisesRegex(RuntimeError, err_msg):
            f2.wait()

    def test_wait(self) -> None:
        f = Future[torch.Tensor]()
        f.set_result(torch.ones(2, 2))

        self.assertEqual(f.wait(), torch.ones(2, 2))

    def test_wait_multi_thread(self) -> None:

        def slow_set_future(fut, value):
            time.sleep(0.5)
            fut.set_result(value)

        f = Future[torch.Tensor]()

        t = threading.Thread(target=slow_set_future, args=(f, torch.ones(2, 2)))
        t.start()

        self.assertEqual(f.wait(), torch.ones(2, 2))
        t.join()

    def test_mark_future_twice(self) -> None:
        fut = Future[int]()
        fut.set_result(1)
        with self.assertRaisesRegex(
            RuntimeError,
            "Future can only be marked completed once"
        ):
            fut.set_result(1)

    def test_pickle_future(self):
        fut = Future[int]()
        errMsg = "Can not pickle torch.futures.Future"
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                torch.save(fut, fname)

    def test_then(self):
        fut = Future[torch.Tensor]()
        then_fut = fut.then(lambda x: x.wait() + 1)

        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        self.assertEqual(then_fut.wait(), torch.ones(2, 2) + 1)

    def test_chained_then(self):
        fut = Future[torch.Tensor]()
        futs = []
        last_fut = fut
        for _ in range(20):
            last_fut = last_fut.then(add_one)
            futs.append(last_fut)

        fut.set_result(torch.ones(2, 2))

        for i in range(len(futs)):
            self.assertEqual(futs[i].wait(), torch.ones(2, 2) + i + 1)

    def _test_then_error(self, cb, errMsg):
        fut = Future[int]()
        then_fut = fut.then(cb)

        fut.set_result(5)
        self.assertEqual(5, fut.wait())
        with self.assertRaisesRegex(RuntimeError, errMsg):
            then_fut.wait()

    def test_then_wrong_arg(self):

        def wrong_arg(tensor):
            return tensor + 1

        self._test_then_error(wrong_arg, "unsupported operand type.*Future.*int")

    def test_then_no_arg(self):

        def no_arg():
            return True

        self._test_then_error(no_arg, "takes 0 positional arguments but 1 was given")

    def test_then_raise(self):

        def raise_value_error(fut):
            raise ValueError("Expected error")

        self._test_then_error(raise_value_error, "Expected error")

    def test_add_done_callback_simple(self):
        callback_result = False

        def callback(fut):
            nonlocal callback_result
            fut.wait()
            callback_result = True

        fut = Future[torch.Tensor]()
        fut.add_done_callback(callback)

        self.assertFalse(callback_result)
        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        self.assertTrue(callback_result)

    def test_add_done_callback_maintains_callback_order(self):
        callback_result = 0

        def callback_set1(fut):
            nonlocal callback_result
            fut.wait()
            callback_result = 1

        def callback_set2(fut):
            nonlocal callback_result
            fut.wait()
            callback_result = 2

        fut = Future[torch.Tensor]()
        fut.add_done_callback(callback_set1)
        fut.add_done_callback(callback_set2)

        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        # set2 called last, callback_result = 2
        self.assertEqual(callback_result, 2)

    def _test_add_done_callback_error_ignored(self, cb):
        fut = Future[int]()
        fut.add_done_callback(cb)

        fut.set_result(5)
        # error msg logged to stdout
        self.assertEqual(5, fut.wait())

    def test_add_done_callback_error_is_ignored(self):

        def raise_value_error(fut):
            raise ValueError("Expected error")

        self._test_add_done_callback_error_ignored(raise_value_error)

    def test_add_done_callback_no_arg_error_is_ignored(self):

        def no_arg():
            return True

        # Adding another level of function indirection here on purpose.
        # Otherwise mypy will pick up on no_arg having an incompatible type and fail CI
        self._test_add_done_callback_error_ignored(no_arg)

    def test_interleaving_then_and_add_done_callback_maintains_callback_order(self):
        callback_result = 0

        def callback_set1(fut):
            nonlocal callback_result
            fut.wait()
            callback_result = 1

        def callback_set2(fut):
            nonlocal callback_result
            fut.wait()
            callback_result = 2

        def callback_then(fut):
            nonlocal callback_result
            return fut.wait() + callback_result

        fut = Future[torch.Tensor]()
        fut.add_done_callback(callback_set1)
        then_fut = fut.then(callback_then)
        fut.add_done_callback(callback_set2)

        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        # then_fut's callback is called with callback_result = 1
        self.assertEqual(then_fut.wait(), torch.ones(2, 2) + 1)
        # set2 called last, callback_result = 2
        self.assertEqual(callback_result, 2)

    def test_interleaving_then_and_add_done_callback_propagates_error(self):
        def raise_value_error(fut):
            raise ValueError("Expected error")

        fut = Future[torch.Tensor]()
        then_fut = fut.then(raise_value_error)
        fut.add_done_callback(raise_value_error)
        fut.set_result(torch.ones(2, 2))

        # error from add_done_callback's callback is swallowed
        # error from then's callback is not
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            then_fut.wait()

    def test_collect_all(self):
        fut1 = Future[int]()
        fut2 = Future[int]()
        fut_all = torch.futures.collect_all([fut1, fut2])

        def slow_in_thread(fut, value):
            time.sleep(0.1)
            fut.set_result(value)

        t = threading.Thread(target=slow_in_thread, args=(fut1, 1))
        fut2.set_result(2)
        t.start()

        res = fut_all.wait()
        self.assertEqual(res[0].wait(), 1)
        self.assertEqual(res[1].wait(), 2)
        t.join()

    @unittest.skipIf(IS_WINDOWS, "TODO: need to fix this testcase for Windows")
    def test_wait_all(self):
        fut1 = Future[int]()
        fut2 = Future[int]()

        # No error version
        fut1.set_result(1)
        fut2.set_result(2)
        res = torch.futures.wait_all([fut1, fut2])
        print(res)
        self.assertEqual(res, [1, 2])

        # Version with an exception
        def raise_in_fut(fut):
            raise ValueError("Expected error")
        fut3 = fut1.then(raise_in_fut)
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            torch.futures.wait_all([fut3, fut2])

if __name__ == '__main__':
    run_tests()
