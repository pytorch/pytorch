# Do not add this to test/run_test.py as this is run with
# test/test_autograd.py.
#
# If you add a TestCase here, import it in test/test_autograd.py.

import sys
import threading
import torch
import torch.autograd

from torch.testing._internal.common_utils import TestCase, run_tests

class TestMultithreadAutograd(TestCase):
    def _run_py_multithread_fn(self, fn, args=(), num_threads=10, kwargs=None):

        class PropagatingThread(threading.Thread):
            '''Helper class to propagate exception from child
            thread to main thread on join.

            Reference: https://stackoverflow.com/a/31614591/5602957
            '''

            def run(self):
                self.exception = None
                try:
                    self.ret = super(PropagatingThread, self).run()
                except Exception as e:
                    self.exception = e

            def join(self, timeout=None):
                super(PropagatingThread, self).join(timeout)
                if self.exception:
                    raise self.exception from self.exception
                return self.ret

        threads = []
        for _ in range(num_threads):
            p = PropagatingThread(target=fn, args=args)
            p.start()
            threads.append(p)

        for p in threads:
            p.join()

    def test_multithreaded_exception_propagation(self):
        # Test whether exception in child thread
        # are propagated to main thread.
        def fn():
            self.assertTrue(False)

        with self.assertRaises(AssertionError):
            self._run_py_multithread_fn(fn)

    def test_simple_backward(self):
        # simple multithreaded backward that create threads in the beginning of training
        # and everything else is training separately, i.e. inputs, operations, etc.
        def train_fn():
            x = torch.ones(5, 5, requires_grad=True)
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()
            self.assertEqual(x.grad, x + 3.5)

        self._run_py_multithread_fn(train_fn)

    def test_simple_backward_same_input(self):
        # simple multithreaded backward with only shared inputs (i.e. This is common
        # for things like Hogwild multithreaded training with multiple CPU threads)
        def train_fn_backward(x):
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()

        x = torch.ones(5, 5, requires_grad=True)
        self._run_py_multithread_fn(train_fn_backward, (x,))
        # Since we are calling backward from multiple threads
        # and all threads share the same input, when we do backward
        # concurrently, different backwards will all accumulate to
        # the same .grad for each input, and the gradients should
        # be equal to num_threads * gradient
        self.assertEqual(x.grad, 10 * (x + 3.5))

        def train_fn_grad(x):
            y = (x + 3) * (x + 4) * 0.5
            grads = torch.autograd.grad(y.sum(), x)
            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0], x + 3.5)

        # since we use functional grad() api, gradients will not
        # be accumulate to the same place and should be the same
        self._run_py_multithread_fn(train_fn_grad, (x,))

    def test_python_thread_in_middle(self):
        # User might write a network that starts on one CPU thread, then runs its second half
        # concurrently with other threads (either via python threading or fork/join calls),
        # then calls backward()/grad() on BOTH threads, like a Y pattern from input at the
        # bottom to output at the top. This way part of the GraphTask is being shared across
        # different threads and we need to ensure user specify retain_graph=True, otherwise
        # error out with the correct error message

        # Case 1: multiple backward with python threads, retain_graph=False
        # should throw error in some threads with no retain_graph.
        success_vs_raises = [0, 0]

        def train_fn_no_retain_graph(x):
            y = x + x ** 2
            try:
                y.sum().backward()
                success_vs_raises[0] += 1
            except RuntimeError as error:
                success_vs_raises[1] += 1
                self.assertRegex(str(error), "Specify retain_graph=True")

        x_no_retain = torch.ones(5, 5, requires_grad=True)
        y_no_retain = x_no_retain + x_no_retain ** 2
        self._run_py_multithread_fn(train_fn_no_retain_graph, (y_no_retain,), num_threads=5)
        # at least one thread will be success in this case, all other threads should raise
        # with the error that throw to user to recommend them specify retain_graph=True
        self.assertTrue(success_vs_raises[0] >= 1)

        # multiple backward with python threads, no error with retain_graph=True
        def train_fn_retain_graph(x):
            y = x + x ** 2
            y.sum().backward(retain_graph=True)

        x_retain = torch.ones(5, 5, requires_grad=True)
        y_retain = x_retain + x_retain ** 2
        self._run_py_multithread_fn(train_fn_retain_graph, (y_retain,), num_threads=5)
        # result should equal to num_thread * gradients
        self.assertEqual(x_retain.grad, 5 * (4 * x_retain ** 3 + 6 * (x_retain ** 2) + 4 * x_retain + 1))

    def test_fork_join_in_middle(self):
        # multiple backward with jit threads (fork/join primitive)
        # similar to test_python_thread_in_middle, we test with retain_graph=False/True

        # Case 1: multiple grad() calls with jit threads, retain_graph=False
        # should throw error in some threads with no retain_graph.
        @torch.jit.script
        def train_fn_jit_no_retain(middle, orig_x):
            y = middle + middle ** 2
            return torch.autograd.grad([y.sum()], [orig_x])

        @torch.jit.script
        def train_fn_fork_join_calls_no_retain(x):
            y_no_retain = (x + 3) * (x + 4) * 0.5

            fut = torch.jit._fork(train_fn_jit_no_retain, y_no_retain, x)
            grad_hat = train_fn_jit_no_retain(y_no_retain, x)
            grad = torch.jit._wait(fut)
            return grad, grad_hat

        try:
            train_fn_fork_join_calls_no_retain(torch.randn(5, 5, requires_grad=True))
        except RuntimeError as error:
            self.assertRegex(str(error), "Specify retain_graph=True")

        # Case 2: no error with retain_graph=True
        @torch.jit.script
        def train_fn_jit_retain(middle, orig_x):
            y = middle + middle ** 2
            return torch.autograd.grad([y.sum()], [orig_x], retain_graph=True)

        @torch.jit.script
        def train_fn_fork_join_calls_retain(x):
            y_retain = (x + 3) * (x + 4) * 0.5
            fut1 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            fut2 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            grad = train_fn_jit_retain(y_retain, x)
            grad1 = torch.jit._wait(fut1)
            grad2 = torch.jit._wait(fut2)
            return grad, grad1, grad2

        grad, grad1, grad2 = train_fn_fork_join_calls_retain(torch.randn(5, 5, requires_grad=True))
        self.assertEqual(grad, grad1)
        self.assertEqual(grad, grad2)

    def test_preserve_backtrace(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, *grad):
                raise ValueError("something")

        t = torch.rand(10, requires_grad=True)
        try:
            Foo.apply(t).sum().backward()
        except Exception:
            import traceback
            tb = sys.exc_info()[2]
            tb_str = "\n".join(traceback.format_tb(tb))
            self.assertTrue('raise ValueError("something")' in tb_str)

    # TODO(@anjali411): add an OpInfo based test for torch.cat
    # Issue: https://github.com/pytorch/pytorch/issues/51627
    def test_cat_r_to_c(self):
        inp_c = torch.rand(3, 2, dtype=torch.cdouble, requires_grad=True)
        inp_r = torch.randn(3, 2, dtype=torch.double, requires_grad=True)

        def fn(x1, x2):
            return torch.cat((x1, x2), dim=-1)

        torch.autograd.gradcheck(fn, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn, [inp_c, inp_r], check_forward_ad=True)


if __name__ == '__main__':
    run_tests()
