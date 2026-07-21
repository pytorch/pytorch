# Owner(s): ["module: dynamo"]

import os
import torch
import torch._dynamo.test_case


class FrameHookTests(torch._dynamo.test_case.TestCase):
    def test_frame_hook_simple(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.sin()

        x = torch.randn(2)
        self.assertEqual(fn(x), x.sin())

    def test_frame_hook_error(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            raise AssertionError("test error")

        x = torch.randn(2)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn(x)

    def test_frame_hook_with_cell_vars(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x.sin()

            def inner():
                return y

            return inner()

        x = torch.randn(2)
        self.assertEqual(fn(x), x.sin())

    def test_frame_hook_with_free_vars(self):
        l1 = torch.randn(2)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.sin() + l1

        x = torch.randn(2)
        self.assertEqual(fn(x), x.sin() + l1)

    def test_frame_hook_with_graph_break(self):
        @torch.compile(backend="eager", fullgraph=False)
        def fn(x):
            y = x.sin()
            torch._dynamo.graph_break()
            z = x.cos()
            return y + z

        x = torch.randn(2)
        self.assertEqual(fn(x), x.sin() + x.cos())

    def test_frame_hook_generator_resume_with_exception_state(self):
        # A generator that yields while handling an exception carries its own
        # exception state (gi_exc_state), linked into the thread's exc_info
        # across each resume. The frame hook must not run on a generator resume:
        # doing so corrupts that exception-state chain, leaving a dangling
        # exc_value that later crashes the interpreter ("generator already
        # executing" followed by a segfault in _PyErr_SetObject).
        def gen():
            try:
                raise ValueError("boom")
            except ValueError:
                yield 1
                yield 2

        @torch.compile(backend="eager", fullgraph=False)
        def fn(x):
            x = x.sin()
            torch._dynamo.graph_break()
            g = gen()
            next(g)  # enter except block: gi_exc_state now live
            next(g)  # resume while the exception is still being handled
            g.close()  # GeneratorExit unwind through the except block
            return x.cos()

        x = torch.randn(2)
        self.assertEqual(fn(x), x.sin().cos())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
