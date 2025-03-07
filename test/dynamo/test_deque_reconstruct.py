# Owner(s): ["module: dynamo"]

import collections
import contextlib

import torch
import torch._inductor.test_case


class TestDequeReconstruct(torch._inductor.test_case.TestCase):
    UNSET = object()

    @contextlib.contextmanager
    def set_deque_in_globals(self, value):
        assert "deque" not in globals()

        try:
            if value is not self.UNSET:
                globals()["deque"] = collections.deque

                # This does not emit a NameError
                dummy = deque([0, 1, 2], maxlen=2)  # noqa: F821
                self.assertIsInstance(dummy, collections.deque)
                self.assertEqual(list(dummy), [1, 2])

            yield
        finally:
            globals().pop("deque", None)
            assert "deque" not in globals()

    def test_deque_reconstruct(self):
        with self.set_deque_in_globals(self.UNSET):

            @torch.compile(backend="eager", fullgraph=True)
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = torch.randn(3, 4)
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))

    def test_deque_reconstruct_shallows_globals(self):
        with self.set_deque_in_globals(collections.deque):

            @torch.compile(backend="eager", fullgraph=True)
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = torch.randn(3, 4)
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
