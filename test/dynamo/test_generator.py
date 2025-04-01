# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

import torch
import torch._dynamo.test_case


class GeneratorTestsBase(torch._dynamo.test_case.CPythonTestCase):
    def setUp(self):
        super().setUp()
        self._old = torch._dynamo.config.enable_faithful_generator_behavior
        torch._dynamo.config.enable_faithful_generator_behavior = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_faithful_generator_behavior = self._old


class GeneratorCloseTests(GeneratorTestsBase):
    # Taken from commit
    # https://github.com/python/cpython/blob/d51a4ca1123e3e49e5cae4273355bdfd9e419a10

    def test_close_no_return_value(self):
        def f():
            yield

        gen = f()
        gen.send(None)
        self.assertIsNone(gen.close())

    def test_close_return_value(self):
        def f():
            try:
                yield
                # close() raises GeneratorExit here, which is caught
            except GeneratorExit:
                return 0

        gen = f()
        gen.send(None)
        self.assertEqual(gen.close(), 0)

    def test_close_not_catching_exit(self):
        def f():
            yield
            # close() raises GeneratorExit here, which isn't caught and
            # therefore propagates -- no return value
            return 0

        gen = f()
        gen.send(None)
        self.assertIsNone(gen.close())

    def test_close_not_started(self):
        def f():
            try:
                yield
            except GeneratorExit:
                return 0

        gen = f()
        self.assertIsNone(gen.close())

    def test_close_exhausted(self):
        def f():
            try:
                yield
            except GeneratorExit:
                return 0

        gen = f()
        next(gen)
        with self.assertRaises(StopIteration):
            next(gen)
        self.assertIsNone(gen.close())

    def test_close_closed(self):
        def f():
            try:
                yield
            except GeneratorExit:
                return 0

        gen = f()
        gen.send(None)
        self.assertEqual(gen.close(), 0)
        self.assertIsNone(gen.close())

    def test_close_raises(self):
        def f():
            try:
                yield
            except GeneratorExit:
                pass
            raise RuntimeError

        gen = f()
        gen.send(None)
        with self.assertRaises(RuntimeError):
            gen.close()


class GeneratorThrowTests(GeneratorTestsBase):
    # Taken from commit
    # https://github.com/python/cpython/blob/d51a4ca1123e3e49e5cae4273355bdfd9e419a10

    def test_exception_context_with_yield(self):
        def f():
            try:
                raise KeyError("a")
            except Exception:
                yield

        gen = f()
        gen.send(None)
        with self.assertRaises(ValueError) as cm:
            gen.throw(ValueError)
        context = cm.exception.__context__
        self.assertEqual((type(context), context.args), (KeyError, ("a",)))

    def test_exception_context_with_yield_inside_generator(self):
        # Check that the context is also available from inside the generator
        # with yield, as opposed to outside.
        def f():
            try:
                raise KeyError("a")
            except Exception:
                try:
                    yield
                except Exception as exc:
                    self.assertEqual(type(exc), ValueError)
                    context = exc.__context__
                    self.assertEqual((type(context), context.args), (KeyError, ("a",)))
                    yield "b"

        gen = f()
        gen.send(None)
        actual = gen.throw(ValueError)
        # This ensures that the assertions inside were executed.
        self.assertEqual(actual, "b")

    def test_exception_context_with_yield_from(self):
        def f():
            yield

        def g():
            try:
                raise KeyError("a")
            except Exception:
                yield from f()

        gen = g()
        gen.send(None)
        with self.assertRaises(ValueError) as cm:
            gen.throw(ValueError)
        context = cm.exception.__context__
        self.assertEqual((type(context), context.args), (KeyError, ("a",)))

    def test_exception_context_with_yield_from_with_context_cycle(self):
        # Check trying to create an exception context cycle:
        # https://bugs.python.org/issue40696
        has_cycle = None

        def f():
            yield

        def g(exc):
            nonlocal has_cycle
            try:
                raise exc
            except Exception:
                try:
                    yield from f()
                except Exception as exc:
                    has_cycle = exc is exc.__context__
            yield

        exc = KeyError("a")
        gen = g(exc)
        gen.send(None)
        gen.throw(exc)
        # This also distinguishes from the initial has_cycle=None.
        self.assertEqual(has_cycle, False)

    def test_throw_after_none_exc_type(self):
        def g():
            try:
                raise KeyError
            except KeyError:
                pass

            try:
                yield
            except Exception:
                raise RuntimeError

        gen = g()
        gen.send(None)
        with self.assertRaises(RuntimeError) as cm:
            gen.throw(ValueError)


class GeneratorTests(GeneratorTestsBase):
    # Taken from commit
    # https://github.com/python/cpython/blob/d51a4ca1123e3e49e5cae4273355bdfd9e419a10

    def test_send_non_none_to_new_gen(self):
        def f():
            yield 1

        g = f()
        with self.assertRaises(TypeError):
            g.send(0)
        self.assertEqual(next(g), 1)

    def test_issue103488(self):
        def gen_raises():
            yield
            raise ValueError()

        def loop():
            try:
                for _ in gen_raises():
                    if True is False:
                        return
            except ValueError:
                pass

        # This should not raise
        loop()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
