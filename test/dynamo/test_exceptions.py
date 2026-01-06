# Owner(s): ["module: dynamo"]

import contextlib
import sys

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn
import torch.utils.checkpoint
from torch._dynamo.bytecode_transformation import Instruction
from torch._dynamo.exc import Unsupported
from torch._dynamo.symbolic_convert import SpeculationLog, SpeculationLogDivergence
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


class CustomException(Exception):
    pass


class CustomExceptionMeta(type):
    def __instancecheck__(cls, instance):
        return True


class CustomExceptionWithInstanceCheck(Exception, metaclass=CustomExceptionMeta):
    pass


class CustomExceptionWithArgs(Exception):
    def __init__(self, a, b=None):
        self.a = a
        self.b = b


class MyException(OSError):
    pass


class ExceptionTests(torch._dynamo.test_case.TestCase):
    def test_exception(self):
        def fn(x):
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                raise NotImplementedError
            except Exception:
                x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception2(self):
        def fn(x):
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                raise NotImplementedError
            except (NotImplementedError, AttributeError):
                x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception3(self):
        def fn(x):
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                raise NotImplementedError("Not implemented")
            except AssertionError:
                x = torch.sigmoid(x)
            except NotImplementedError:
                x = torch.cos(x)
            finally:
                x = torch.cos(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception4(self):
        def fn(x):
            for i in range(10):
                if i == 5:
                    return x
                try:
                    x = torch.sin(x)
                    raise NotImplementedError
                except Exception:
                    x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_with_another_exception(self):
        def fn(x):
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                raise NotImplementedError("Not implemented")
            except NotImplementedError:
                x = torch.sigmoid(x)
                try:
                    x = torch.cos(x)
                    raise AssertionError  # noqa: B904
                except AssertionError:
                    x = torch.cos(x)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_with_vars(self):
        def fn(x):
            try:
                vars(42)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_autocast_with_exception(self):
        class Optimizer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                raise NotImplementedError("Not implemented")

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out

        @torch.compile
        def f(x: torch.Tensor):
            try:
                with torch.autocast(device_type="cpu", dtype=None):
                    Optimizer.apply(x)
            except NotImplementedError:
                return x + 1

        inp = torch.ones(3)
        out = f(inp)
        self.assertTrue(torch.equal(out, inp + 1))

    @make_dynamo_test
    def test_isinstance_CustomException(self):
        assert isinstance(CustomException, type)
        assert not isinstance(CustomException(), type)
        C = CustomExceptionWithInstanceCheck
        assert isinstance(C, C)
        assert isinstance(C(), C)

    @make_dynamo_test
    def test_propagate_exception_inside_ctx_manager(self):
        @contextlib.contextmanager
        def cm():
            try:
                yield
            except BaseException:  # noqa: B036
                raise ValueError  # noqa: B904

        @contextlib.contextmanager
        def nothing():
            try:
                yield
            finally:
                pass

        z = 0
        with nothing():
            try:
                with cm():
                    raise IndexError
            except ValueError:
                z = 1
            except IndexError:
                z = 2
            assert z == 1

    def test_exception_else(self):
        def gn(x):
            return torch.cos(x)

        def fn(x):
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                x = gn(x)
            except Exception:
                x = torch.sigmoid(x)
            else:
                x = torch.cos(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @make_dynamo_test
    def test_raise_match(self):
        a = AttributeError
        b = BytesWarning
        c = ConnectionError
        d = DeprecationWarning
        e = Exception

        def fn(a, b):
            try:
                raise a
            finally:
                raise b

        def fix_exc_context(frame_exc, new_exc, old_exc):
            # slightly change from ExitStack.fix_exc_context function
            while 1:
                exc_context = new_exc.__context__
                if exc_context is None or exc_context is old_exc:
                    return
                if exc_context is frame_exc:
                    break
                new_exc = exc_context
            new_exc.__context__ = old_exc

        @contextlib.contextmanager
        def ctx():
            try:
                yield
            finally:
                frame_exc = prev_exc = sys.exc_info()
                args = [(d, c), (b, a)]
                for x, y in args:
                    try:
                        fn(x, y)
                    except BaseException:  # noqa: B036
                        new_exc = sys.exc_info()
                        fix_exc_context(frame_exc[1], new_exc[1], prev_exc[1])
                        prev_exc = new_exc

                try:
                    fixed_ctx = prev_exc[1].__context__
                    raise prev_exc[1]
                except BaseException:  # noqa: B036
                    prev_exc[1].__context__ = fixed_ctx
                    raise

        try:
            with ctx():
                raise e
        except Exception as exc:
            assert isinstance(exc, a)
            assert isinstance(exc.__context__, b)
            assert isinstance(exc.__context__.__context__, c)
            assert isinstance(exc.__context__.__context__.__context__, d)
            assert isinstance(exc.__context__.__context__.__context__.__context__, e)

    # TODO(anijain2305) - does not work with fullgraph=True
    def test_exception_with_another_exception2(self):
        def gn(x):
            try:
                x = torch.cos(x)
                raise NotImplementedError("Not implemented")
            except NotImplementedError:
                x = torch.sigmoid(x)
                raise

        def fn(x):
            try:
                x = torch.cos(x)
                gn(x)
            except Exception:
                pass
            return x

        x = torch.randn(4)
        fn(x)
        # Can't use fullgraph=True because RERAISE is not supported
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(x)

    def test_exception_with_ctx_manager(self):
        def fn(x):
            x = torch.cos(x)
            try:
                with torch.no_grad():
                    x = torch.sin(x)
                    raise NotImplementedError("Not implemented")
            except NotImplementedError:
                x = torch.sigmoid(x)
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_raised_from_child(self):
        def gn():
            raise NotImplementedError("foo")

        def fn(x):
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                gn()
                x = torch.sin(x)
            except Exception:
                x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_dynamo_undo_kw_names(self):
        def g(x, k=None):
            if k:
                raise TypeError("error")
            return x.sin()

        def fn(x):
            d = {"a": x}
            try:
                g(x, k=True)
            except Exception:
                y = 0
                for _, b in d.items():  # noqa: PERF102
                    y += b.sum()
            return y

        x = torch.randn(2, 3)
        expected = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        got = opt_fn(x)
        self.assertEqual(expected, got)

    def test_raise_custom_exception(self):
        class Exc(Exception):
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise Exc
            except Exc:
                return t.sin()
            except Exception:
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())

    def test_raise_custom_exception_with_args(self):
        class Exc(Exception):
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise Exc(1, 2.0)
            except Exc as e:
                return t.sin() + e.args[0] + e.args[1]
            except Exception:
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin() + 1 + 2.0)

    def test_nn_module_getattr(self):
        class A:
            def __init__(self) -> None:
                self._b = 20

            def __getattr__(self, name):
                fixed_name = "_" + name
                if fixed_name in self.__dict__:
                    return self.__dict__[fixed_name]
                raise AttributeError(f"{name} absent")

        class B(A):
            def __init__(self) -> None:
                self.a = 10

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return 30

        obj = B()

        def fn(x):
            return x * obj.a * obj.b * obj.c

        x = torch.ones(4)
        ref = fn(x)
        print(ref)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    def test_custom_getattr_on_module_exception(self):
        class Foo(torch.nn.Module):
            def __init__(self, a=3):
                super().__init__()
                self.register_parameter("a", torch.nn.Parameter(torch.ones(4) * 2))

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)  # defer to nn.Module's logic
                except AttributeError:
                    if name == "a_copy":
                        return self.a
                    raise

            def forward(self, x):
                return x * self.a * self.a_copy

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)

        x = torch.ones(4)
        self.assertEqual(mod(x), opt_mod(x))

    def test_attribute_error_from_getattr(self):
        class Mock:
            def __init__(self):
                self.a = 5

            def __getattr__(self, name):
                if name != "a":
                    raise AttributeError("missing")
                return self.__dict__["a"]

        mock = Mock()

        def fn(x):
            if hasattr(mock, "b"):
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_stop_iteration(self):
        def zip_longest(*iterables, fillvalue=None):
            # Get the iterators for each iterable
            iterators = [iter(it) for it in iterables]

            result = []
            while True:
                for it in iterators:
                    try:
                        value = next(it)
                    except StopIteration:
                        result.append(fillvalue)
                        return result
                    result.append(value)

        def fn(x, y):
            torch.cos(torch.randn(4))
            return tuple(zip_longest(x, y))

        x = [1, 2, 3, 4]
        y = [10, 11, 12]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nn_reraise(self):
        class M(torch.nn.Module):
            def forward(self, x):
                raise ValueError("woof")
                return x + 2

        m = M()
        m.register_forward_pre_hook(lambda m, go: None)

        torch._dynamo.utils.clear_compilation_metrics()
        opt_call = torch.compile(lambda x: m(x), backend="eager")
        self.assertRaises(ValueError, lambda: opt_call(torch.randn(3)))
        metrics = torch._dynamo.utils.get_compilation_metrics()
        self.assertIn("Observed exception", metrics[0].fail_reason)

    def test_key_error(self):
        def fn(x, d):
            try:
                a = d["b"]
            except KeyError:
                a = 2
            return x * a

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        d = {"a": 1}
        ref = fn(x, d)
        res = opt_fn(x, d)
        self.assertEqual(ref, res)

    def test_atrribute_error(self):
        class Mock:
            def __init__(self):
                self.a = 1

        mock = Mock()

        def fn(x):
            try:
                c = 2
                mock.b
            except AttributeError:
                c = 3
            return torch.sin(x) * c

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_raise_from_None(self):
        # Inspired from os.environ
        class MyMapping:
            def __init__(self, d):
                self._d = d

            def __getitem__(self, key):
                try:
                    value = self._d[key]
                except KeyError:
                    raise KeyError(key) from None
                return value

        d = MyMapping({"a": 10, "b": 20})

        def mapping_get(obj, key, value=None):
            try:
                return obj.__getitem__(key)
            except KeyError:
                return value

        def fn(x, d, key):
            x = torch.sin(x + 1)
            return x, mapping_get(d, key)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.rand(2, 3)
        ref = fn(x, d, "m")
        res = opt_fn(x, d, "m")
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

    @make_dynamo_test
    def test_raise_from_None_2(self):
        def fn():
            try:
                raise ValueError
            except Exception:
                raise TypeError from None

        try:
            fn()
        except TypeError as e:
            assert e.__cause__ is None
            assert e.__suppress_context__ is True

    @make_dynamo_test
    def test_raise_from_other(self):
        def fn():
            try:
                raise ValueError
            except Exception as e:
                raise TypeError from e

        try:
            fn()
        except TypeError as e:
            assert isinstance(e.__cause__, ValueError)
            assert e.__suppress_context__ is True

    @make_dynamo_test
    def test_reraise_first_exc(self):
        def fn():
            try:
                raise ZeroDivisionError
            except ZeroDivisionError:
                try:
                    raise ValueError  # noqa: B904
                except ValueError:
                    pass
                raise

        try:
            fn()
        except ZeroDivisionError:
            pass
        assert sys.exc_info()[0] is None

    @make_dynamo_test
    def test_ensure_exception_is_active_after_try_except_block(self):
        try:
            try:
                raise ZeroDivisionError
            except ZeroDivisionError:
                for exc in (KeyError, IndexError):
                    try:
                        raise exc
                    except exc:
                        pass
                raise
        except ZeroDivisionError:
            pass
        assert sys.exc_info()[0] is None

    @make_dynamo_test
    def test_ensure_exception_is_active_inside_try_except_block(self):
        try:
            try:
                raise ZeroDivisionError
            except ZeroDivisionError:
                for exc in (KeyError, IndexError):
                    try:
                        raise exc
                    except exc as e:
                        assert isinstance(e.__context__, ZeroDivisionError)
                raise
        except ZeroDivisionError:
            pass
        assert sys.exc_info()[0] is None

    @make_dynamo_test
    def test_handle_all_exceptions(self):
        def cm():
            try:
                yield 1
            except ValueError:
                try:
                    raise TypeError  # noqa: B904
                finally:
                    pass

        try:
            gen = cm()
            next(gen)
            gen.throw(ValueError)
        except TypeError:
            pass
        assert sys.exc_info()[0] is None

    @make_dynamo_test
    def test_reraise(self):
        try:
            try:
                raise ValueError
            except ValueError:  # noqa: TRY203
                raise
        except ValueError:
            pass
        assert sys.exc_info()[0] is None

    @make_dynamo_test
    def test_raise_finally_simple(self):
        def fn():
            try:
                raise ValueError
            except ValueError:
                try:
                    raise TypeError  # noqa: B904
                finally:
                    pass

        try:
            fn()
        except TypeError:
            pass
        assert sys.exc_info()[0] is None

    def test_reconstruct___context__(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            v = ValueError(1, 2, 3)
            v.__context__ = TypeError()
            v.__cause__ = RuntimeError()
            return t.sin(), v

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, ValueError)
        self.assertIsInstance(v.__context__, TypeError)
        self.assertIsInstance(v.__cause__, RuntimeError)
        self.assertTrue(v.__suppress_context__)

    def test_reconstruct_exception_2(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise ValueError(1, 2, 3)
            except Exception:
                try:
                    raise TypeError(4, 5) from None
                except Exception as e:
                    e.__cause__ = RuntimeError(6, 7)
                    return t.sin(), e

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, TypeError)
        self.assertIsInstance(v.__context__, ValueError)
        self.assertIsInstance(v.__cause__, RuntimeError)

    def test_raise_GeneratorExit(self):
        # GeneratorExit does not inherit from Exception
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise GeneratorExit
            except Exception:
                return t.sin()
            except BaseException:  # noqa: B036
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.cos())

    def test_speculation_exception(self):
        log = SpeculationLog()
        log.next("fake", 555, "fake", Instruction(1, "fake", 1, 1))
        log.restart()
        with self.assertRaises(SpeculationLogDivergence):
            log.next("bad", 58, "bad", Instruction(2, "different", 2, 2))

    def test_dict_pop(self):
        # Pattern from inspect.bind
        def fn(dt, x):
            try:
                dt.pop("b")
            except KeyError:
                return torch.sin(x)
            else:
                return torch.cos(x)

        d = {"a": 1}
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)
        self.assertEqual(fn(d, x), opt_fn(d, x))
        self.assertEqual(fn({"a": 1, "b": 2}, x), opt_fn({"a": 1, "b": 2}, x))

    def test_block_stack_cleanup(self):
        params = {
            "a": 3,
            "b": 4,
            "c": 5,
        }

        dt = {
            "c": 5,
        }

        def fn(x):
            for name in params:
                try:
                    x = x * dt[name]
                except KeyError:
                    x = x * torch.sin(x)
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_set_cause_with_arg(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t, err):
            err.__cause__ = ValueError()
            return t.sin()

        t = torch.randn(2)
        e = TypeError("abcd")
        fn(t, e)
        self.assertIsInstance(e.__cause__, ValueError)

    def test_set_cause_with_arg_error(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t, err):
            err.__cause__ = 2
            return t.sin()

        t = torch.randn(2)
        e = TypeError("abcd")
        with self.assertRaisesRegex(TypeError, "exception cause must be"):
            fn(t, e)

    @parametrize(
        "ex",
        [TypeError, CustomException],
        name_fn=lambda x: x.__name__,
    )
    @make_dynamo_test
    def test_set___cause__(self, ex):
        def fn():
            try:
                raise ex
            except ex:
                raise TypeError from None

        try:
            fn()
        except TypeError as e:
            assert isinstance(e.__context__, ex)
            assert e.__cause__ is None
            assert e.__suppress_context__ is True

    @parametrize(
        "ex",
        [RuntimeError, CustomException],
        name_fn=lambda x: x.__name__,
    )
    @make_dynamo_test
    def test_set___cause___error(self, ex):
        def fn():
            try:
                raise ex
            except Exception as e:
                e.__cause__ = 2
                raise

        z = 0

        try:
            fn()
        except TypeError as e:
            z = 1
            assert e.args == (
                "exception cause must be None or derive from BaseException",
            )
        except Exception:
            raise AssertionError from None

        assert z == 1

    def test_user_defined_exception_variable(self):
        def fn(t):
            z = 0
            try:
                raise CustomException
            except ValueError:
                z = 1
            except CustomException as e:
                # trying to call python_type on the
                # UserDefinedExceptionClassVariable
                cls = type(e)
                if type(cls) is type:
                    t = t + 1
                z = 2
            assert z == 2
            return t.sin()

        t = torch.randn(2)
        fn(t)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(t), opt_fn(t))

    def test_user_defined_exception_with_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            z = 0
            try:
                raise CustomExceptionWithArgs(2, b=3)
            except ValueError:
                z = 1
            except CustomExceptionWithArgs:
                z = 2
            assert z == 2

        t = torch.randn(2)
        fn(t)

    @make_dynamo_test
    def test_raise_set___context__(self):
        try:
            raise TypeError
        except TypeError as e:
            exc = e

        assert exc.__context__ is None

        try:
            raise ValueError
        except ValueError as e:
            exc2 = e

        assert exc2.__context__ is None

    def test_exception_kwargs(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            raise AttributeError(name="a")

        self.assertRaises(Unsupported, fn)

    def test_stack_trace_from_observed_exception(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                # no attribute w on self.linear
                weight = self.linear.w
                return torch.nn.functional.linear(x, weight)

        x = (torch.randn(4, 16, requires_grad=True),)

        with self.assertRaisesRegex(Exception, "weight = self.linear.w"):
            torch._dynamo.functional_export.dynamo_graph_capture_for_export(Model())(x)

    def test_context_manager_preserves_exception_stack(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/167900
        # When an exception is raised inside a context manager and the context manager
        # doesn't suppress it, the error message should point to the original raise
        # location, not the context manager cleanup code.
        def g():
            assert False  # noqa: B011

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            with torch.no_grad():
                g()
            return x

        with self.assertRaises(Unsupported) as ctx:
            f(torch.randn(1))

        # The error should point to "assert False" in g(), not "return x"
        self.assertIn("in g", str(ctx.exception))
        self.assertIn("assert False", str(ctx.exception))
        self.assertNotIn("return x", str(ctx.exception))

    def test_reraise_preserves_exception_stack(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/167900
        # When an exception is caught and re-raised, the error message should
        # point to the original raise location, not the reraise location.
        def g():
            raise Exception("Invalid")  # noqa: TRY002

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            try:
                g()
            except Exception:  # noqa: TRY203
                raise
            return x

        with self.assertRaises(Unsupported) as ctx:
            f(torch.randn(1))

        # The error should point to 'raise Exception("Invalid")' in g()
        self.assertIn("in g", str(ctx.exception))
        self.assertIn('raise Exception("Invalid")', str(ctx.exception))


instantiate_parametrized_tests(ExceptionTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
