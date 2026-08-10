# Owner(s): ["module: dynamo"]

import contextlib
import dataclasses
import operator
import sys
from unittest import mock

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn
import torch.utils.checkpoint
from torch._dynamo.bytecode_transformation import Instruction
from torch._dynamo.exc import TorchRuntimeError, Unsupported
from torch._dynamo.symbolic_convert import SpeculationLog, SpeculationLogDivergence
from torch._dynamo.testing import CompileCounter, EagerAndRecordGraphs, skipIfNotPy311
from torch._subclasses.fake_tensor import FakeTensorMode
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


class CustomRuntimeError(RuntimeError):
    pass


class CustomRuntimeErrorWithStr(RuntimeError):
    def __str__(self):
        return "custom str"


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
                    raise AssertionError
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

    def test_vars_too_many_args_type_error(self):
        def fn(x):
            return x + 1, vars(1, 2)

        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaisesRegex(
            TypeError, "vars expected at most 1 argument, got 2"
        ):
            opt_fn(torch.ones(1))

    def test_vars_keyword_args_type_error(self):
        def fn(x):
            try:
                vars(obj=x)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_range_too_many_args_type_error(self):
        def fn(x):
            return x + 1, range(1, 2, 3, 4, 5, 6)

        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaisesRegex(
            TypeError, "range expected at most 3 arguments, got 6"
        ):
            opt_fn(torch.ones(1))

    def test_raise_non_exception_type_error(self):
        # PyExceptionClass_Check must reject non-exception builtins: they are
        # BuiltinVariables but not BaseException subclasses.
        def fn(x):
            try:
                raise int
            except TypeError as e:
                return x.sin(), str(e)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_raise_from_non_exception_type_error(self):
        def fn(x):
            try:
                raise ValueError("v") from dict
            except TypeError as e:
                return x.sin(), str(e)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_builtin_arg_count_type_errors(self):
        def check(fn):
            x = torch.randn(4)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(x), opt_fn(x))

        def len_no_args(x):
            try:
                len()
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def len_too_many_args(x):
            try:
                len(x, x)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def getitem_too_few_args(x):
            try:
                operator.getitem(x)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def getitem_no_args(x):
            try:
                operator.getitem()
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def getitem_too_many_args(x):
            try:
                operator.getitem(x, 0, 1)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def getitem_keyword_args(x):
            try:
                operator.getitem(a=x, b=0)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def next_no_args(x):
            try:
                next()
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def next_too_many_args(x):
            try:
                next(iter([x]), x, x)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def range_no_args(x):
            try:
                range()
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def range_too_many_args(x):
            try:
                range(1, 2, 3, 4, 5, 6)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        check(len_no_args)
        check(len_too_many_args)
        check(getitem_no_args)
        check(getitem_too_few_args)
        check(getitem_too_many_args)
        check(getitem_keyword_args)
        check(next_no_args)
        check(next_too_many_args)
        check(range_no_args)
        check(range_too_many_args)

    def test_user_class_as_tensor_method_arg(self):
        class MyClass:
            pass

        def fn(x):
            try:
                y = x.new_full(MyClass, 3.14)
            except TypeError:
                y = x + 1.0
            return y

        x = torch.ones(4)
        ref = fn(x)
        res = torch.compile(fn, backend="eager")(x)
        self.assertEqual(ref, res)

    def test_autocast_with_exception(self):
        class Optimizer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                raise NotImplementedError("Not implemented")

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out

        @torch.compile(backend="eager")
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
        assert isinstance(CustomException, type)  # noqa: S101
        assert not isinstance(CustomException(), type)  # noqa: S101
        C = CustomExceptionWithInstanceCheck
        assert isinstance(C, C)  # noqa: S101
        assert isinstance(C(), C)  # noqa: S101

    @make_dynamo_test
    def test_propagate_exception_inside_ctx_manager(self):
        @contextlib.contextmanager
        def cm():
            try:
                yield
            except BaseException:
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
            assert z == 1  # noqa: S101

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
                    except BaseException:
                        new_exc = sys.exc_info()
                        fix_exc_context(frame_exc[1], new_exc[1], prev_exc[1])
                        prev_exc = new_exc

                try:
                    fixed_ctx = prev_exc[1].__context__
                    raise prev_exc[1]
                except BaseException:
                    prev_exc[1].__context__ = fixed_ctx
                    raise

        try:
            with ctx():
                raise e
        except Exception as exc:
            assert isinstance(exc, a)  # noqa: S101
            assert isinstance(exc.__context__, b)  # noqa: S101
            assert isinstance(exc.__context__.__context__, c)  # noqa: S101
            assert isinstance(exc.__context__.__context__.__context__, d)  # noqa: S101
            assert isinstance(exc.__context__.__context__.__context__.__context__, e)  # noqa: S101

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

    def test_observed_exception_formats_fstring_message(self):
        from torch.utils._pytree import tree_map_with_path

        def check_tensor(path, x):
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Expected Tensor at {path=}")
            return x * 2

        def fn(tree):
            return tree_map_with_path(check_tensor, tree)

        tree = {"a": torch.randn(10), "b": 5}

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(Unsupported) as compiled_ctx:
            compiled_fn(tree)

        exc_str = str(compiled_ctx.exception)
        self.assertIn("Observed exception", exc_str)
        self.assertIn("Expected Tensor at path=(MappingKey(key='b'),)", exc_str)
        self.assertNotIn("Failed to trace builtin operator", exc_str)
        self.assertNotIn("StringFormatVariable", exc_str)
        self.assertNotIn("ConstantVariable(", exc_str)

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

    def test_tensor_attribute_error_in_try_except(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor([2.0]))

            def forward(self, x):
                try:
                    return x.this_attribute_does_not_exist
                except AttributeError:
                    return x * self.scale

        m = M()
        opt_m = torch.compile(m, backend="eager")
        x = torch.randn(4, 4)
        ref = m(x)
        res = opt_m(x)
        self.assertEqual(ref, res)

    def test_runtime_error_in_try_except(self):
        def fn(x):
            try:
                torch.linalg.inv(x)
            except RuntimeError:
                return x + 1
            return x

        opt_m = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2, 3)
        ref = fn(x)
        res = opt_m(x)
        self.assertEqual(ref, res)

    def test_runtime_error_graph_break(self):
        cnt = CompileCounter()

        def fn(x):
            return torch.linalg.inv(x)

        opt_m = torch.compile(fn, backend=cnt)
        x = torch.randn(2, 3)
        with self.assertRaises(RuntimeError):
            opt_m(x)
        self.assertEqual(cnt.frame_count, 0)

    def test_fake_tensor_runtime_error_in_try_except(self):
        backend = EagerAndRecordGraphs()

        def fn(t):
            t0 = torch.randn(2)
            try:
                t.expand_as(t0)
            except RuntimeError:
                return t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("sin", node_targets)

    def test_fake_tensor_runtime_error_in_parent_try_except(self):
        backend = EagerAndRecordGraphs()

        def inner(t, t0):
            t.expand_as(t0)
            return t.cos()

        def fn(t):
            t0 = torch.randn(2)
            try:
                return inner(t, t0)
            except RuntimeError:
                return t.sin()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("sin", node_targets)

    def test_fake_tensor_runtime_error_in_yield_from_throw(self):
        backend = EagerAndRecordGraphs()

        def subgen(t):
            try:
                yield t.cos()
            except ValueError:
                t.expand_as(torch.randn(2))
                yield t.tan()

        def outer(t):
            yield from subgen(t)
            yield t + 1

        def fn(t):
            gen = outer(t)
            first = next(gen)
            try:
                gen.throw(ValueError)
            except RuntimeError:
                caught = t.sin()
            else:
                caught = t.cos()
            try:
                after = next(gen)
            except StopIteration:
                after = t + 2
            return first + caught + after

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("sin", node_targets)

    def test_fake_tensor_runtime_error_in_yield_from_close(self):
        backend = EagerAndRecordGraphs()

        def subgen(t):
            try:
                yield t.cos()
            finally:
                t.expand_as(torch.randn(2))

        def outer(t):
            yield from subgen(t)
            yield t + 1

        def fn(t):
            gen = outer(t)
            first = next(gen)
            try:
                gen.close()
            except RuntimeError:
                caught = t.sin()
            else:
                caught = t.cos()
            try:
                after = next(gen)
            except StopIteration:
                after = t + 2
            return first + caught + after

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("sin", node_targets)

    def test_fake_tensor_runtime_error_custom_fake_subclass_not_caught(self):
        @torch.library.custom_op("test_dynamo::runtime_error_subclass", mutates_args=())
        def runtime_error_subclass(t: torch.Tensor) -> torch.Tensor:
            return t.cos()

        @runtime_error_subclass.register_fake
        def _(t):
            raise CustomRuntimeError("fake-only custom runtime")

        def fn(t):
            try:
                return runtime_error_subclass(t)
            except CustomRuntimeError:
                return t.sin()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.cos())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(TorchRuntimeError, "fake-only custom runtime"):
            opt_fn(t)

    def test_fake_tensor_runtime_error_internal_error_not_caught(self):
        @torch.library.custom_op("test_dynamo::metadata_mismatch", mutates_args=())
        def metadata_mismatch(t: torch.Tensor) -> torch.Tensor:
            return t.cos()

        @metadata_mismatch.register_fake
        def _(t):
            raise torch._subclasses.fake_tensor.MetadataMismatchError(
                "fake-only metadata mismatch"
            )

        def fn(t):
            try:
                metadata_mismatch(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.cos())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(t)

    def test_fake_tensor_runtime_error_plain_fake_only_not_caught(self):
        @torch.library.custom_op(
            "test_dynamo::plain_fake_only_runtime_error", mutates_args=()
        )
        def plain_fake_only_runtime_error(t: torch.Tensor) -> torch.Tensor:
            return t.cos()

        @plain_fake_only_runtime_error.register_fake
        def _(t):
            raise RuntimeError("fake-only failure")

        def fn(t):
            try:
                plain_fake_only_runtime_error(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.cos())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(t)

    def test_fake_tensor_runtime_error_unmarked_builtin_not_caught(self):
        # Exact RuntimeError from mutable/C++ meta registrations is ambiguous:
        # some such errors are fake-only. Keep it hard unless the core check
        # site has explicitly opted into user exception routing.
        def fn(t):
            try:
                t.reshape(5)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.sin())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(TorchRuntimeError, "shape.*invalid"):
            opt_fn(t)

    def test_fake_tensor_runtime_error_custom_fake_check_not_caught(self):
        @torch.library.custom_op("test_dynamo::fake_only_check", mutates_args=())
        def fake_only_check(t: torch.Tensor) -> torch.Tensor:
            return t.cos()

        @fake_only_check.register_fake
        def _(t):
            torch._check(False, lambda: "fake-only check")

        def fn(t):
            try:
                fake_only_check(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.cos())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(TorchRuntimeError, "fake-only check"):
            opt_fn(t)

    def test_fake_tensor_runtime_error_nested_custom_fake_check_not_caught(self):
        @torch.library.custom_op("test_dynamo::nested_fake_only_check", mutates_args=())
        def nested_fake_only_check(t: torch.Tensor) -> torch.Tensor:
            return t.cos()

        @nested_fake_only_check.register_fake
        def _(t):
            return t.expand(2)

        def fn(t):
            try:
                nested_fake_only_check(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.cos())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(TorchRuntimeError, "too few dimensions"):
            opt_fn(t)

    def test_fake_tensor_runtime_error_nested_custom_meta_check_not_caught(self):
        lib = torch.library.Library("test_dynamo", "FRAGMENT")  # noqa: SCOPED_LIBRARY
        lib.define("nested_meta_check(Tensor t) -> Tensor")
        lib.impl("nested_meta_check", lambda t: t.cos(), "CPU")
        lib.impl("nested_meta_check", lambda t: t.expand(2), "Meta")

        def fn(t):
            try:
                torch.ops.test_dynamo.nested_meta_check(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.cos())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(TorchRuntimeError, "too few dimensions"):
            opt_fn(t)

    def test_fake_tensor_runtime_error_prim_meta_impl_not_caught(self):
        op = torch.ops.prims.sin.default

        def fn(t):
            try:
                return op(t)
            except RuntimeError:
                return t.cos()

        t = torch.randn(2, 3)
        with mock.patch.object(op, "prim_meta_impl", lambda t: t.expand(2)):
            self.assertEqual(fn(t), t.sin())
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(TorchRuntimeError, "too few dimensions"):
                opt_fn(t)

    def test_fake_tensor_runtime_error_unsafe_fallback_not_caught(self):
        lib = torch.library.Library("vision", "FRAGMENT")  # noqa: SCOPED_LIBRARY
        lib.define("fallback_user_check(Tensor t) -> Tensor")

        def fallback_user_check(t):
            if t.sum().item() == 0:
                return torch._refs.expand(t, 2)
            return t.clone()

        lib.impl("fallback_user_check", fallback_user_check, "CPU")
        op = torch.ops.vision.fallback_user_check.default

        def fn(t):
            try:
                op(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.ones(2, 3)
        self.assertEqual(fn(t), t.cos())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(TorchRuntimeError, "too few dimensions"):
            opt_fn(t)

    def test_fake_tensor_runtime_error_check_classification(self):
        def check_error(check, message):
            with self.assertRaises(RuntimeError) as cm:
                check(False, lambda: message)
            return cm.exception

        direct = check_error(torch._check, "direct")
        self.assertIs(type(direct), RuntimeError)
        self.assertEqual(repr(direct), "RuntimeError('direct')")
        self.assertNotIn("_torch_check_user_error", direct.__dict__)

        direct_user_error = check_error(torch._check_user_error, "direct user error")
        self.assertIs(type(direct_user_error), RuntimeError)

        with FakeTensorMode():
            with self.assertRaises(RuntimeError) as cm:
                torch.empty(2, 3).expand(2)
        self.assertIs(type(cm.exception), RuntimeError)

        with torch._enable_torch_check_user_error():
            user_error = check_error(torch._check_user_error, "user error")
        self.assertIs(type(user_error), torch._TorchCheckUserError)
        self.assertEqual(user_error.args, ("user error",))

        with torch._enable_torch_check_user_error():
            with torch._suppress_torch_check_user_error():
                suppressed = check_error(torch._check_user_error, "suppressed")
        self.assertIs(type(suppressed), RuntimeError)

        with torch._enable_torch_check_user_error():
            escaping = check_error(torch._check_user_error, "escaping")
            with self.assertRaisesRegex(RuntimeError, "escaping") as cm:
                with torch._suppress_torch_check_user_error():
                    raise escaping
        self.assertIs(type(cm.exception), RuntimeError)

        def bad_message():
            raise RuntimeError("message failed")

        with self.assertRaisesRegex(RuntimeError, "message failed") as cm:
            torch._check_user_error(False, bad_message)
        self.assertNotIn("_torch_check_user_error", cm.exception.__dict__)

    def test_fake_tensor_runtime_error_propagate_real_internal_error_not_caught(self):
        @torch.library.custom_op(
            "test_dynamo::propagate_real_tensors_error", mutates_args=()
        )
        def propagate_real_tensors_error(t: torch.Tensor) -> int:
            return t.dim()

        def fn(t):
            try:
                propagate_real_tensors_error(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.cos())

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(
                TorchRuntimeError, "RuntimeError when making fake tensor call"
            ):
                opt_fn(t)

    def test_fake_tensor_runtime_error_propagated_real_error_not_caught(self):
        class RealKernelError(RuntimeError):
            pass

        @torch.library.custom_op(
            "test_dynamo::propagated_real_runtime_error", mutates_args=()
        )
        def propagated_real_runtime_error(t: torch.Tensor) -> torch.Tensor:
            if t.sum().item() > 0:
                raise RealKernelError("real kernel rejected positive input")
            return t.clone()

        @propagated_real_runtime_error.register_fake
        def _(t):
            return torch.empty_like(t)

        def fn(t):
            try:
                propagated_real_runtime_error(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.ones(2, 3)
        self.assertEqual(fn(t), t.sin())
        self.assertEqual(fn(-t), (-t).cos())

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(TorchRuntimeError, "real kernel rejected"):
                opt_fn(t)

    def test_fake_tensor_runtime_error_propagated_real_check_not_caught(self):
        @torch.library.custom_op(
            "test_dynamo::propagated_real_user_check", mutates_args=()
        )
        def propagated_real_user_check(t: torch.Tensor) -> torch.Tensor:
            if t.sum().item() > 0:
                return torch._refs.expand(t, 2)
            return t.clone()

        @propagated_real_user_check.register_fake
        def _(t):
            return t.clone()

        def fn(t):
            try:
                propagated_real_user_check(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.ones(2, 3)
        self.assertEqual(fn(t), t.sin())
        self.assertEqual(fn(-t), (-t).cos())

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(TorchRuntimeError, "too few dimensions"):
                opt_fn(t)

    def test_fake_tensor_runtime_error_without_try_except(self):
        def fn(t):
            t.expand_as(torch.randn(2))
            return t.cos()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_in_nonstrict_trace(self):
        @torch._dynamo.nonstrict_trace
        def inner(t, t0):
            return t.expand_as(t0)

        def fn(t):
            try:
                inner(t, torch.randn(2))
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(t), opt_fn(t))

    def test_fake_tensor_runtime_error_in_nonstrict_trace_without_handler(self):
        @torch._dynamo.nonstrict_trace
        def inner(t, t0):
            return t.expand_as(t0)

        def fn(t):
            return inner(t, torch.randn(2))

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_plain_in_nonstrict_trace_not_caught(self):
        @torch._dynamo.nonstrict_trace
        def inner(t):
            raise RuntimeError("fake-only failure")

        def direct(t):
            return inner(t)

        def with_handler(t):
            try:
                return inner(t)
            except RuntimeError:
                return t.sin()

        t = torch.randn(2, 3)
        for fn in (direct, with_handler):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(TorchRuntimeError, "fake-only failure"):
                opt_fn(t)

    def test_fake_tensor_runtime_error_bound_exception(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError as e:
                if "too few dimensions" in str(e):
                    return t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    @skipIfNotPy311
    def test_fake_tensor_runtime_error_sys_exception(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError:
                e = sys.exception()
                return str(e)
            return t

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_sys_exc_info_traceback(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError:
                return sys.exc_info()[2] is not None
            return False

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_args_inspection(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError as e:
                return e.args[0]
            return "ok"

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_f_string(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError as e:
                return f"{(e,)}"
            return "ok"

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_format(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError as e:
                return "{}".format((e,))  # noqa: UP032
            return "ok"

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_repr(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError as e:
                return repr(e)
            return "ok"

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_builtin_format(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError as e:
                return format([e])
            return "ok"

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_return_exception(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError as e:
                return e
            return None

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    @parametrize(
        "attr",
        ("__cause__", "__context__", "__suppress_context__", "__traceback__"),
    )
    def test_fake_tensor_runtime_error_state_attr(self, attr):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            except RuntimeError as e:
                getattr(e, attr)
                return t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_non_string_arg(self):
        format_calls = []

        class NonStringArgRuntimeError(RuntimeError):
            pass

        class Arg:
            def __str__(self):
                format_calls.append("__str__")
                return "1"

            def __repr__(self):
                format_calls.append("__repr__")
                return "Arg()"

        @torch.library.custom_op(
            "test_dynamo::runtime_error_non_string", mutates_args=()
        )
        def runtime_error_non_string(t: torch.Tensor) -> torch.Tensor:
            raise NonStringArgRuntimeError(Arg())

        @runtime_error_non_string.register_fake
        def _(t):
            # The fake RuntimeError args should not be stringified while tracing.
            raise NonStringArgRuntimeError(Arg())

        def fn(t):
            try:
                runtime_error_non_string(t)
            except ValueError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(t)
        self.assertEqual(format_calls, [])

    def test_fake_tensor_runtime_error_custom_format_not_read(self):
        format_calls = []

        class CustomFormatRuntimeError(RuntimeError):
            def __str__(self):
                format_calls.append("__str__")
                return "custom str"

            def __repr__(self):
                format_calls.append("__repr__")
                return "custom repr"

        @torch.library.custom_op(
            "test_dynamo::runtime_error_custom_format", mutates_args=()
        )
        def runtime_error_custom_format(t: torch.Tensor) -> torch.Tensor:
            return t.cos()

        @runtime_error_custom_format.register_fake
        def _(t):
            raise CustomFormatRuntimeError("fake-only failure")

        def fn(t):
            try:
                return runtime_error_custom_format(t)
            except RuntimeError:
                return t.sin()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(torch.randn(2, 3))
        self.assertEqual(format_calls, [])

    def test_fake_tensor_runtime_error_args_property_not_read(self):
        args_calls = []

        class ArgsPropertyRuntimeError(RuntimeError):
            @property
            def args(self):
                args_calls.append("args")
                raise RuntimeError("args property should not run")

        @torch.library.custom_op(
            "test_dynamo::runtime_error_args_property", mutates_args=()
        )
        def runtime_error_args_property(t: torch.Tensor) -> torch.Tensor:
            raise ArgsPropertyRuntimeError("real")

        @runtime_error_args_property.register_fake
        def _(t):
            raise ArgsPropertyRuntimeError("fake")

        def fn(t):
            try:
                runtime_error_args_property(t)
            except ValueError:
                return t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(torch.randn(2, 3))
        self.assertEqual(args_calls, [])

    def test_fake_tensor_runtime_error_metaclass_name_not_read(self):
        name_reads = []

        class NameReadingMeta(type):
            def __getattribute__(cls, name):
                if name == "__name__":
                    name_reads.append(name)
                return super().__getattribute__(name)

        class MetaclassNameRuntimeError(RuntimeError, metaclass=NameReadingMeta):
            pass

        @torch.library.custom_op(
            "test_dynamo::runtime_error_metaclass_name", mutates_args=()
        )
        def runtime_error_metaclass_name(t: torch.Tensor) -> torch.Tensor:
            raise MetaclassNameRuntimeError("real")

        @runtime_error_metaclass_name.register_fake
        def _(t):
            raise MetaclassNameRuntimeError("fake")

        def fn(t):
            try:
                runtime_error_metaclass_name(t)
            except RuntimeError:
                if name_reads:
                    return t + 2
                return t
            return t + 1

        t = torch.randn(2, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(t)
        self.assertEqual(name_reads, [])

    def test_fake_tensor_runtime_error_metaclass_name_descriptor_not_read(self):
        name_reads = []

        class NameDescriptor:
            def __get__(self, obj, typ=None):
                name_reads.append("__name__")
                raise AssertionError("__name__ descriptor should not run")

        class NameDescriptorMeta(type):
            __name__ = NameDescriptor()

        class MetaclassNameDescriptorRuntimeError(
            RuntimeError, metaclass=NameDescriptorMeta
        ):
            pass

        @torch.library.custom_op(
            "test_dynamo::runtime_error_metaclass_name_descriptor",
            mutates_args=(),
        )
        def runtime_error_metaclass_name_descriptor(t: torch.Tensor) -> torch.Tensor:
            raise MetaclassNameDescriptorRuntimeError("real")

        @runtime_error_metaclass_name_descriptor.register_fake
        def _(t):
            raise MetaclassNameDescriptorRuntimeError("fake")

        def fn(t):
            try:
                runtime_error_metaclass_name_descriptor(t)
            except RuntimeError:
                if name_reads:
                    return t + 2
                return t
            return t + 1

        t = torch.randn(2, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(t)
        self.assertEqual(name_reads, [])

    def test_fake_tensor_runtime_error_missing_op_profile_not_caught(self):
        from torch._library.fake_profile import MissingOpProfile

        @torch.library.custom_op("test_dynamo::missing_op_profile", mutates_args=())
        def missing_op_profile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        @missing_op_profile.register_fake
        def _(a, b):
            raise MissingOpProfile("missing fake profile")

        def fn(a, b):
            try:
                missing_op_profile(a, b)
            except RuntimeError:
                return a.sin()
            return a + b

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        self.assertEqual(fn(a, b), a + b)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(a, b)

    def test_fake_tensor_runtime_error_missing_fake_impl_not_caught(self):
        @torch.library.custom_op("test_dynamo::missing_fake_impl", mutates_args=())
        def missing_fake_impl(t: torch.Tensor) -> torch.Tensor:
            return t.cos()

        def fn(t):
            try:
                missing_fake_impl(t)
            except RuntimeError:
                return t.sin()
            return t.cos()

        t = torch.randn(2, 3)
        self.assertEqual(fn(t), t.cos())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "There was no fake impl registered"
        ):
            opt_fn(t)

    def test_fake_tensor_runtime_error_in_with_cleanup(self):
        class SwallowRuntimeError:
            def __enter__(self):
                return self

            def __exit__(self, typ, exc, tb):
                return typ is RuntimeError

        def fn(t):
            with SwallowRuntimeError():
                t.expand_as(torch.randn(2))
            return t.cos()

        backend = EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("cos", node_targets)

    def test_fake_tensor_runtime_error_with_exit_traceback_inspection(self):
        class InspectTraceback:
            def __enter__(self):
                return self

            def __exit__(self, typ, exc, tb):
                if tb is not None:
                    return tb.tb_lineno > 0
                return False

        def fn(t):
            with InspectTraceback():
                t.expand_as(torch.randn(2))
            return t.cos()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Fake RuntimeError inspection"):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_in_with_inside_try_except(self):
        backend = EagerAndRecordGraphs()

        class PassThroughRuntimeError:
            def __enter__(self):
                return self

            def __exit__(self, typ, exc, tb):
                return False

        def fn(t):
            try:
                with PassThroughRuntimeError():
                    t.expand_as(torch.randn(2))
            except RuntimeError:
                return t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("sin", node_targets)

    def test_fake_tensor_runtime_error_in_suppressing_with_inside_try_except(self):
        backend = EagerAndRecordGraphs()

        class SwallowRuntimeError:
            def __enter__(self):
                return self

            def __exit__(self, typ, exc, tb):
                return typ is RuntimeError

        def fn(t):
            try:
                with SwallowRuntimeError():
                    t.expand_as(torch.randn(2))
            except RuntimeError:
                return t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("cos", node_targets)
        self.assertNotIn("sin", node_targets)

    def test_fake_tensor_runtime_error_in_try_finally(self):
        def fn(t):
            try:
                t.expand_as(torch.randn(2))
            finally:
                t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            TorchRuntimeError, "RuntimeError when making fake tensor call"
        ):
            opt_fn(torch.randn(2, 3))

    def test_fake_tensor_runtime_error_in_try_finally_inside_try_except(self):
        backend = EagerAndRecordGraphs()

        def fn(t):
            try:
                try:
                    t.expand_as(torch.randn(2))
                finally:
                    t.cos()
            except RuntimeError:
                return t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("sin", node_targets)

    def test_fake_tensor_runtime_error_in_bare_except(self):
        backend = EagerAndRecordGraphs()

        def fn(t):
            t0 = torch.randn(2)
            try:
                t.expand_as(t0)
            except:  # noqa: E722
                return t.sin()
            return t.cos()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        t = torch.randn(2, 3)
        self.assertEqual(fn(t), opt_fn(t))

        self.assertEqual(len(backend.graphs), 1)
        node_targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertNotIn("expand_as", node_targets)
        self.assertIn("sin", node_targets)

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
            assert e.__cause__ is None  # noqa: S101
            assert e.__suppress_context__ is True  # noqa: S101

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
            assert isinstance(e.__cause__, ValueError)  # noqa: S101
            assert e.__suppress_context__ is True  # noqa: S101

    @make_dynamo_test
    def test_reraise_first_exc(self):
        def fn():
            try:
                raise ZeroDivisionError
            except ZeroDivisionError:
                try:
                    raise ValueError
                except ValueError:
                    pass
                raise

        try:
            fn()
        except ZeroDivisionError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

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
        assert sys.exc_info()[0] is None  # noqa: S101

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
                        assert isinstance(e.__context__, ZeroDivisionError)  # noqa: S101
                raise
        except ZeroDivisionError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    @make_dynamo_test
    def test_handle_all_exceptions(self):
        def cm():
            try:
                yield 1
            except ValueError:
                try:
                    raise TypeError
                finally:
                    pass

        try:
            gen = cm()
            next(gen)
            gen.throw(ValueError)
        except TypeError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    @make_dynamo_test
    def test_reraise(self):
        try:
            try:
                raise ValueError
            except ValueError:  # noqa: TRY203
                raise
        except ValueError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    @make_dynamo_test
    def test_raise_finally_simple(self):
        def fn():
            try:
                raise ValueError
            except ValueError:
                try:
                    raise TypeError
                finally:
                    pass

        try:
            fn()
        except TypeError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

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

    def test_reconstruct_AttributeError(self):
        sentinel = object()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return t.sin(), AttributeError("boom", name="myattr", obj=sentinel)

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, AttributeError)
        self.assertEqual(v.args, ("boom",))
        self.assertEqual(v.name, "myattr")
        self.assertIs(v.obj, sentinel)

    def test_reconstruct_AttributeError_from_getattr(self):
        class Foo:
            bar = 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            obj = Foo()
            try:
                obj.missing
            except AttributeError as e:
                return t.sin(), e

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, AttributeError)
        self.assertEqual(v.name, "missing")

    def test_reconstruct_NameError(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return t.sin(), NameError("boom", name="myvar")

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, NameError)
        self.assertEqual(v.args, ("boom",))
        self.assertEqual(v.name, "myvar")

    def test_reconstruct_NameError_from_undefined(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                undefined_name
            except NameError as e:
                return t.sin(), e

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, NameError)
        self.assertEqual(v.name, "undefined_name")

    def test_reconstruct_StopIteration(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return t.sin(), StopIteration(42)

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, StopIteration)
        self.assertEqual(v.args, (42,))
        self.assertEqual(v.value, 42)

    def test_raise_GeneratorExit(self):
        # GeneratorExit does not inherit from Exception
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise GeneratorExit
            except Exception:
                return t.sin()
            except BaseException:
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
            assert isinstance(e.__context__, ex)  # noqa: S101
            assert e.__cause__ is None  # noqa: S101
            assert e.__suppress_context__ is True  # noqa: S101

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
            assert e.args == (  # noqa: S101
                "exception cause must be None or derive from BaseException",
            )
        except Exception:
            raise AssertionError from None

        assert z == 1  # noqa: S101

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
            assert z == 2  # noqa: S101
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
            assert z == 2  # noqa: S101

        t = torch.randn(2)
        fn(t)

    @make_dynamo_test
    def test_raise_set___context__(self):
        try:
            raise TypeError
        except TypeError as e:
            exc = e

        assert exc.__context__ is None  # noqa: S101

        try:
            raise ValueError
        except ValueError as e:
            exc2 = e

        assert exc2.__context__ is None  # noqa: S101

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
            assert False  # noqa: B011, S101

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

    def test_str_repr_exception_no_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise ValueError
            except ValueError as e:
                return t.sin(), str(e), repr(e)

        t = torch.randn(2)
        y, s, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(s, "")
        self.assertEqual(r, "ValueError()")

    def test_str_repr_exception_single_arg(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise ValueError("test error")
            except ValueError as e:
                return t.sin(), str(e), repr(e)

        t = torch.randn(2)
        y, s, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(s, "test error")
        self.assertEqual(r, "ValueError('test error')")

    def test_str_user_defined_exception_custom_str(self):
        def fn(t):
            try:
                raise CustomRuntimeErrorWithStr("arg")
            except CustomRuntimeErrorWithStr as e:
                return t.sin(), str(e)

        t = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        y, s = opt_fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(s, "custom str")
        self.assertEqual(fn(t), opt_fn(t))

    def test_string_format_self_referential_list(self):
        def fn(t):
            items = []
            items.append(items)
            return f"{items}", t.sin()

        t = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(t), opt_fn(t))

    def test_str_repr_exception_multi_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise ValueError("hello", 42)
            except ValueError as e:
                return t.sin(), str(e), repr(e)

        t = torch.randn(2)
        y, s, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(s, str(("hello", 42)))
        self.assertEqual(r, "ValueError('hello', 42)")

    def test_frozen_dataclass_setattr_raises(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: int

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            dc = TestDataClass(1)
            try:
                dc.x = 2
            except dataclasses.FrozenInstanceError:
                return t + 1
            except Exception:
                return t + 2
            return t + dc.x

        self.assertEqual(fn(torch.zeros(1)), 1)

    def test_exception_traceback_access(self):
        # Test that __traceback__ is accessible after raising/catching an exception
        def fn(x):
            try:
                raise ValueError("oops")
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None:
                    x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_traceback_tb_next(self):
        # Test that tb_next can be accessed on a traceback
        def fn(x):
            try:
                raise ValueError("oops")
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None:
                    # tb_next is None for a single-frame traceback
                    if tb.tb_next is None:
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_traceback_tb_lineno(self):
        # Test that tb_lineno is accessible on a traceback
        def fn(x):
            try:
                raise ValueError("oops")
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None and tb.tb_lineno > 0:
                    x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_with_traceback_method(self):
        # Test the with_traceback() method
        def fn(x):
            try:
                raise ValueError("first")
            except ValueError as e:
                tb = e.__traceback__
                try:
                    raise RuntimeError("second").with_traceback(tb) from None
                except RuntimeError as e2:
                    if e2.__traceback__ is not None:
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_set_traceback(self):
        # Test assigning __traceback__ on an exception
        def fn(x):
            try:
                raise ValueError("first")
            except ValueError as e:
                tb = e.__traceback__
                try:
                    raise RuntimeError("second") from None
                except RuntimeError as e2:
                    e2.__traceback__ = tb
                    if e2.__traceback__ is not None:
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_set_traceback_none(self):
        # Test assigning None to __traceback__
        def fn(x):
            try:
                raise ValueError("oops")
            except ValueError as e:
                e.__traceback__ = None
                if e.__traceback__ is None:
                    x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_traceback_tb_lasti_graph_break(self):
        # Accessing tb_lasti should cause a graph break
        def fn(x):
            try:
                raise ValueError("oops")
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None:
                    _ = tb.tb_lasti
                    x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        # Should graph break but still produce correct results
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_set_tb_next(self):
        # Test setting tb_next on a traceback
        def fn(x):
            try:
                raise ValueError("first")
            except ValueError as e:
                tb1 = e.__traceback__
                try:
                    raise RuntimeError("second") from None
                except RuntimeError as e2:
                    tb2 = e2.__traceback__
                    if tb2 is not None and tb1 is not None:
                        tb2.tb_next = None
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_traceback_chained(self):
        # Test traceback chaining through multiple frames
        def inner():
            raise ValueError("inner")

        def fn(x):
            try:
                inner()
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None:
                    x = x + 1
                    # Walk the traceback chain
                    depth = 0
                    curr = tb
                    while curr is not None:
                        depth += 1
                        curr = curr.tb_next
                    if depth > 0:
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @parametrize(
        "exc_type_1,exc_type_2",
        [
            (ValueError, TypeError),
            (CustomException, ValueError),
        ],
        name_fn=lambda exc1, exc2: f"{exc1.__name__}_to_{exc2.__name__}",
    )
    def test_exception_set_context(self, exc_type_1, exc_type_2):
        # Test explicitly assigning to __context__ attribute (reaches ExceptionVariable.__context__ assignment)
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            exc1 = exc_type_1("first")
            exc2 = exc_type_2("second")

            # This explicitly sets __context__ via call_setattr
            exc2.__context__ = exc1

            # Verify it was set correctly
            if exc2.__context__ is exc1:
                return t.sin()
            else:
                return t.cos()

        t = torch.randn(2)
        ref_result = t.sin()
        result = fn(t)
        self.assertEqual(result, ref_result)

    @parametrize(
        "exc_type_1,exc_type_2,exc_type_3",
        [
            (ValueError, TypeError, RuntimeError),
            (CustomException, ValueError, TypeError),
        ],
        name_fn=lambda exc1, exc2, exc3: (
            f"{exc1.__name__}_chain_{exc2.__name__}_{exc3.__name__}"
        ),
    )
    def test_exception_context_chain(self, exc_type_1, exc_type_2, exc_type_3):
        # Test chaining contexts through multiple exceptions
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            exc1 = exc_type_1("first")
            exc2 = exc_type_2("second")
            exc3 = exc_type_3("third")

            exc2.__context__ = exc1
            exc3.__context__ = exc2

            # Verify the chain
            if isinstance(exc3.__context__, exc_type_2) and isinstance(
                exc3.__context__.__context__, exc_type_1
            ):
                return t.sin()
            else:
                return t.cos()

        t = torch.randn(2)
        ref_result = t.sin()
        result = fn(t)
        self.assertEqual(result, ref_result)

    @make_dynamo_test
    def test_exception_custom_attribute(self):
        e = RuntimeError("boom")
        e.foo = 42
        assert e.foo == 42  # noqa: S101

    @make_dynamo_test
    def test_exception_set_args_from_iterable(self):
        e = RuntimeError("boom")
        e.args = [1, 2, 3]
        assert e.args == (1, 2, 3)  # noqa: S101

    @make_dynamo_test
    def test_exception_set_args_not_iterable(self):
        e = RuntimeError("boom")
        try:
            e.args = 2
        except TypeError as exc:
            assert "object is not iterable" in str(exc)  # noqa: S101
        else:
            raise AssertionError

    @make_dynamo_test
    def test_exception_setstate_dict(self):
        e = RuntimeError("boom")
        e.__setstate__({"foo": 7})
        assert e.foo == 7  # noqa: S101

    @make_dynamo_test
    def test_exception_setstate_none_noop(self):
        e = RuntimeError("boom")
        assert e.__setstate__(None) is None  # noqa: S101

    @make_dynamo_test
    def test_exception_setstate_not_dict(self):
        e = RuntimeError("boom")
        try:
            e.__setstate__(2)
        except TypeError as exc:
            assert "state is not a dictionary" in str(exc)  # noqa: S101
        else:
            raise AssertionError

    def test_exception_custom_attribute_side_effect_replayed(self):
        # The exception escapes the compiled region, so the custom attributes
        # set during tracing must be replayed onto the real object handed back
        # to eager (exercises the side-effects codegen, not just tracing).
        def fn(x):
            e = RuntimeError("boom")
            e.foo = 42
            e.__setstate__({"bar": 7})
            return e, x + 1

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        e, y = opt_fn(x)
        self.assertIsInstance(e, RuntimeError)
        self.assertEqual(e.foo, 42)
        self.assertEqual(e.bar, 7)
        self.assertEqual(y, x + 1)

    def test_exception_setstate_non_string_key_diverges_from_eager(self):
        # Eager's BaseException.__setattr__ requires string names, so
        # __setstate__ with a non-string key raises TypeError. Dynamo currently
        # stores non-string constant keys in the side-effect dict without error
        # -- a known divergence from eager, to be fixed alongside tp_setattro.
        def fn(x):
            e = RuntimeError("boom")
            e.__setstate__({1: 2})
            return x + 1

        x = torch.randn(4)
        with self.assertRaisesRegex(TypeError, "attribute name must be string"):
            fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        opt_fn(x)  # diverges: Dynamo does not raise


instantiate_parametrized_tests(ExceptionTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
