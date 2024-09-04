# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint


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
            except (NotImplementedError, AttributeError) as e:
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
            except NotImplementedError as e:
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

    # TODO(anijain2305) - does not work with fullgraph=True
    def test_exception_with_another_exception2(self):
        def gn(x):
            try:
                x = torch.cos(x)
                raise NotImplementedError("Not implemented")
            except NotImplementedError as e:
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
        ref = fn(x)
        # Cant use fullgraph=True because RERAISE is not supported
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)

    # TODO(anijain2305) - does not work with fullgraph=True
    def test_exception_with_ctx_manager(self):
        def fn(x):
            x = torch.cos(x)
            try:
                with torch.no_grad():
                    x = torch.sin(x)
                    raise NotImplementedError("Not implemented")
            except NotImplementedError as e:
                x = torch.sigmoid(x)
            return x

        x = torch.randn(4)
        ref = fn(x)
        # Cant use fullgraph=True because WITH_EXCEPT_START is not supported
        opt_fn = torch.compile(fn, backend="eager")
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
