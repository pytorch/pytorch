# Owner(s): ["module: dynamo"]
import functools
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same

try:
    from . import test_modules, test_repros
except ImportError:
    import test_modules
    import test_repros


def make_unspec_fn(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with patch.object(torch._dynamo.config, "specialize_int_float", False):
            return fn(*args, **kwargs)

    return _fn


def make_unspec_cls(cls):
    class UnspecTest(cls):
        pass

    UnspecTest.__name__ = f"Unspec{cls.__name__}"

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                continue
            new_name = f"{name}_unspec"
            fn = make_unspec_fn(fn)
            fn.__name__ = new_name
            setattr(UnspecTest, name, None)
            setattr(UnspecTest, new_name, fn)

    return UnspecTest


UnspecReproTests = make_unspec_cls(test_repros.ReproTests)
UnspecNNModuleTests = make_unspec_cls(test_modules.NNModuleTests)


@patch.object(torch._dynamo.config, "specialize_int_float", False)
class UnspecTests(torch._dynamo.test_case.TestCase):
    def test_numpy_correctness(self):
        def fn(x, y, z):
            xy = [x + y, y, False]
            np_x = x.numpy()
            np_y = y.numpy()
            return {
                "x": x,
                "z": z,
                "a": np_y.sum(),
                "b": xy,
                "c": np_y[0][0] / 68,
                "d": np_x.sum(),
            }, x + np_y.sum() + z

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        y = torch.ones([2, 2], dtype=torch.int64)
        z = np.int64(12)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res2 = opt_fn(x, y, z)
        self.assertTrue(same(res1, res2))

    def test_no_recompilations(self):
        # no recompilations if passing on different numpy int values
        def fn(x, y):
            return {"a": x + 1, "b": y / 2}

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        for i in range(10):
            opt_fn(x, np.int64(i))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_builtin_max_min(self):
        # test unspecialized primitive max/min
        def fn(x, y, z):
            return z + 1, max(x, y), min(x - 4, y)

        x = np.int64(12)
        y = 10
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res2 = opt_fn(x, y, z)
        self.assertTrue(same(res1, res2))

    def test_feed_random_values_into_graph_only(self):
        def fn(shape):
            torch.manual_seed(123)
            x = torch.randn(shape, device="cpu") * random.randint(30, 100)
            return x

        shape = [2, 3]
        random.seed(1)
        res1 = fn(shape)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        random.seed(1)
        res2 = opt_fn(shape)

        self.assertTrue(same(res1, res2))

    def test_random_values_with_graph_break(self):
        def fn(x):
            r1 = random.random()
            y = x + random.uniform(10, 20)
            y.sum().item()
            r2 = random.randint(2, 18)  # no graph output in this frame
            y.sum().item()
            return y + r1, r2

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
    def test_multiple_consecutive_random_calls_before_graph(self):
        def fn(x):
            dim1 = random.randrange(start=0, stop=5)
            dim2 = random.randrange(start=0, stop=5)
            dim3 = random.randrange(start=0, stop=5)
            y = torch.rand(dim1, dim2, dim3)
            return x + 2, y

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_random_call_with_while_loop(self):
        def fn(x):
            dim1 = random.randrange(start=0, stop=3)
            dim2 = dim1
            while dim1 == dim2:
                dim2 = random.randrange(start=0, stop=3)
            return x * 2

        x = torch.randn(4)
        random.seed(1)
        res1 = fn(x)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    # TypeError: zeros(): argument 'size' (position 1) must be tuple of SymInts, not FakeTensor
    @unittest.expectedFailure
    def test_builtin_getitem(self):
        # builtin getitem args[0] is python list and args[1] is unspec
        def fn(x, idx):
            return (torch.zeros(idx), x[idx], x[idx:])

        x = list(range(50))
        ref = fn(x, 48)  # 48 is unspecialized
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x, 48)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_builtin_functions_on_cuda(self):
        def fn(x, scaler):
            m = torch.nn.ReLU()
            y = m(x) * scaler
            return y

        x = torch.randn([3, 6], device="cuda")
        scaler = 0.23  # 0.23 is unspecialized
        ref = fn(x, scaler)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x, scaler)
        self.assertTrue(same(ref, res))
        self.assertEqual(ref.device, res.device)

    def test_unspec_float_precision(self):
        def fn(image, scale_factor):
            image = torch.nn.functional.interpolate(
                image[None],
                size=None,
                scale_factor=scale_factor,
                mode="bilinear",
                recompute_scale_factor=True,
                align_corners=False,
            )[0]

            return image.shape

        x = torch.rand([3, 427, 640])
        scale_factor = 1.873536229133606
        ref = fn(x, scale_factor)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x, scale_factor)
        self.assertTrue(same(ref, res))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
