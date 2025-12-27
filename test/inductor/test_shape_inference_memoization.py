# Owner(s): ["module: inductor"]

import sympy

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestShapeInferenceMemoization(TestCase):
    def test_permute_view_get_size_memoization(self):
        def model(x):
            x = x.view(16, 64)
            x = x.transpose(0, 1)
            x = x.reshape(32, 32)
            return x

        torch._dynamo.reset()
        result = torch.compile(model)(torch.randn(1024))

        assert result.shape == (32, 32)

    def test_view_chain_shape_inference(self):
        def model(x):
            x = x.view(8, 8, 16)
            x = x.permute(2, 0, 1)
            x = x.reshape(16, 64)
            x = x.transpose(0, 1)
            return x

        torch._dynamo.reset()
        result = torch.compile(model)(torch.randn(1024))

        self.assertEqual(result.shape, model(torch.randn(1024)).shape)

    def test_dtype_view_shape_inference(self):
        def model(x):
            x = x.view(32, 32)
            x = x.to(torch.float16)
            x = x.transpose(0, 1)
            return x

        torch._dynamo.reset()
        result = torch.compile(model)(torch.randn(1024))

        self.assertEqual(result.shape, (32, 32))
        self.assertEqual(result.dtype, torch.float16)

    def test_repeated_get_size_calls(self):
        def model(x):
            x = x.reshape(4, 4, 64)
            x = x.permute(2, 0, 1)
            x = x.contiguous()
            x = x.view(256, 4)
            return x

        torch._dynamo.reset()
        inp = torch.randn(1024)
        result = torch.compile(model)(inp)

        self.assertEqual(result.shape, model(inp).shape)
        torch.testing.assert_close(result, model(inp))


class TestPermuteViewMemoization(TestCase):
    def test_permute_view_caches_get_size(self):
        from torch._inductor.ir import PermuteView, Pointwise

        with torch._subclasses.FakeTensorMode() as fake_mode:
            with torch._inductor.virtualized.V.set_graph_handler(
                MockGraphHandler(fake_mode.shape_env)
            ):
                pointwise = Pointwise(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    inner_fn=lambda idx: 0,
                    ranges=[sympy.Integer(16), sympy.Integer(64)],
                )

                permute = PermuteView(data=pointwise, dims=[1, 0])

                size1 = permute.get_size()
                size2 = permute.get_size()

                self.assertEqual(list(size1), list(size2))
                self.assertEqual(list(size1), [sympy.Integer(64), sympy.Integer(16)])
                self.assertTrue(hasattr(permute, "__get_size_cache"))


class TestReinterpretViewMemoization(TestCase):
    def test_reinterpret_view_caches_size_and_stride(self):
        from torch._inductor.ir import FixedLayout, InputBuffer, ReinterpretView

        with torch._subclasses.FakeTensorMode() as fake_mode:
            with torch._inductor.virtualized.V.set_graph_handler(
                MockGraphHandler(fake_mode.shape_env)
            ):
                layout = FixedLayout(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    size=[sympy.Integer(16), sympy.Integer(64)],
                    stride=[sympy.Integer(64), sympy.Integer(1)],
                )

                buffer = InputBuffer(name="test_buffer", layout=layout)

                new_layout = FixedLayout(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    size=[sympy.Integer(64), sympy.Integer(16)],
                    stride=[sympy.Integer(16), sympy.Integer(1)],
                )
                view = ReinterpretView(data=buffer, layout=new_layout)

                size1 = view.get_size()
                size2 = view.get_size()
                self.assertEqual(list(size1), list(size2))
                self.assertTrue(hasattr(view, "__get_size_cache"))

                stride1 = view.get_stride()
                stride2 = view.get_stride()
                self.assertEqual(list(stride1), list(stride2))
                self.assertTrue(hasattr(view, "__get_stride_cache"))


class MockGraphHandler:
    def __init__(self, shape_env):
        self.sizevars = MockSizeVars(shape_env)
        self.name_to_buffer = {}
        self.constants = {}

    def register_buffer(self, buffer):
        name = f"buf{len(self.name_to_buffer)}"
        self.name_to_buffer[name] = buffer
        return name


class MockSizeVars:
    def __init__(self, shape_env):
        self.shape_env = shape_env

    def statically_known_geq(self, a, b):
        return False

    def is_size_one_or_false(self, s):
        return s == 1 or s == sympy.Integer(1)


if __name__ == "__main__":
    run_tests()
