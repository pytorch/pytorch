# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case


@torch._dynamo.config.patch("capture_scalar_outputs", True)
class ViewTests(torch._dynamo.test_case.TestCase):
    def test_view_to_2d(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(t, _u0):
            u0 = t[0].item()
            u1 = t[1].item()
            torch._check_is_size(u0)
            torch._check_is_size(u1)
            n = u0 * u1
            a = torch.randn(n)
            return a.view(-1, _u0)

        t = torch.tensor([2, 4], dtype=torch.int32)
        f(t, 2)

    def test_view_to_1d(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(t, _n):
            u0 = t[0].item()
            u1 = t[1].item()
            torch._check_is_size(u0)
            torch._check_is_size(u1)
            a = torch.randn(u0, u1)
            return a.view(_n)

        t = torch.tensor([2, 4], dtype=torch.int32)
        f(t, 8)

    def test_view_with_tensor_shape_params(self):
        # Test for issue #156720: aten.view.default with tensor shape parameters
        class TestModel(torch.nn.Module):
            def forward(self, x, shape_params):
                return torch.ops.aten.view.default(x, shape_params)

        x = torch.randn(24)
        shape_params = [
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        ]

        model = TestModel()
        expected = model(x, shape_params)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, shape_params)

        torch.testing.assert_close(result, expected)

    def test_tensor_view_with_tensor_shape_params(self):
        # Test tensor.view() method with tensor shape parameters (list version)
        class TestModel(torch.nn.Module):
            def forward(self, x, shape_params):
                return x.view(shape_params)

        x = torch.randn(24)
        shape_params = (
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        )

        model = TestModel()
        expected = model(x, shape_params)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, shape_params)

        torch.testing.assert_close(result, expected)

    def test_tensor_view_with_tensor_args(self):
        # Test tensor.view() method with individual tensor arguments
        class TestModel(torch.nn.Module):
            def forward(self, x, dim1, dim2, dim3):
                return x.view(dim1, dim2, dim3)

        x = torch.randn(24)
        dim1 = torch.tensor(2, dtype=torch.int32)
        dim2 = torch.tensor(3, dtype=torch.int32)
        dim3 = torch.tensor(4, dtype=torch.int32)

        model = TestModel()
        expected = model(x, dim1, dim2, dim3)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, dim1, dim2, dim3)

        torch.testing.assert_close(result, expected)

    def test_torch_reshape_with_tensor_shape_params(self):
        # Test torch.reshape() function with tensor shape parameters
        def test_fn(x, shape_params):
            return torch.reshape(x, shape_params)

        x = torch.randn(24)
        shape_params = [
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        ]

        expected = test_fn(x, shape_params)

        compiled_fn = torch.compile(test_fn, backend="eager")
        result = compiled_fn(x, shape_params)

        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
