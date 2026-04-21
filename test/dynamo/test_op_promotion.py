# Owner(s): ["module: dynamo"]
# flake8: noqa: B001,B006,B020,B021,B950,C405,C416,E711,E721,E722,E731,F401,F403,F405,F541,F821,F823
# ruff: noqa: F403,F405,F841
try:
    from .test_misc import *
except ImportError:
    from test_misc import *


class DynamoOpPromotionTests(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_tensor_mul(self):
        def symbool_mul_fn(x_bool, sentinel):
            result = x_bool * sentinel
            return result

        x_true = torch.tensor([True], device="cuda")
        x_false = torch.tensor([False], device="cuda")
        sentinel = torch.tensor(2.0, requires_grad=True, device="cuda")
        eager_result_true = symbool_mul_fn(x_true, sentinel)
        eager_result_false = symbool_mul_fn(x_false, sentinel)
        compiled_fn = torch.compile(
            symbool_mul_fn, fullgraph=True, dynamic=True, backend="eager"
        )
        compiled_result_true = compiled_fn(x_true, sentinel)
        compiled_result_false = compiled_fn(x_false, sentinel)
        self.assertEqual(eager_result_true, compiled_result_true)
        self.assertEqual(eager_result_false, compiled_result_false)
        self.assertEqual(compiled_result_true.item(), 2.0)
        self.assertEqual(compiled_result_false.item(), 0.0)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_guard_or_false(self):
        def symbool_guard_fn(a_bool_tensor, b):
            u0 = a_bool_tensor.item()
            # Make sure guard_or_false still handles SymBool produced by .item()
            if guard_or_false(u0):
                return b * 10
            else:
                return b * 100

        compiled_guard_fn = torch.compile(
            symbool_guard_fn, backend="eager", dynamic=True
        )
        a_true = torch.tensor(True, device="cuda")
        a_false = torch.tensor(False, device="cuda")
        b = torch.randn(6, device="cuda")
        eager_res_true = symbool_guard_fn(a_true, b)
        compiled_res_true = compiled_guard_fn(a_true, b)
        self.assertEqual(eager_res_true, compiled_res_true)
        eager_res_false = symbool_guard_fn(a_false, b)
        compiled_res_false = compiled_guard_fn(a_false, b)
        self.assertEqual(eager_res_false, compiled_res_false)
        self.assertEqual(compiled_res_true, b * 10)
        self.assertEqual(compiled_res_false, b * 100)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_tensor_mul_does_not_fail(self):
        def fuzzed_program(arg_0, sentinel):
            var_node_2 = arg_0
            var_node_1 = torch.squeeze(var_node_2)
            var_node_0 = var_node_1.item()
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        sentinel = torch.tensor(1.0, requires_grad=True, device="cuda")
        arg_0 = torch.tensor([True], dtype=torch.bool, device="cuda")
        args = (arg_0,) + (sentinel,)
        try:
            compiled_program = torch.compile(
                fuzzed_program, fullgraph=True, dynamic=True, backend="eager"
            )
            compiled_program(*args)
        except Exception as e:
            self.fail(f"torch.compile failed with error: {e}")

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorify_track_item_symint(self):
        def _random_resize(image: torch.Tensor):
            image_metanet = image
            default_patch_size = 14
            rand_cnn_resolution = (224, 256)
            min_nump = rand_cnn_resolution[0] // default_patch_size
            max_nump = rand_cnn_resolution[1] // default_patch_size
            new_nump = torch.randint(min_nump, max_nump + 1, (1,)).item()
            torch._check(new_nump > 0)
            torch._check(new_nump * default_patch_size > 1)

            image_metanet = F.interpolate(
                image_metanet,
                size=(new_nump * default_patch_size, new_nump * default_patch_size),
                mode="bilinear",
                align_corners=True,
            )
            img_h_new, img_w_new = image_metanet.shape[2:]

            return (img_h_new, img_w_new), image_metanet

        _random_resize_compiled = torch.compile(fullgraph=True, backend="eager")(
            _random_resize
        )

        # Test the function
        input_tensor = torch.rand(1, 3, 224, 224)
        (h, w), output = _random_resize_compiled(input_tensor)

        # Verify output properties
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 3)
        self.assertEqual(output.shape[2], h)
        self.assertEqual(output.shape[3], w)
        self.assertTrue(h % 14 == 0)
        self.assertTrue(w % 14 == 0)
        self.assertTrue(224 <= h <= 256)
        self.assertTrue(224 <= w <= 256)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_module_to_with_shared_weights_compile(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)

            def forward(self, x):
                token_ids = torch.randint(0, 10, (4,), device=x.device)
                embedded = self.embedding(token_ids).sum()
                return x.sum() + embedded.sum()

        class Container(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = Model()

            def forward(self, x):
                if "cuda" in str(x.device):
                    mod = self.mod.to(x.device)
                    return mod(x)
                else:
                    return x.sum()

        container = Container()
        container_eager = copy.deepcopy(container)
        with torch._dynamo.config.patch(graph_break_on_nn_param_ctor=False):
            compiled = torch.compile(container, backend="eager", fullgraph=True)

            inp1 = torch.randn(4, 4, 4, device="cuda")

            # First call with CUDA input
            compiled_result1 = compiled(inp1)
            eager_result1 = container_eager(inp1)
            same(compiled_result1, eager_result1)

            # Second call - weights are now on CUDA from first call
            # This tests that .to(cuda) on already-cuda weights doesn't fail
            compiled_result2 = compiled(inp1)
            eager_result2 = container_eager(inp1)
            same(compiled_result2, eager_result2)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_module_to_move_compile(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 10)

            def forward(self, x):
                x = self.fc(x)
                self.to("cpu")
                return x

        mod = Model().cuda()
        with torch._dynamo.config.patch(graph_break_on_nn_param_ctor=False):
            fn = torch.compile(mod, backend="aot_eager", fullgraph=True)
            x = torch.randn(10, 10, device="cuda")
            ref = fn(x)
            self.assertEqual(str(mod.fc.weight.device), "cpu")
            mod.cuda()
            ref = fn(
                x
            )  # second time compile runs, we should also move the module to cpu device
            self.assertEqual(str(mod.fc.weight.device), "cpu")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
