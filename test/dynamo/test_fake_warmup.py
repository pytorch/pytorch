"""
Tests for torch.compiler.fake_warmup() API.

fake_warmup allows pre-warming the compilation cache by running with FakeTensors,
so subsequent runs with real tensors hit the cache.
"""
import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter


class TestFakeWarmup(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def test_fake_warmup_with_program(self):
        """Test fake_warmup with a realistic program containing multiple compiled functions."""
        counter1 = CompileCounter()
        counter2 = CompileCounter()

        @torch.compile(backend=counter1)
        def fn1(x):
            if x.size()[0] > 10:
                return x.sum(dim=0) + x.mean()
            else:
                return x * 100

        @torch.compile(backend=counter2)
        def fn2(x, y):
            return x + y * 100

        def program(x):
            # eager code
            fake_x = x + 100
            tmp1 = fn1(fake_x)
            # more eager
            tmp2 = tmp1.contiguous()
            tmp3 = fn2(tmp1, tmp2)
            return tmp3

        # One warmup with mark_dynamic
        x_warmup = torch.randn(32, 10)
        torch._dynamo.mark_dynamic(x_warmup, 0)
        torch.compiler.fake_warmup(program, args=(x_warmup,))

        # After warmup: 1 compilation each
        self.assertEqual(counter1.frame_count, 1)
        self.assertEqual(counter2.frame_count, 1)

        # Run with real tensors of different shapes - should hit cache
        for shape in [(32, 10), (64, 10), (128, 10)]:
            x = torch.randn(*shape)
            compiled_result = program(x)
            with torch.compiler.set_stance("force_eager"):
                eager_result = program(x)
            self.assertTrue(torch.allclose(compiled_result, eager_result, atol=1e-5))
            self.assertEqual(counter1.frame_count, 1)
            self.assertEqual(counter2.frame_count, 1)


    def test_fake_warmup_with_program_dde(self):
        """Test that fake_warmup raises a clear error for DDE in eager regions."""
        counter1 = CompileCounter()
        counter2 = CompileCounter()

        @torch.compile(backend=counter1)
        def fn1(x):
            if x.size()[0] > 10:
                return x.sum(dim=0) + x.mean()
            else:
                return x * 100

        @torch.compile(backend=counter2)
        def fn2(x, y):
            return x + y * 100

        def program(x):
            # eager code
            fake_x = x + 100
            tmp1 = fn1(fake_x)
            # more eager
            tmp2 = tmp1.contiguous()
            # This causes DDE in eager region - branching on tensor value
            d = tmp2[0]
            if d > 10:
                return torch.tensor(10)
            tmp3 = fn2(tmp1, tmp2)
            return tmp3

        # Should raise RuntimeError about DDE in eager regions
        x_warmup = torch.randn(32, 10)
        torch._dynamo.mark_dynamic(x_warmup, 0)
        with self.assertRaisesRegex(
            RuntimeError,
            "fake_warmup: branching on data-dependent operations results"
        ):
            torch.compiler.fake_warmup(program, args=(x_warmup,))

    def test_fake_warmup_with_program_fake_not_supported(self):
        """Test that fake_warmup raises an error when an op doesn't have a fake impl."""
        # Create a custom op without a fake implementation
        @torch.library.custom_op("test_fake_warmup::no_fake_impl", mutates_args=())
        def no_fake_impl(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        def program(x):
            # Call the custom op that has no fake implementation
            return no_fake_impl(x)

        x_warmup = torch.randn(32, 10)
        # Should raise an error about missing fake implementation
        with self.assertRaisesRegex(
            RuntimeError,
            "fake_warmup requires all ops to have a FakeTensor"
        ):
            torch.compiler.fake_warmup(program, args=(x_warmup,))
    
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
