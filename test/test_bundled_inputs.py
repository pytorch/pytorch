#!/usr/bin/env python3
import io
import torch
import torch.utils.bundled_inputs
from torch.testing._internal.common_utils import TestCase, run_tests


def model_size(sm):
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    return len(buffer.getvalue())


def save_and_load(sm):
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)


class TestBundledInputs(TestCase):

    def test_single_tensors(self):
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg

        sm = torch.jit.script(SingleTensorModel())
        original_size = model_size(sm)
        get_expr = []
        samples = [
            # Tensor with small numel and small storage.
            (torch.tensor([1]),),
            # Tensor with large numel and small storage.
            (torch.tensor([[2, 3, 4]]).expand(1 << 16, -1)[:, ::2],),
            # Tensor with small numel and large storage.
            (torch.tensor(range(1 << 16))[-8:],),
            # Large zero tensor.
            (torch.zeros(1 << 16),),
            # Large channels-last ones tensor.
            (torch.ones(4, 8, 32, 32).contiguous(memory_format=torch.channels_last),),
            # Special encoding of random tensor.
            (torch.utils.bundled_inputs.bundle_randn(1 << 16),),
        ]
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            sm, samples, get_expr)
        # print(get_expr[0])
        # print(sm._generate_bundled_inputs.code)

        # Make sure the model only grew a little bit,
        # despite having nominally large bundled inputs.
        augmented_size = model_size(sm)
        self.assertLess(augmented_size, original_size + (1 << 12))

        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        self.assertEqual(loaded.get_num_bundled_inputs(), len(samples))
        self.assertEqual(len(inflated), len(samples))
        self.assertTrue(loaded.run_on_bundled_input(0) is inflated[0][0])

        for idx, inp in enumerate(inflated):
            self.assertIsInstance(inp, tuple)
            self.assertEqual(len(inp), 1)
            self.assertIsInstance(inp[0], torch.Tensor)
            if idx != 5:
                # Strides might be important for benchmarking.
                self.assertEqual(inp[0].stride(), samples[idx][0].stride())
                self.assertEqual(inp[0], samples[idx][0], exact_dtype=True)

        # This tensor is random, but with 100,000 trials,
        # mean and std had ranges of (-0.0154, 0.0144) and (0.9907, 1.0105).
        self.assertEqual(inflated[5][0].shape, (1 << 16,))
        self.assertEqual(inflated[5][0].mean().item(), 0, atol=0.025, rtol=0)
        self.assertEqual(inflated[5][0].std().item(), 1, atol=0.02, rtol=0)


    def test_large_tensor_with_inflation(self):
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg
        sm = torch.jit.script(SingleTensorModel())
        sample_tensor = torch.randn(1 << 16)
        # We can store tensors with custom inflation functions regardless
        # of size, even if inflation is just the identity.
        sample = torch.utils.bundled_inputs.bundle_large_tensor(sample_tensor)
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            sm, [(sample,)])

        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        self.assertEqual(len(inflated), 1)

        self.assertEqual(inflated[0][0], sample_tensor)


    def test_rejected_tensors(self):
        def check_tensor(sample):
            # Need to define the class in this scope to get a fresh type for each run.
            class SingleTensorModel(torch.nn.Module):
                def forward(self, arg):
                    return arg
            sm = torch.jit.script(SingleTensorModel())
            with self.assertRaisesRegex(Exception, "Bundled input argument"):
                torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                    sm, [(sample,)])

        # Plain old big tensor.
        check_tensor(torch.randn(1 << 16))
        # This tensor has two elements, but they're far apart in memory.
        # We currently cannot represent this compactly while preserving
        # the strides.
        small_sparse = torch.randn(2, 1 << 16)[:, 0:1]
        self.assertEqual(small_sparse.numel(), 2)
        check_tensor(small_sparse)


    def test_non_tensors(self):
        class StringAndIntModel(torch.nn.Module):
            def forward(self, fmt: str, num: int):
                return fmt.format(num)

        sm = torch.jit.script(StringAndIntModel())
        samples = [
            ("first {}", 1),
            ("second {}", 2),
        ]
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            sm, samples)

        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        self.assertEqual(inflated, samples)
        self.assertTrue(loaded.run_on_bundled_input(0) == "first 1")


if __name__ == '__main__':
    run_tests()
