# Owner(s): ["module: nn"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_utils import (
    gradcheck,
    gradgradcheck,
    run_tests,
    TestCase,
)


class TestInfoNCELoss(TestCase):
    """Tests for nn.InfoNCELoss / F.info_nce_loss."""

    def _reference(self, query, positive_key, negative_keys, temperature):
        """Independent reference: Oord et al. Eq. (4), per-sample."""
        query = F.normalize(query, dim=1)
        positive_key = F.normalize(positive_key, dim=1)
        if negative_keys is None:
            logits = query @ positive_key.t() / temperature
            targets = torch.arange(query.shape[0], device=query.device)
        else:
            negative_keys = F.normalize(negative_keys, dim=1)
            positive_sim = (query * positive_key).sum(1, keepdim=True)
            logits = torch.cat([positive_sim, query @ negative_keys.t()], 1)
            logits = logits / temperature
            targets = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        return F.cross_entropy(logits, targets, reduction="none")

    # -- basic ------------------------------------------------------------
    def test_forward_functional_and_module_agree(self):
        query = torch.randn(32, 128)
        positive_key = torch.randn(32, 128)
        expected = F.info_nce_loss(query, positive_key, temperature=0.07)
        self.assertEqual(expected.shape, torch.Size([]))
        self.assertGreaterEqual(expected.item(), 0)
        self.assertEqual(
            nn.InfoNCELoss(temperature=0.07)(query, positive_key), expected
        )

    def test_forward_with_explicit_negatives(self):
        query = torch.randn(16, 64)
        positive_key = torch.randn(16, 64)
        negative_keys = torch.randn(128, 64)
        loss = F.info_nce_loss(query, positive_key, negative_keys)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0)
        self.assertEqual(nn.InfoNCELoss()(query, positive_key, negative_keys), loss)

    # -- correctness against the reference --------------------------------
    def test_matches_reference_in_batch(self):
        for temperature in (0.01, 0.07, 0.5, 1.0, 10.0):
            for shape in ((4, 8), (16, 64), (33, 7)):
                with self.subTest(temperature=temperature, shape=shape):
                    query, positive_key = torch.randn(*shape), torch.randn(*shape)
                    self.assertEqual(
                        F.info_nce_loss(
                            query,
                            positive_key,
                            temperature=temperature,
                            reduction="none",
                        ),
                        self._reference(query, positive_key, None, temperature),
                    )

    def test_matches_reference_with_negatives(self):
        for temperature in (0.05, 0.2, 1.0):
            for num_negatives in (1, 7, 128):
                with self.subTest(temperature=temperature, n=num_negatives):
                    query, positive_key = torch.randn(9, 16), torch.randn(9, 16)
                    negative_keys = torch.randn(num_negatives, 16)
                    self.assertEqual(
                        F.info_nce_loss(
                            query,
                            positive_key,
                            negative_keys,
                            temperature=temperature,
                            reduction="none",
                        ),
                        self._reference(
                            query, positive_key, negative_keys, temperature
                        ),
                    )

    def test_high_temperature_limit_is_log_n(self):
        """As tau -> inf the logits flatten and the loss tends to log(N)."""
        import math

        n, d = 8, 64
        query = torch.eye(n, d)
        loss = F.info_nce_loss(query, query.clone(), temperature=1e6)
        self.assertEqual(loss.item(), math.log(n), atol=1e-3, rtol=0)

    def test_loss_decreases_with_positive_similarity(self):
        torch.manual_seed(123)
        query = torch.randn(16, 64)
        similar = F.info_nce_loss(query, query + torch.randn(16, 64) * 0.1)
        unrelated = F.info_nce_loss(query, torch.randn(16, 64))
        self.assertLess(similar.item(), unrelated.item())

    # -- reduction --------------------------------------------------------
    def test_reduction_modes_are_consistent(self):
        for use_negatives in (False, True):
            with self.subTest(use_negatives=use_negatives):
                query, positive_key = torch.randn(12, 24), torch.randn(12, 24)
                negatives = torch.randn(30, 24) if use_negatives else None
                unreduced = F.info_nce_loss(
                    query, positive_key, negatives, reduction="none"
                )
                self.assertEqual(unreduced.shape, torch.Size([12]))
                self.assertTrue((unreduced >= 0).all())
                self.assertEqual(
                    F.info_nce_loss(query, positive_key, negatives, reduction="mean"),
                    unreduced.mean(),
                )
                self.assertEqual(
                    F.info_nce_loss(query, positive_key, negatives, reduction="sum"),
                    unreduced.sum(),
                )

    def test_invalid_reduction_raises(self):
        query, positive_key = torch.randn(8, 32), torch.randn(8, 32)
        with self.assertRaisesRegex(ValueError, "not a valid value for reduction"):
            F.info_nce_loss(query, positive_key, reduction="invalid")

    def test_invalid_reduction_in_module_raises_at_forward(self):
        loss_fn = nn.InfoNCELoss(reduction="bogus")
        with self.assertRaisesRegex(ValueError, "not a valid value for reduction"):
            loss_fn(torch.randn(4, 8), torch.randn(4, 8))

    # -- temperature ------------------------------------------------------
    def test_temperature_must_be_finite_and_positive(self):
        query, positive_key = torch.randn(8, 32), torch.randn(8, 32)
        for temperature in (0.0, -0.07, float("nan"), float("inf"), float("-inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature must be finite"):
                    F.info_nce_loss(query, positive_key, temperature=temperature)

    def test_non_finite_temperature_rejected_with_negatives(self):
        """A non-finite temperature would silently produce a NaN loss."""
        query, positive_key = torch.randn(8, 32), torch.randn(8, 32)
        negative_keys = torch.randn(64, 32)
        with self.assertRaisesRegex(ValueError, "temperature must be finite"):
            F.info_nce_loss(
                query, positive_key, negative_keys, temperature=float("nan")
            )

    def test_module_rejects_non_finite_temperature_at_forward(self):
        loss_fn = nn.InfoNCELoss(temperature=float("nan"))
        with self.assertRaisesRegex(ValueError, "temperature must be finite"):
            loss_fn(torch.randn(8, 32), torch.randn(8, 32))

    def test_temperature_changes_loss(self):
        torch.manual_seed(42)
        query, positive_key = torch.randn(16, 64), torch.randn(16, 64)
        self.assertNotEqual(
            F.info_nce_loss(query, positive_key, temperature=0.01).item(),
            F.info_nce_loss(query, positive_key, temperature=1.0).item(),
        )

    def test_module_stores_temperature(self):
        self.assertEqual(nn.InfoNCELoss(temperature=0.5).temperature, 0.5)

    # -- gradients --------------------------------------------------------
    def test_gradient_flow(self):
        query = torch.randn(8, 32, requires_grad=True)
        positive_key = torch.randn(8, 32, requires_grad=True)
        negative_keys = torch.randn(64, 32, requires_grad=True)
        F.info_nce_loss(query, positive_key, negative_keys).backward()
        for tensor in (query, positive_key, negative_keys):
            self.assertIsNotNone(tensor.grad)
            self.assertFalse(torch.isnan(tensor.grad).any())

    def test_gradcheck(self):
        query = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
        positive_key = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
        self.assertTrue(
            gradcheck(
                lambda a, b: F.info_nce_loss(a, b, temperature=0.5),
                (query, positive_key),
            )
        )

    def test_gradcheck_with_negatives(self):
        query = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        positive_key = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        negative_keys = torch.randn(9, 6, dtype=torch.double, requires_grad=True)
        self.assertTrue(
            gradcheck(
                lambda a, b, c: F.info_nce_loss(a, b, c, temperature=0.3),
                (query, positive_key, negative_keys),
            )
        )

    def test_gradgradcheck(self):
        query = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
        positive_key = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
        self.assertTrue(
            gradgradcheck(
                lambda a, b: F.info_nce_loss(a, b, temperature=0.5),
                (query, positive_key),
            )
        )

    def test_optimization_reduces_loss(self):
        torch.manual_seed(31)
        encoder = nn.Linear(32, 16)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.05)
        inputs = torch.randn(24, 32)
        augmented = inputs + torch.randn(24, 32) * 0.05
        losses = []
        for _ in range(60):
            loss = F.info_nce_loss(encoder(inputs), encoder(augmented), temperature=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    # -- edge cases -------------------------------------------------------
    def test_batch_size_one_is_zero_and_differentiable(self):
        """No negatives exist, so the loss is 0 but the graph stays connected."""
        query = torch.randn(1, 32, requires_grad=True)
        positive_key = torch.randn(1, 32, requires_grad=True)
        loss = F.info_nce_loss(query, positive_key)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(query.grad)
        self.assertFalse(torch.isnan(query.grad).any())

    def test_batch_size_one_reduction_none(self):
        loss = F.info_nce_loss(torch.randn(1, 32), torch.randn(1, 32), reduction="none")
        self.assertEqual(loss.shape, torch.Size([1]))
        self.assertEqual(loss.item(), 0.0)

    def test_batch_size_one_with_negatives_is_nonzero(self):
        loss = F.info_nce_loss(
            torch.randn(1, 32), torch.randn(1, 32), torch.randn(64, 32)
        )
        self.assertGreater(loss.item(), 0)

    def test_batch_size_two(self):
        loss = F.info_nce_loss(torch.randn(2, 32), torch.randn(2, 32))
        self.assertGreaterEqual(loss.item(), 0)

    def test_embedding_dim_one(self):
        self.assertTrue(
            torch.isfinite(F.info_nce_loss(torch.randn(8, 1), torch.randn(8, 1)))
        )

    def test_empty_batch_does_not_crash(self):
        """Matches other PyTorch losses: an empty mean reduction is NaN."""
        loss = F.info_nce_loss(torch.randn(0, 16), torch.randn(0, 16))
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(
            F.info_nce_loss(
                torch.randn(0, 16), torch.randn(0, 16), reduction="none"
            ).shape,
            torch.Size([0]),
        )

    def test_non_contiguous_inputs(self):
        query = torch.randn(16, 64)[::2]
        self.assertFalse(query.is_contiguous())
        self.assertTrue(torch.isfinite(F.info_nce_loss(query, torch.randn(8, 64))))

    def test_invalid_dimensions_raise(self):
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            F.info_nce_loss(torch.randn(8), torch.randn(8, 32))
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            F.info_nce_loss(torch.randn(2, 3, 4), torch.randn(2, 3, 4))

    def test_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            F.info_nce_loss(torch.randn(8, 32), torch.randn(16, 32))

    def test_negative_keys_validation(self):
        query, positive_key = torch.randn(8, 32), torch.randn(8, 32)
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            F.info_nce_loss(query, positive_key, torch.randn(64))
        with self.assertRaisesRegex(ValueError, "embedding dim must match"):
            F.info_nce_loss(query, positive_key, torch.randn(64, 64))

    # -- numerical stability ----------------------------------------------
    def test_extreme_input_scales(self):
        for scale in (1e-8, 1e-4, 1e4, 1e8):
            with self.subTest(scale=scale):
                loss = F.info_nce_loss(
                    torch.randn(8, 16) * scale, torch.randn(8, 16) * scale
                )
                self.assertTrue(torch.isfinite(loss))

    def test_extreme_temperatures(self):
        query, positive_key = torch.randn(8, 16), torch.randn(8, 16)
        for temperature in (1e-6, 1e-3, 1e3, 1e6):
            with self.subTest(temperature=temperature):
                loss = F.info_nce_loss(query, positive_key, temperature=temperature)
                self.assertTrue(torch.isfinite(loss))

    def test_zero_vectors_do_not_produce_nan(self):
        loss = F.info_nce_loss(torch.zeros(8, 16), torch.zeros(8, 16))
        self.assertFalse(torch.isnan(loss))

    def test_gradient_stability_with_large_logits(self):
        query = (torch.randn(8, 32) * 10).requires_grad_(True)
        positive_key = (torch.randn(8, 32) * 10).requires_grad_(True)
        F.info_nce_loss(query, positive_key, temperature=0.01).backward()
        self.assertFalse(torch.isnan(query.grad).any())
        self.assertFalse(torch.isnan(positive_key.grad).any())

    # -- invariances ------------------------------------------------------
    def test_invariant_to_input_rescaling(self):
        """L2 normalization makes the loss scale invariant."""
        query, positive_key = torch.randn(8, 16), torch.randn(8, 16)
        self.assertEqual(
            F.info_nce_loss(query, positive_key),
            F.info_nce_loss(query * 3, positive_key * 11),
        )

    def test_inputs_are_not_mutated(self):
        query, positive_key = torch.randn(8, 16), torch.randn(8, 16)
        negative_keys = torch.randn(20, 16)
        originals = [t.clone() for t in (query, positive_key, negative_keys)]
        F.info_nce_loss(query, positive_key, negative_keys)
        for tensor, original in zip(
            (query, positive_key, negative_keys), originals, strict=True
        ):
            self.assertEqual(tensor, original)

    def test_symmetric_loss(self):
        """CLIP-style bidirectional InfoNCE."""
        query = torch.randn(16, 64, requires_grad=True)
        key = torch.randn(16, 64, requires_grad=True)
        loss = 0.5 * (F.info_nce_loss(query, key) + F.info_nce_loss(key, query))
        self.assertGreaterEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)

    # -- dtypes -----------------------------------------------------------
    def test_dtype_is_preserved(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                query = torch.randn(8, 16, dtype=dtype)
                positive_key = torch.randn(8, 16, dtype=dtype)
                self.assertEqual(F.info_nce_loss(query, positive_key).dtype, dtype)

    def test_low_precision_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                query = torch.randn(16, 32, dtype=dtype)
                positive_key = torch.randn(16, 32, dtype=dtype)
                self.assertTrue(torch.isfinite(F.info_nce_loss(query, positive_key)))

    # -- module plumbing --------------------------------------------------
    def test_module_attributes(self):
        self.assertIn("temperature", nn.InfoNCELoss.__constants__)
        self.assertIn("reduction", nn.InfoNCELoss.__constants__)
        loss_fn = nn.InfoNCELoss(temperature=0.1, reduction="sum")
        self.assertIn("InfoNCELoss", repr(loss_fn))
        self.assertIsInstance(loss_fn, nn.Module)
        self.assertEqual(list(loss_fn.parameters()), [])

    def test_module_matches_functional_for_all_reductions(self):
        query, positive_key = torch.randn(10, 20), torch.randn(10, 20)
        for temperature in (0.05, 0.5):
            for reduction in ("none", "mean", "sum"):
                with self.subTest(temperature=temperature, reduction=reduction):
                    loss_fn = nn.InfoNCELoss(
                        temperature=temperature, reduction=reduction
                    )
                    self.assertEqual(
                        loss_fn(query, positive_key),
                        F.info_nce_loss(
                            query,
                            positive_key,
                            temperature=temperature,
                            reduction=reduction,
                        ),
                    )

    def test_deepcopy_and_state_dict(self):
        import copy

        loss_fn = nn.InfoNCELoss(temperature=0.33)
        self.assertEqual(copy.deepcopy(loss_fn).temperature, 0.33)
        other = nn.InfoNCELoss()
        other.load_state_dict(loss_fn.state_dict())
        self.assertEqual(loss_fn.state_dict(), other.state_dict())

    # -- __torch_function__ dispatch ---------------------------------------
    def test_dispatch_covers_every_tensor_argument(self):
        """A subclass must be seen even when it is only the negative_keys."""

        class LoggingTensor(torch.Tensor):
            intercepted = None

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if func is F.info_nce_loss:
                    LoggingTensor.intercepted = func
                    return torch.zeros(())
                return super().__torch_function__(func, types, args, kwargs or {})

        query, positive_key = torch.randn(4, 8), torch.randn(4, 8)
        negative_keys = torch.randn(6, 8)
        cases = {
            "query": (query.as_subclass(LoggingTensor), positive_key, negative_keys),
            "positive_key": (
                query,
                positive_key.as_subclass(LoggingTensor),
                negative_keys,
            ),
            "negative_keys": (
                query,
                positive_key,
                negative_keys.as_subclass(LoggingTensor),
            ),
        }
        for name, args in cases.items():
            with self.subTest(argument=name):
                LoggingTensor.intercepted = None
                F.info_nce_loss(*args)
                self.assertIs(LoggingTensor.intercepted, F.info_nce_loss)

    def test_plain_tensors_do_not_dispatch(self):
        loss = F.info_nce_loss(torch.randn(4, 8), torch.randn(4, 8), torch.randn(6, 8))
        self.assertGreater(loss.item(), 0)

    # -- scripting and compilation ----------------------------------------
    def test_torchscript(self):
        scripted = torch.jit.script(nn.InfoNCELoss(temperature=0.07))
        query = torch.randn(8, 64, requires_grad=True)
        positive_key = torch.randn(8, 64)
        loss = scripted(query, positive_key)
        self.assertEqual(loss, F.info_nce_loss(query, positive_key, temperature=0.07))
        loss.backward()
        self.assertEqual(query.grad.shape, query.shape)

    def test_torchscript_with_negatives(self):
        scripted = torch.jit.script(nn.InfoNCELoss(temperature=0.1))
        query, positive_key = torch.randn(8, 64), torch.randn(8, 64)
        negative_keys = torch.randn(32, 64)
        self.assertEqual(
            scripted(query, positive_key, negative_keys),
            F.info_nce_loss(query, positive_key, negative_keys, temperature=0.1),
        )

    def test_compile_fullgraph(self):
        compiled = torch.compile(F.info_nce_loss, fullgraph=True, backend="eager")
        query, positive_key = torch.randn(16, 32), torch.randn(16, 32)
        self.assertEqual(
            compiled(query, positive_key), F.info_nce_loss(query, positive_key)
        )

    def test_compile_module(self):
        loss_fn = nn.InfoNCELoss()
        compiled = torch.compile(loss_fn, fullgraph=True, backend="eager")
        query, positive_key = torch.randn(16, 64), torch.randn(16, 64)
        self.assertEqual(compiled(query, positive_key), loss_fn(query, positive_key))

    def test_compile_dynamic_shapes(self):
        compiled = torch.compile(F.info_nce_loss, dynamic=True, backend="eager")
        for batch_size in (8, 16, 32):
            with self.subTest(batch_size=batch_size):
                query = torch.randn(batch_size, 24)
                positive_key = torch.randn(batch_size, 24)
                self.assertEqual(
                    compiled(query, positive_key),
                    F.info_nce_loss(query, positive_key),
                )

    def test_compile_backward(self):
        compiled = torch.compile(F.info_nce_loss, backend="eager")
        query = torch.randn(8, 16, requires_grad=True)
        compiled(query, torch.randn(8, 16)).backward()
        self.assertFalse(torch.isnan(query.grad).any())


class TestInfoNCELossDevice(TestCase):
    """Device-parameterized coverage."""

    def test_device_of_output_matches_input(self, device):
        query = torch.randn(16, 64, device=device)
        positive_key = torch.randn(16, 64, device=device)
        loss = F.info_nce_loss(query, positive_key)
        self.assertEqual(loss.device.type, torch.device(device).type)

    def test_gradient_on_device(self, device):
        query = torch.randn(8, 32, device=device, requires_grad=True)
        positive_key = torch.randn(8, 32, device=device)
        F.info_nce_loss(query, positive_key).backward()
        self.assertIsNotNone(query.grad)
        self.assertFalse(torch.isnan(query.grad).any())

    @onlyCPU
    def test_cpu_double_precision(self, device):
        query = torch.randn(8, 16, device=device, dtype=torch.double)
        positive_key = torch.randn(8, 16, device=device, dtype=torch.double)
        self.assertEqual(F.info_nce_loss(query, positive_key).dtype, torch.double)


instantiate_device_type_tests(TestInfoNCELossDevice, globals())


if __name__ == "__main__":
    run_tests()
