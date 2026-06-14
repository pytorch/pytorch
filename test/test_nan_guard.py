# Owner(s): ["module: nn"]

import warnings

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.nan_guard import nan_guard, NaNGuard, NaNGuardError


class _ProduceNaN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + float("nan")


class _ProduceInf(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + float("inf")


class _DictOut(nn.Module):
    def forward(self, x: torch.Tensor):
        return {"clean": x, "bad": x + float("nan")}


class _Identity(nn.Module):
    def forward(self, x):
        return x


class TestNaNGuard(TestCase):
    def test_nan_in_output_raises(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), _ProduceNaN())
        x = torch.randn(2, 4)
        with self.assertRaisesRegex(NaNGuardError, r"nan="):
            with NaNGuard(model):
                model(x)

    def test_message_names_offending_submodule(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), _ProduceNaN())
        x = torch.randn(2, 4)
        with self.assertRaises(NaNGuardError) as cm:
            with NaNGuard(model):
                model(x)
        msg = str(cm.exception)
        self.assertIn("'1'", msg)
        self.assertIn("_ProduceNaN", msg)

    def test_inf_default_caught(self) -> None:
        model = _ProduceInf()
        x = torch.randn(2, 4)
        with self.assertRaisesRegex(NaNGuardError, r"inf="):
            with NaNGuard(model):
                model(x)

    def test_inf_disabled_passes_through(self) -> None:
        model = _ProduceInf()
        x = torch.randn(2, 4)
        with NaNGuard(model, check_inf=False):
            out = model(x)
        self.assertTrue(torch.isinf(out).any())

    def test_dict_output_path_reported(self) -> None:
        model = _DictOut()
        x = torch.randn(2, 4)
        with self.assertRaises(NaNGuardError) as cm:
            with NaNGuard(model):
                model(x)
        self.assertIn("['bad']", str(cm.exception))

    def test_check_inputs_skips_propagator(self) -> None:
        bad = torch.full((4,), float("nan"))
        identity = _Identity()
        with NaNGuard(identity, check_inputs=True):
            out = identity(bad)
        self.assertTrue(torch.isnan(out).all())

    def test_check_inputs_still_catches_producer(self) -> None:
        model = nn.Sequential(_ProduceNaN(), _Identity())
        x = torch.randn(4)
        with self.assertRaises(NaNGuardError) as cm:
            with NaNGuard(model, check_inputs=True):
                model(x)
        self.assertIn("_ProduceNaN", str(cm.exception))

    def test_warn_mode_does_not_raise(self) -> None:
        model = _ProduceNaN()
        x = torch.randn(4)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            with NaNGuard(model, on_detect="warn"):
                out = model(x)
        self.assertTrue(any("NaNGuard" in str(w.message) for w in ws))
        self.assertTrue(torch.isnan(out).all())

    def test_hooks_removed_on_normal_exit(self) -> None:
        model = nn.Linear(4, 4)
        before = len(model._forward_hooks)
        with NaNGuard(model):
            self.assertGreater(len(model._forward_hooks), before)
        self.assertEqual(len(model._forward_hooks), before)

    def test_hooks_removed_on_exception(self) -> None:
        model = _ProduceNaN()
        before = len(model._forward_hooks)
        with self.assertRaises(NaNGuardError):
            with NaNGuard(model):
                model(torch.randn(4))
        self.assertEqual(len(model._forward_hooks), before)

    def test_clean_forward_is_quiet(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        x = torch.randn(2, 4)
        with NaNGuard(model):
            out = model(x)
        self.assertEqual(out.shape, (2, 4))
        self.assertFalse(torch.isnan(out).any())

    def test_integer_outputs_are_skipped(self) -> None:
        class IntOut(nn.Module):
            def forward(self, x):
                return torch.tensor([1, 2, 3], dtype=torch.int64)

        model = IntOut()
        with NaNGuard(model):
            out = model(torch.randn(1))
        self.assertEqual(out.dtype, torch.int64)

    def test_invalid_on_detect(self) -> None:
        with self.assertRaisesRegex(ValueError, "on_detect"):
            NaNGuard(nn.Linear(2, 2), on_detect="ignore")  # type: ignore[arg-type]

    def test_helper_function(self) -> None:
        model = _ProduceNaN()
        with self.assertRaises(NaNGuardError):
            with nan_guard(model):
                model(torch.randn(4))


if __name__ == "__main__":
    run_tests()
