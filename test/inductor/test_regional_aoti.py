# Owner(s): ["module: inductor"]
"""Tests for the prototype torch._inductor.regional_aoti API.

Compile/runtime tests are device-generic (via instantiate_device_type_tests) so
they run on CPU and, on the GPU target, CUDA. No ``only_for`` is passed, so under
the ASAN CPU target (where CUDA cannot init) only CPU variants are generated.
"""

from unittest import mock

import torch
import torch.nn as nn
from torch._inductor import regional_aoti
from torch._inductor.regional_aoti import (
    AOTIRegionConfig,
    AOTIRegionExportError,
    CompiledAOTIRegion,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


# ---------------------------------------------------------------------------
# Test modules
# ---------------------------------------------------------------------------
class GoodBlock(nn.Module):
    """A cleanly exportable region (marked on the class)."""

    def __init__(self, dim=8):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin2(torch.relu(self.lin1(x)))


regional_aoti.region()(GoodBlock)  # class-level marking


class MethodMarkedBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    @regional_aoti.region()
    def forward(self, x):
        return torch.relu(self.lin(x))


class PlainBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(x)


class BadBlock(nn.Module):
    """A region that cannot be exported (numpy round-trip breaks under export)."""

    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    @regional_aoti.region()
    def forward(self, x):
        y = self.lin(x)
        arr = y.detach().cpu().numpy()  # not traceable by export
        return torch.from_numpy(arr).to(x.device) * 2.0


class OuterNested(nn.Module):
    """Marked outer region containing a marked inner region."""

    def __init__(self, dim=8):
        super().__init__()
        self.inner = GoodBlock(dim)  # also marked

    @regional_aoti.region()
    def forward(self, x):
        return self.inner(x)


class ConditionalModel(nn.Module):
    """The marked region is only reached when ``use_region`` is True."""

    def __init__(self, dim=8):
        super().__init__()
        self.region = GoodBlock(dim)
        self.fallback_lin = nn.Linear(dim, dim)

    def forward(self, x, use_region: bool):
        if use_region:
            return self.region(x)
        return self.fallback_lin(x)


class TwoRegionModel(nn.Module):
    """One good + one bad region wrapped in an eager outer model."""

    def __init__(self, dim=8):
        super().__init__()
        self.pre = nn.Linear(dim, dim)  # eager, not marked
        self.good = GoodBlock(dim)
        self.bad = BadBlock(dim)

    def forward(self, x):
        x = torch.relu(self.pre(x))
        x = self.good(x)
        x = self.bad(x)
        return x


class SingleGoodModel(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.pre = nn.Linear(dim, dim)
        self.good = GoodBlock(dim)

    def forward(self, x):
        return self.good(torch.relu(self.pre(x)))


# ---------------------------------------------------------------------------
# Discovery (device-agnostic: no compilation)
# ---------------------------------------------------------------------------
class RegionDiscoveryTest(TestCase):
    def test_class_decorated_region_discovered(self):
        model = SingleGoodModel()
        regions = regional_aoti.discover_regions(model)
        self.assertEqual([r.name for r in regions], ["good"])

    def test_method_decorated_region_discovered(self):
        model = MethodMarkedBlock()
        regions = regional_aoti.discover_regions(model)
        self.assertEqual([r.name for r in regions], [""])

    def test_programmatic_mark_region(self):
        model = PlainBlock()
        self.assertEqual(regional_aoti.discover_regions(model), [])
        regional_aoti.mark_region(model)
        regions = regional_aoti.discover_regions(model)
        self.assertEqual([r.name for r in regions], [""])

    def test_mark_region_on_submodule_instance(self):
        # mark_region() on a plain submodule *instance* is discovered too.
        model = SingleGoodModel()
        regional_aoti.mark_region(model.pre)
        names = {r.name for r in regional_aoti.discover_regions(model)}
        self.assertIn("pre", names)

    def test_nested_regions_returns_top_level_only(self):
        model = OuterNested()
        regions = regional_aoti.discover_regions(model)
        # Only the outer region; the nested "inner" is skipped.
        self.assertEqual([r.name for r in regions], [""])

    def test_config_carried_through(self):
        dim_spec = {"x": {0: torch.export.Dim("batch")}}

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

            @regional_aoti.region(dynamic_shapes=dim_spec)
            def forward(self, x):
                return self.lin(x)

        regions = regional_aoti.discover_regions(M())
        self.assertEqual(len(regions), 1)
        self.assertIsInstance(regions[0].config, AOTIRegionConfig)
        self.assertEqual(regions[0].config.dynamic_shapes, dim_spec)


# ---------------------------------------------------------------------------
# Capture (device-agnostic: eager forward only)
# ---------------------------------------------------------------------------
class RegionCaptureTest(TestCase):
    def test_capture_positional_args(self):
        model = SingleGoodModel()
        x = torch.randn(3, 8)
        regions = regional_aoti.discover_regions(model)
        regional_aoti.capture_region_inputs(model, regions, (x,))
        (r,) = regions
        self.assertEqual(r.status, "captured")
        self.assertEqual(len(r.captured_args), 1)
        self.assertEqual(r.captured_args[0].shape, torch.Size([3, 8]))

    def test_capture_kwargs(self):
        class KwModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.good = GoodBlock(8)

            def forward(self, x):
                return self.good(x=x)

        model = KwModel()
        regions = regional_aoti.discover_regions(model)
        regional_aoti.capture_region_inputs(model, regions, (torch.randn(2, 8),))
        (r,) = regions
        self.assertEqual(r.status, "captured")
        self.assertIn("x", r.captured_kwargs)

    def test_region_not_reached_reported(self):
        model = ConditionalModel()
        regions = regional_aoti.discover_regions(model)
        regional_aoti.capture_region_inputs(
            model, regions, (torch.randn(2, 8),), {"use_region": False}
        )
        (r,) = regions
        self.assertEqual(r.status, "not_reached")


# ---------------------------------------------------------------------------
# Export + compile + fallback (device-generic: CPU + GPU)
# ---------------------------------------------------------------------------
class RegionCompileTest(TestCase):
    def test_single_region_compiles_with_parity(self, device):
        model = SingleGoodModel().to(device)
        x = torch.randn(3, 8, device=device)
        with torch.no_grad():
            expected = model(x)
        compiled = regional_aoti.compile_regions(
            model, (x,), fallback="error", check_parity=True
        )
        result = compiled._regional_aoti_result
        self.assertEqual([r.name for r in result.compiled()], ["good"])
        self.assertIsInstance(compiled.good, CompiledAOTIRegion)
        with torch.no_grad():
            got = compiled(x)
        self.assertEqual(got, expected, rtol=1e-3, atol=1e-3)

    def test_region_in_modulelist_is_replaced(self, device):
        # A region inside an nn.ModuleList must be swapped in-place (container
        # elements are index-addressed, not plain attributes).
        class Stacked(nn.Module):
            def __init__(self, dim=8):
                super().__init__()
                self.blocks = nn.ModuleList([GoodBlock(dim)])

            def forward(self, x):
                return self.blocks[0](x)

        model = Stacked().to(device)
        x = torch.randn(3, 8, device=device)
        with torch.no_grad():
            expected = model(x)
        compiled = regional_aoti.compile_regions(
            model, (x,), fallback="error", check_parity=True
        )
        result = compiled._regional_aoti_result
        self.assertEqual([r.name for r in result.compiled()], ["blocks.0"])
        self.assertIsInstance(compiled.blocks[0], CompiledAOTIRegion)
        with torch.no_grad():
            got = compiled(x)
        self.assertEqual(got, expected, rtol=1e-3, atol=1e-3)

    def test_parity_failure_falls_back_eager(self, device):
        # A parity mismatch under fallback="eager" keeps the region eager instead
        # of escaping as an uncaught AssertionError.
        x = torch.randn(3, 8, device=device)
        err = regional_aoti.AOTIRegionParityError("forced mismatch")
        with (
            mock.patch.object(
                regional_aoti, "_compile_region", return_value="unused.pt2"
            ),
            mock.patch.object(regional_aoti, "_check_parity", side_effect=err),
        ):
            compiled = regional_aoti.compile_regions(
                SingleGoodModel().to(device), (x,), fallback="eager", check_parity=True
            )
        result = compiled._regional_aoti_result
        self.assertEqual([r.name for r in result.fallback()], ["good"])
        self.assertNotIsInstance(compiled.good, CompiledAOTIRegion)

    def test_parity_failure_raises_under_error(self, device):
        x = torch.randn(3, 8, device=device)
        err = regional_aoti.AOTIRegionParityError("forced mismatch")
        with (
            mock.patch.object(
                regional_aoti, "_compile_region", return_value="unused.pt2"
            ),
            mock.patch.object(regional_aoti, "_check_parity", side_effect=err),
        ):
            with self.assertRaises(regional_aoti.AOTIRegionParityError):
                regional_aoti.compile_regions(
                    SingleGoodModel().to(device),
                    (x,),
                    fallback="error",
                    check_parity=True,
                )

    def test_fallback_eager_keeps_bad_region_eager(self, device):
        model = TwoRegionModel().to(device)
        x = torch.randn(3, 8, device=device)
        with torch.no_grad():
            expected = model(x)
        compiled = regional_aoti.compile_regions(model, (x,), fallback="eager")
        result = compiled._regional_aoti_result
        self.assertEqual([r.name for r in result.compiled()], ["good"])
        self.assertEqual([r.name for r in result.fallback()], ["bad"])
        # Good region is compiled; bad region stays the original eager module.
        self.assertIsInstance(compiled.good, CompiledAOTIRegion)
        self.assertNotIsInstance(compiled.bad, CompiledAOTIRegion)
        with torch.no_grad():
            got = compiled(x)
        self.assertEqual(got, expected, rtol=1e-3, atol=1e-3)

    def test_fallback_error_raises_on_bad_region(self, device):
        # Only a non-exportable region is marked, so compile_regions raises
        # immediately without paying for an unrelated (good-region) compile.
        class OnlyBad(nn.Module):
            def __init__(self):
                super().__init__()
                self.bad = BadBlock()

            def forward(self, x):
                return self.bad(x)

        model = OnlyBad().to(device)
        x = torch.randn(3, 8, device=device)
        with self.assertRaises(AOTIRegionExportError):
            regional_aoti.compile_regions(model, (x,), fallback="error")


# ---------------------------------------------------------------------------
# Runtime wrapper + state ownership (device-generic: CPU + GPU)
# ---------------------------------------------------------------------------
class RegionStateTest(TestCase):
    def test_state_dict_keys_preserved(self, device):
        model = SingleGoodModel().to(device)
        x = torch.randn(2, 8, device=device)
        before_keys = set(model.state_dict().keys())
        compiled = regional_aoti.compile_regions(model, (x,), fallback="error")
        after_keys = set(compiled.state_dict().keys())
        self.assertEqual(before_keys, after_keys)

    def test_runtime_fallback_paths(self, device):
        # A compiled region that throws at runtime: runtime_fallback="error"
        # surfaces AOTIRegionRuntimeError; "eager" falls back to the eager module.
        model = SingleGoodModel().to(device)
        x = torch.randn(2, 8, device=device)
        with torch.no_grad():
            expected = model(x)
        compiled = regional_aoti.compile_regions(model, (x,), fallback="error")
        region = compiled.good
        self.assertIsInstance(region, CompiledAOTIRegion)

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        region._compiled = boom  # inject a failing compiled artifact

        region._runtime_fallback = "error"
        with self.assertRaises(regional_aoti.AOTIRegionRuntimeError):
            compiled(x)

        region._runtime_fallback = "eager"
        with torch.no_grad():
            got = compiled(x)
        self.assertEqual(got, expected, rtol=1e-3, atol=1e-3)

    def test_load_state_dict_changes_output(self, device):
        model = SingleGoodModel().to(device)
        x = torch.randn(2, 8, device=device)
        compiled = regional_aoti.compile_regions(model, (x,), fallback="error")
        with torch.no_grad():
            out1 = compiled(x)

        # Load clearly-distinct weights so the "output changed" check can't flake.
        new_model = SingleGoodModel().to(device)
        new_sd = {k: torch.full_like(v, 0.5) for k, v in model.state_dict().items()}
        new_model.load_state_dict(new_sd)
        compiled.load_state_dict(new_sd)
        with torch.no_grad():
            out2 = compiled(x)
            expected = new_model(x)
        self.assertFalse(torch.allclose(out1, out2, rtol=1e-3, atol=1e-3))
        self.assertEqual(out2, expected, rtol=1e-3, atol=1e-3)


# Generate CPU (always) and CUDA (when available) variants. No ``only_for`` so
# the ASAN CPU target -- where CUDA cannot init -- yields CPU-only variants.
instantiate_device_type_tests(RegionCompileTest, globals())
instantiate_device_type_tests(RegionStateTest, globals())


if __name__ == "__main__":
    run_tests()
