# Owner(s): ["module: inductor"]
"""Tests for the experimental torch._inductor.regional_aoti API."""

import torch
import torch.nn as nn
from torch._inductor import regional_aoti
from torch._inductor.regional_aoti import RegionConfig
from torch._inductor.test_case import run_tests, TestCase


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


regional_aoti.mark_region(GoodBlock)


class MarkedRootBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.lin(x))


regional_aoti.mark_region(MarkedRootBlock)


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

    def forward(self, x):
        y = self.lin(x)
        arr = y.detach().cpu().numpy()  # not traceable by export
        return torch.from_numpy(arr).to(x.device) * 2.0


regional_aoti.mark_region(BadBlock)


class OuterNested(nn.Module):
    """Marked outer region containing a marked inner region."""

    def __init__(self, dim=8):
        super().__init__()
        self.inner = GoodBlock(dim)  # also marked

    def forward(self, x):
        return self.inner(x)


regional_aoti.mark_region(OuterNested)


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


class OnlyBadModel(nn.Module):
    """Only a non-exportable region is marked, so nothing compiles."""

    def __init__(self, dim=8):
        super().__init__()
        self.bad = BadBlock(dim)

    def forward(self, x):
        return self.bad(x)


class MiddleRegionModel(nn.Module):
    """Marked region sits between two unmarked eager modules."""

    def __init__(self, dim=8):
        super().__init__()
        self.pre = nn.Linear(dim, dim)  # unmarked, before the region
        self.mid = GoodBlock(dim)  # marked region
        self.post = nn.Linear(dim, dim)  # unmarked, after the region

    def forward(self, x):
        return self.post(self.mid(torch.relu(self.pre(x))))


# ---------------------------------------------------------------------------
# Discovery (device-agnostic: no compilation)
# ---------------------------------------------------------------------------
class RegionDiscoveryTest(TestCase):
    def test_class_marked_submodule_is_discovered(self):
        model = SingleGoodModel()
        regions = regional_aoti._find_marked_regions(model)
        self.assertEqual([r.name for r in regions], ["good"])

    def test_marked_root_module_is_a_region(self):
        model = MarkedRootBlock()
        regions = regional_aoti._find_marked_regions(model)
        self.assertEqual([r.name for r in regions], [""])

    def test_class_marker_does_not_select_subclass(self):
        class SpecializedGoodBlock(GoodBlock):
            pass

        model = SpecializedGoodBlock()
        self.assertEqual(regional_aoti._find_marked_regions(model), [])

    def test_mark_region_works_as_class_decorator(self):
        @regional_aoti.mark_region
        class Bare(PlainBlock):
            pass

        @regional_aoti.mark_region(config=RegionConfig(rtol=1e-2))
        class Configured(PlainBlock):
            pass

        model = nn.Module()
        model.bare = Bare()
        model.configured = Configured()
        regions = {r.name: r for r in regional_aoti._find_marked_regions(model)}
        self.assertEqual(sorted(regions), ["bare", "configured"])
        self.assertEqual(regions["configured"].config.rtol, 1e-2)

    def test_mark_region_rejects_non_module_class(self):
        with self.assertRaisesRegex(TypeError, "only nn.Module classes"):

            @regional_aoti.mark_region
            class NotAModule:
                pass

    def test_programmatic_mark_region(self):
        model = PlainBlock()
        self.assertEqual(regional_aoti._find_marked_regions(model), [])
        regional_aoti.mark_region(model)
        regions = regional_aoti._find_marked_regions(model)
        self.assertEqual([r.name for r in regions], [""])

    def test_mark_region_on_submodule_instance(self):
        # mark_region() on a plain submodule *instance* is discovered too.
        model = SingleGoodModel()
        regional_aoti.mark_region(model.pre)
        names = {r.name for r in regional_aoti._find_marked_regions(model)}
        self.assertIn("pre", names)

    def test_nested_regions_returns_top_level_only(self):
        model = OuterNested()
        regions = regional_aoti._find_marked_regions(model)
        # Only the outer region; the nested "inner" is skipped.
        self.assertEqual([r.name for r in regions], [""])

    def test_config_carried_through(self):
        dim_spec = {"x": {0: torch.export.Dim("batch")}}

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        regional_aoti.mark_region(
            M, config=regional_aoti.RegionConfig(dynamic_shapes=dim_spec)
        )

        regions = regional_aoti._find_marked_regions(M())
        self.assertEqual(len(regions), 1)
        self.assertIsInstance(regions[0].config, RegionConfig)
        self.assertEqual(regions[0].config.dynamic_shapes, dim_spec)

    def test_alias_paths_are_preserved(self):
        class Aliased(nn.Module):
            def __init__(self):
                super().__init__()
                shared = PlainBlock()
                self.left = shared
                self.right = shared
                regional_aoti.mark_region(shared)

            def forward(self, x):
                return self.left(x) + self.right(x)

        (region,) = regional_aoti._find_marked_regions(Aliased())
        self.assertEqual(region.aliases, ["left", "right"])

    def test_alias_crossing_a_region_boundary_is_rejected(self):
        class Crossing(nn.Module):
            def __init__(self):
                super().__init__()
                shared = PlainBlock()
                regional_aoti.mark_region(shared)
                self.outer = GoodBlock()
                self.outer.inner = shared
                self.top = shared

            def forward(self, x):
                return self.top(self.outer(x))

        with self.assertRaisesRegex(
            regional_aoti.AOTIRegionDiscoveryError, "crosses a selected-region boundary"
        ):
            regional_aoti._find_marked_regions(Crossing())


if __name__ == "__main__":
    run_tests()
