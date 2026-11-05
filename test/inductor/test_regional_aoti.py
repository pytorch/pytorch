# Owner(s): ["module: inductor"]
"""Tests for the experimental torch._inductor.regional_aoti API."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch._inductor import regional_aoti
from torch._inductor.regional_aoti import RegionConfig
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


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


# ---------------------------------------------------------------------------
# Capture (device-agnostic: eager forward only)
# ---------------------------------------------------------------------------
class RegionCaptureTest(TestCase):
    def test_capture_records_input_at_region_boundary(self):
        model = MiddleRegionModel()
        x = torch.randn(3, 8)
        with torch.no_grad():
            region_input = torch.relu(model.pre(x))
            expected = model(x)
        regions = regional_aoti._find_marked_regions(model)
        regional_aoti.capture_region_inputs(model, regions, (x,))
        (r,) = regions
        self.assertEqual(len(r.captured_calls), 1)
        self.assertEqual(len(r.captured_args), 1)
        self.assertEqual(r.captured_args[0], region_input)
        self.assertIsNone(r.status)
        with torch.no_grad():
            self.assertEqual(model(x), expected)

    def test_capture_kwargs(self):
        class KwModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.good = GoodBlock(8)

            def forward(self, x):
                return self.good(x=x)

        model = KwModel()
        regions = regional_aoti._find_marked_regions(model)
        regional_aoti.capture_region_inputs(model, regions, (torch.randn(2, 8),))
        (r,) = regions
        self.assertEqual(len(r.captured_calls), 1)
        self.assertIn("x", r.captured_kwargs)

    def test_region_not_reached_reported(self):
        model = ConditionalModel()
        regions = regional_aoti._find_marked_regions(model)
        regional_aoti.capture_region_inputs(
            model, regions, (torch.randn(2, 8),), {"use_region": True}
        )
        (r,) = regions
        self.assertEqual(len(r.captured_calls), 1)
        regional_aoti.capture_region_inputs(
            model, regions, (torch.randn(2, 8),), {"use_region": False}
        )
        self.assertEqual(r.status, "not_reached")
        self.assertEqual(r.captured_calls, [])

    def test_clone_inputs_preserves_repeated_tensor_identity(self):
        class PairRegion(nn.Module):
            def forward(self, x, y):
                return x + y

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.region = PairRegion()
                regional_aoti.mark_region(self.region)

            def forward(self, x):
                return self.region(x, x)

        model = Model()
        regions = regional_aoti._find_marked_regions(model)
        regional_aoti.capture_region_inputs(
            model, regions, (torch.randn(2, 8),), clone_inputs=True
        )
        captured_x, captured_y = regions[0].captured_args
        self.assertIs(captured_x, captured_y)

    def test_clone_inputs_rejects_distinct_views_with_shared_storage(self):
        class PairRegion(nn.Module):
            def forward(self, x, y):
                return x + y

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.region = PairRegion()
                regional_aoti.mark_region(self.region)

            def forward(self, x):
                return self.region(x, x.view_as(x))

        model = Model()
        regions = regional_aoti._find_marked_regions(model)
        with self.assertRaisesRegex(
            regional_aoti.AOTIRegionCaptureError, "distinct tensor views"
        ):
            regional_aoti.capture_region_inputs(
                model, regions, (torch.randn(2, 8),), clone_inputs=True
            )

    def test_clone_inputs_preserves_non_dense_stride(self):
        block = nn.Identity()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        x = torch._neg_view(torch.arange(8)[::2])
        regional_aoti.capture_region_inputs(block, [region], (x,), clone_inputs=True)
        captured = region.captured_args[0]
        self.assertEqual(captured, x)
        self.assertEqual(captured.stride(), x.stride())
        self.assertTrue(captured.is_neg())

    def test_clone_inputs_preserves_storage_offset(self):
        block = nn.Identity()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        x = torch.arange(8)[3:7]

        regional_aoti.capture_region_inputs(block, [region], (x,), clone_inputs=True)

        captured = region.captured_args[0]
        self.assertEqual(captured, x)
        self.assertEqual(captured.stride(), x.stride())
        self.assertEqual(captured.storage_offset(), x.storage_offset())

    def test_clone_inputs_preserves_expanded_stride(self):
        block = nn.Identity()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        x = torch.arange(8).expand(4, 8)

        regional_aoti.capture_region_inputs(block, [region], (x,), clone_inputs=True)

        captured = region.captured_args[0]
        self.assertEqual(captured, x)
        self.assertIsNot(captured, x)
        self.assertEqual(captured.stride(), x.stride())

    def test_clone_inputs_preserves_empty_view_offset(self):
        block = nn.Identity()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        x = torch.empty(0).as_strided((0,), (1,), storage_offset=5)

        regional_aoti.capture_region_inputs(block, [region], (x,), clone_inputs=True)

        captured = region.captured_args[0]
        self.assertEqual(captured, x)
        self.assertIsNot(captured, x)
        self.assertEqual(captured.stride(), x.stride())
        self.assertEqual(captured.storage_offset(), x.storage_offset())

    def test_default_capture_snapshots_mutable_non_tensor_leaf(self):
        @dataclass
        class Box:
            value: int

        class Block(nn.Module):
            def forward(self, box):
                return torch.tensor(box.value)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.region = Block()
                regional_aoti.mark_region(self.region)

            def forward(self, box):
                result = self.region(box)
                box.value += 1
                return result

        model = Model()
        regions = regional_aoti._find_marked_regions(model)
        box = Box(1)
        regional_aoti.capture_region_inputs(model, regions, (box,))

        (captured_box,) = regions[0].captured_args
        self.assertEqual(captured_box.value, 1)
        self.assertIsNot(captured_box, box)
        self.assertEqual(box.value, 2)

    def test_capture_rejects_input_identity_shared_with_region_state(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.payload = [1]

            def forward(self, payload):
                return torch.tensor(payload is self.payload)

        block = Block()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)

        with self.assertRaisesRegex(
            regional_aoti.AOTIRegionCaptureError,
            "object identity.*relationships with region state",
        ):
            regional_aoti.capture_region_inputs(block, [region], (block.payload,))

    def test_clone_inputs_rejects_storage_shared_with_region_state(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.arange(8.0))

            def forward(self, x):
                return x + self.buffer

        block = Block()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)

        with self.assertRaisesRegex(
            regional_aoti.AOTIRegionCaptureError,
            "storage relationships with region state",
        ):
            regional_aoti.capture_region_inputs(
                block,
                [region],
                (block.buffer.view_as(block.buffer),),
                clone_inputs=True,
            )

    def test_clone_inputs_rejects_slotted_input_sharing_region_storage(self):
        class Holder:
            __slots__ = ("tensor", "unset")

            def __init__(self, tensor):
                self.tensor = tensor

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.arange(8.0))

            def forward(self, holder):
                return holder.tensor + self.buffer

        block = Block()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        holder = Holder(block.buffer.view_as(block.buffer))
        self.assertEqual(
            regional_aoti._slot_values(holder), [("tensor", holder.tensor)]
        )

        with self.assertRaisesRegex(
            regional_aoti.AOTIRegionCaptureError,
            "storage relationships with region state",
        ):
            regional_aoti.capture_region_inputs(
                block, [region], (holder,), clone_inputs=True
            )

    def test_capture_preserves_requires_grad(self):
        block = nn.Identity()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        x = torch.randn(2, 8, requires_grad=True)

        regional_aoti.capture_region_inputs(block, [region], (x,))

        captured = region.captured_args[0]
        self.assertTrue(captured.requires_grad)
        self.assertIsNot(captured, x)

    @parametrize("clone_inputs", [False, True])
    def test_capture_preserves_parameter_input_type(self, clone_inputs):
        class Block(nn.Module):
            def forward(self, x):
                return torch.tensor(isinstance(x, nn.Parameter))

        block = Block()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        parameter = nn.Parameter(torch.randn(2, 8))

        regional_aoti.capture_region_inputs(
            block, [region], (parameter,), clone_inputs=clone_inputs
        )

        captured = region.captured_args[0]
        self.assertIsInstance(captured, nn.Parameter)
        self.assertEqual(captured.requires_grad, parameter.requires_grad)

    def test_capture_preserves_repeated_container_identity(self):
        class Block(nn.Module):
            def forward(self, left, right):
                return torch.tensor(left is right)

        block = Block()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        shared = [torch.ones(1)]
        regional_aoti.capture_region_inputs(block, [region], (shared, shared))

        captured_left, captured_right = region.captured_args
        self.assertIs(captured_left, captured_right)
        self.assertIsNot(captured_left, shared)

    def test_clone_inputs_supports_sparse_csr(self):
        class Block(nn.Module):
            def forward(self, x):
                return x.to_dense().sum()

        block = Block()
        regional_aoti.mark_region(block)
        (region,) = regional_aoti._find_marked_regions(block)
        x = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse_csr()
        regional_aoti.capture_region_inputs(block, [region], (x,), clone_inputs=True)

        captured = region.captured_args[0]
        self.assertEqual(captured, x)
        self.assertIsNot(captured, x)

    def test_all_alias_invocations_are_captured(self):
        class Aliased(nn.Module):
            def __init__(self):
                super().__init__()
                shared = PlainBlock()
                self.left = shared
                self.right = shared
                regional_aoti.mark_region(shared)

            def forward(self, x):
                return self.left(x) + self.right(x + 1)

        model = Aliased()
        regions = regional_aoti._find_marked_regions(model)
        regional_aoti.capture_region_inputs(model, regions, (torch.randn(2, 8),))
        self.assertEqual(len(regions[0].captured_calls), 2)

    def test_duplicate_region_entries_share_one_capture_hook(self):
        block = nn.Identity()
        first = regional_aoti.RegionResult(
            name="first",
            module=block,
            config=regional_aoti.RegionConfig(),
        )
        second = regional_aoti.RegionResult(
            name="second",
            module=block,
            config=regional_aoti.RegionConfig(),
        )

        regional_aoti.capture_region_inputs(
            block, [first, second], (torch.randn(2, 8),)
        )

        self.assertEqual(len(first.captured_calls), 1)
        self.assertEqual(len(second.captured_calls), 1)


instantiate_parametrized_tests(RegionCaptureTest)


if __name__ == "__main__":
    run_tests()
