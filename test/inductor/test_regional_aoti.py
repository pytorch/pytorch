# Owner(s): ["module: inductor"]
"""Tests for the experimental torch._inductor.regional_aoti API."""

import copy
import dataclasses
import pickle
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._inductor import regional_aoti
from torch._inductor.regional_aoti import RegionConfig
from torch._inductor.test_case import run_tests, TestCase
from torch.nn.utils.parametrize import register_parametrization
from torch.testing._internal.common_utils import HardwareClassification


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


class OuterNested(nn.Module):
    """Marked outer region containing a marked inner region."""

    def __init__(self, dim=8):
        super().__init__()
        self.inner = GoodBlock(dim)  # also marked

    def forward(self, x):
        return self.inner(x)


regional_aoti.mark_region(OuterNested)


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
    hw_classification = HardwareClassification.GENERIC

    def test_class_marked_submodule_is_discovered(self):
        model = SingleGoodModel()
        regions = regional_aoti._find_marked_regions(model)
        self.assertEqual([r.name for r in regions], ["good"])

    def test_marked_root_module_is_a_region(self):
        model = MarkedRootBlock()
        regions = regional_aoti._find_marked_regions(model)
        self.assertEqual([r.name for r in regions], [""])

    def test_class_mark_skips_subclass_with_own_forward(self):
        class ShiftedGoodBlock(GoodBlock):
            def forward(self, x):
                return super().forward(x) + 1

        model = ShiftedGoodBlock()
        self.assertEqual(regional_aoti._find_marked_regions(model), [])

    def test_class_mark_applies_to_subclass_inheriting_forward(self):
        class WideGoodBlock(GoodBlock):
            def __init__(self):
                super().__init__(dim=16)

        regions = regional_aoti._find_marked_regions(WideGoodBlock())
        self.assertEqual([r.name for r in regions], [""])

    def test_class_mark_survives_parametrization_class_swap(self):
        @regional_aoti.mark_region(config=RegionConfig(rtol=1e-2))
        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(8))

            def forward(self, x):
                return x * self.weight

        class Double(nn.Module):
            def forward(self, weight):
                return weight * 2

        model = Scale()
        regional_aoti.mark_region(model, config=RegionConfig(atol=1e-4))
        register_parametrization(model, "weight", Double())
        self.assertIsNot(type(model), Scale)

        (region,) = regional_aoti._find_marked_regions(model)
        self.assertEqual(region.config.rtol, 1e-2)
        self.assertEqual(region.config.atol, 1e-4)

    @unittest.skipIf(not dist.is_available(), "requires torch.distributed")
    def test_class_mark_survives_fsdp_and_replicate_class_swaps(self):
        from torch.distributed._composable.replicate import DDP
        from torch.distributed.fsdp import FSDPModule

        for wrapper in (FSDPModule, DDP):
            with self.subTest(wrapper=wrapper.__name__):
                model = GoodBlock()
                model.__class__ = type(
                    f"{wrapper.__name__}GoodBlock", (wrapper, GoodBlock), {}
                )
                (region,) = regional_aoti._find_marked_regions(model)
                self.assertIs(region.module, model)

    def test_class_marks_merge_along_inherited_forward(self):
        dim_spec = {"x": {0: torch.export.Dim("batch")}}

        @regional_aoti.mark_region(
            config=RegionConfig(dynamic_shapes=dim_spec, rtol=1e-1)
        )
        class Base(PlainBlock):
            pass

        @regional_aoti.mark_region(config=RegionConfig(rtol=1e-2))
        class Child(Base):
            pass

        @regional_aoti.mark_region(config=RegionConfig(atol=1e-4))
        class OwnForward(Base):
            def forward(self, x):
                return super().forward(x)

        (child,) = regional_aoti._find_marked_regions(Child())
        self.assertEqual(child.config.dynamic_shapes, dim_spec)
        self.assertEqual(child.config.rtol, 1e-2)

        (own,) = regional_aoti._find_marked_regions(OwnForward())
        self.assertIsNone(own.config.dynamic_shapes)
        self.assertEqual(own.config.rtol, 1e-3)
        self.assertEqual(own.config.atol, 1e-4)

    def test_class_mark_applies_only_where_its_forward_is_used(self):
        @regional_aoti.mark_region(config=RegionConfig(rtol=1e-2))
        class Tagged(nn.Module):
            def forward(self, x):
                return x

        class TaggedFirst(Tagged, PlainBlock):
            pass

        class TaggedLast(PlainBlock, Tagged):
            pass

        @regional_aoti.mark_region
        class TaggedPlain(PlainBlock):
            pass

        class Override(PlainBlock):
            def forward(self, x):
                return super().forward(x) + 1

        # The MRO puts TaggedPlain before Override, but forward is Override's.
        class Diamond(TaggedPlain, Override):
            pass

        (region,) = regional_aoti._find_marked_regions(TaggedFirst())
        self.assertEqual(region.config.rtol, 1e-2)
        self.assertEqual(regional_aoti._find_marked_regions(TaggedLast()), [])
        self.assertEqual(regional_aoti._find_marked_regions(Diamond()), [])

    def test_mark_region_rejects_class_without_forward(self):
        class Mixin(nn.Module):
            pass

        with self.assertRaisesRegex(TypeError, "needs a module with a forward"):
            regional_aoti.mark_region(Mixin)

    def test_mark_region_rejects_instance_without_forward(self):
        layers = nn.ModuleList([GoodBlock(), GoodBlock()])
        with self.assertRaisesRegex(TypeError, "needs a module with a forward"):
            regional_aoti.mark_region(layers)
        with self.assertRaisesRegex(TypeError, "needs a module with a forward"):
            regional_aoti.mark_region(nn.ModuleDict({"good": GoodBlock()}))

        model = nn.Module()
        model.layers = layers
        names = [r.name for r in regional_aoti._find_marked_regions(model)]
        self.assertEqual(names, ["layers.0", "layers.1"])

        patched = nn.Module()
        patched.forward = lambda x: x
        self.assertIs(regional_aoti.mark_region(patched), patched)

    def test_marking_again_replaces_config(self):
        class Remarked(PlainBlock):
            pass

        for target in (PlainBlock(), Remarked):
            with self.subTest(target=target):
                regional_aoti.mark_region(target, config=RegionConfig(rtol=1e-2))
                regional_aoti.mark_region(target, config=RegionConfig(atol=1e-4))
                model = target() if isinstance(target, type) else target
                (region,) = regional_aoti._find_marked_regions(model)
                self.assertEqual(region.config.rtol, 1e-3)
                self.assertEqual(region.config.atol, 1e-4)

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
        names = [r.name for r in regional_aoti._find_marked_regions(model)]
        self.assertEqual(names, ["pre", "good"])

    def test_instance_mark_leaves_class_unmarked(self):
        regional_aoti.mark_region(PlainBlock())
        self.assertEqual(regional_aoti._find_marked_regions(PlainBlock()), [])

    def test_sibling_whose_name_extends_another_is_not_nested(self):
        model = nn.Module()
        model.blk = regional_aoti.mark_region(PlainBlock())
        model.blk2 = regional_aoti.mark_region(PlainBlock())
        names = [r.name for r in regional_aoti._find_marked_regions(model)]
        self.assertEqual(names, ["blk", "blk2"])

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

    def test_config_survives_deepcopy_of_instance_marked_module(self):
        @regional_aoti.mark_region(config=RegionConfig(rtol=1e-2))
        class ClassMarked(PlainBlock):
            pass

        model = ClassMarked()
        regional_aoti.mark_region(model, config=RegionConfig(atol=1e-4))
        (region,) = regional_aoti._find_marked_regions(copy.deepcopy(model))
        self.assertEqual(region.config.rtol, 1e-2)
        self.assertEqual(region.config.atol, 1e-4)
        self.assertIsNone(region.config.dynamic_shapes)
        self.assertIsNone(region.config.inductor_configs)

    def test_unset_sentinel_survives_pickle(self):
        config = pickle.loads(pickle.dumps(RegionConfig(rtol=1e-2)))
        self.assertIs(config.atol, regional_aoti._UNSET)

    def test_instance_config_overrides_class_config_per_field(self):
        @regional_aoti.mark_region(config=RegionConfig(rtol=1e-2, atol=1e-2))
        class ClassMarked(PlainBlock):
            pass

        model = ClassMarked()
        regional_aoti.mark_region(model, config=RegionConfig(rtol=None, atol=1e-4))
        (region,) = regional_aoti._find_marked_regions(model)
        self.assertIsNone(region.config.rtol)
        self.assertEqual(region.config.atol, 1e-4)

    def test_unset_fields_use_defaults(self):
        model = regional_aoti.mark_region(PlainBlock())
        (region,) = regional_aoti._find_marked_regions(model)
        self.assertEqual(region.config.rtol, 1e-3)
        self.assertEqual(region.config.atol, 1e-3)
        self.assertIsNone(region.config.dynamic_shapes)
        self.assertIsNone(region.config.inductor_configs)

    def test_region_config_is_keyword_only(self):
        with self.assertRaises(TypeError):
            RegionConfig(None, None, 1e-2)

    def test_region_config_rejects_positional_dynamic_shapes(self):
        with self.assertRaisesRegex(TypeError, "dict keyed by forward argument"):
            RegionConfig(dynamic_shapes=({0: torch.export.Dim("batch")},))

    def test_region_config_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            RegionConfig().rtol = 1e-2

    def test_every_config_field_has_a_default(self):
        self.assertEqual(
            {field.name for field in dataclasses.fields(RegionConfig)},
            set(regional_aoti._CONFIG_DEFAULTS),
        )

    def test_config_containers_are_immutable(self):
        inductor_configs = {"max_autotune": True}
        config = RegionConfig(inductor_configs=inductor_configs)
        model = regional_aoti.mark_region(PlainBlock(), config=config)
        inductor_configs["aot_inductor.package"] = True

        (region,) = regional_aoti._find_marked_regions(model)
        self.assertEqual(region.config.inductor_configs, {"max_autotune": True})
        with self.assertRaises(TypeError):
            region.config.inductor_configs["aot_inductor.package"] = True
        with self.assertRaises(TypeError):
            config.inductor_configs["aot_inductor.package"] = True
        self.assertEqual(hash(config), hash(RegionConfig(**vars(config))))

    def test_thawed_config_works_with_export_and_compile_options(self):
        class Inputs(nn.Module):
            def forward(self, x, pair, mapping):
                return x + pair[0] + mapping["a"]

        batch = torch.export.Dim("batch")
        config = RegionConfig(
            dynamic_shapes={
                "x": {0: batch},
                "pair": [{0: batch}],
                "mapping": {"a": {0: batch}},
            },
            inductor_configs={"max_autotune": True},
        )
        args = (torch.randn(4, 2), [torch.randn(4, 2)], {"a": torch.randn(4, 2)})
        ep = torch.export.export(
            Inputs(), args, dynamic_shapes=regional_aoti._thaw(config.dynamic_shapes)
        )
        batch_sizes = [
            node.meta["val"].shape[0]
            for node in ep.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertEqual(len(batch_sizes), 3)
        self.assertTrue(all(isinstance(size, torch.SymInt) for size in batch_sizes))

        # aoti_compile_and_package writes into the dict it receives.
        inductor_configs = regional_aoti._thaw(config.inductor_configs)
        inductor_configs["aot_inductor.package"] = True
        self.assertEqual(config.inductor_configs, {"max_autotune": True})

    def test_nested_dynamic_shapes_are_copied(self):
        dim_spec = {"x": {0: torch.export.Dim("batch")}}

        @regional_aoti.mark_region(config=RegionConfig(dynamic_shapes=dim_spec))
        class ClassMarked(PlainBlock):
            pass

        dim_spec["x"][1] = torch.export.Dim("caller_edit")
        (region,) = regional_aoti._find_marked_regions(ClassMarked())
        self.assertEqual(list(region.config.dynamic_shapes["x"]), [0])
        self.assertIs(region.config.dynamic_shapes["x"][0], dim_spec["x"][0])

        with self.assertRaises(TypeError):
            region.config.dynamic_shapes["x"][1] = torch.export.Dim("result_edit")

    def test_region_config_validates_values(self):
        with self.assertRaisesRegex(TypeError, "rtol must be a number"):
            RegionConfig(rtol="1e-2")
        with self.assertRaisesRegex(TypeError, "atol must be a number"):
            RegionConfig(atol=True)
        with self.assertRaisesRegex(ValueError, "atol must be >= 0"):
            RegionConfig(atol=-1e-3)
        with self.assertRaisesRegex(ValueError, "rtol must be >= 0"):
            RegionConfig(rtol=float("nan"))
        with self.assertRaisesRegex(TypeError, "inductor_configs must be a dict"):
            RegionConfig(inductor_configs=["max_autotune"])
        with self.assertRaisesRegex(TypeError, "dict keyed by forward argument"):
            RegionConfig(dynamic_shapes={0: torch.export.Dim("batch")})
        with self.assertRaisesRegex(TypeError, "dict keyed by forward argument"):
            RegionConfig(dynamic_shapes=[{0: torch.export.Dim("batch")}])
        config = RegionConfig(rtol=0, inductor_configs={"aot_inductor.package": True})
        self.assertEqual(config.rtol, 0)

    def test_inductor_config_keys_are_left_to_compile_time(self):
        with mock.patch(
            "torch._inductor.codegen.common.init_backend_registration"
        ) as init:
            config = RegionConfig(
                inductor_configs={"max-autotune": True, "not_a_real_option": 1}
            )
        init.assert_not_called()
        self.assertEqual(
            config.inductor_configs, {"max-autotune": True, "not_a_real_option": 1}
        )
        with self.assertRaisesRegex(TypeError, "dict keyed by config name"):
            RegionConfig(inductor_configs={1: True})

    def test_discovery_computes_each_module_config_once(self):
        shared = PlainBlock()
        model = nn.Sequential(*[nn.Sequential(shared) for _ in range(4)])
        regional_aoti.mark_region(model[0])
        with mock.patch.object(
            regional_aoti,
            "_get_region_config",
            wraps=regional_aoti._get_region_config,
        ) as get_config:
            regional_aoti._find_marked_regions(model)
        unique_modules = len({id(m) for m in model.modules()})
        self.assertEqual(get_config.call_count, unique_modules)

    def test_region_config_repr_shows_unset(self):
        self.assertIn("rtol=_UNSET", repr(RegionConfig()))

    def test_mark_region_rejects_invalid_arguments(self):
        with self.assertRaisesRegex(TypeError, "config must be RegionConfig"):
            regional_aoti.mark_region(PlainBlock(), config={"rtol": 0.01})
        with self.assertRaisesRegex(TypeError, "expects an nn.Module"):
            regional_aoti.mark_region(object())
        with self.assertRaisesRegex(TypeError, "expects an nn.Module"):
            regional_aoti.mark_region(None)

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

        model = Aliased()
        (region,) = regional_aoti._find_marked_regions(model)
        self.assertEqual(region.name, "left")
        self.assertIs(region.module, model.left)
        self.assertEqual(region.aliases, ["left", "right"])

    def test_submodule_shared_by_two_regions_is_allowed(self):
        shared = PlainBlock()
        regional_aoti.mark_region(shared)
        first = nn.Sequential(shared)
        second = nn.Sequential(shared, nn.Linear(8, 8))
        regional_aoti.mark_region(first)
        regional_aoti.mark_region(second)
        model = nn.Sequential(first, second)

        regions = regional_aoti._find_marked_regions(model)

        self.assertEqual([r.name for r in regions], ["0", "1"])

    def test_discovery_handles_module_cycles(self):
        root = nn.Module()
        root.block = regional_aoti.mark_region(PlainBlock())
        root.block.back = root
        self.assertEqual(len(list(root.named_modules())), 3)

        (region,) = regional_aoti._find_marked_regions(root)
        self.assertEqual(region.aliases, ["block"])

    def test_discovery_handles_cycle_through_unmarked_module(self):
        root = nn.Module()
        root.a = nn.Module()
        root.a.back = root
        root.a.m = regional_aoti.mark_region(PlainBlock())

        (region,) = regional_aoti._find_marked_regions(root)
        self.assertEqual(region.aliases, ["a.m"])

    def test_self_cycle_does_not_hide_an_enclosing_region(self):
        # The region's own cycle reaches it before the enclosing region does.
        inner = regional_aoti.mark_region(PlainBlock())
        inner.me = inner
        outer = regional_aoti.mark_region(nn.Sequential(inner))
        root = nn.Module()
        root.outer = outer
        root.inner = inner

        with self.assertRaisesRegex(
            regional_aoti.AOTIRegionDiscoveryError, "crosses a selected-region boundary"
        ):
            regional_aoti._find_marked_regions(root)

    def test_discovery_skips_and_marking_rejects_jit_modules(self):
        example = torch.randn(1, 8)
        model = nn.Module()
        model.scripted = torch.jit.script(nn.Linear(8, 8))
        model.traced = torch.jit.trace(nn.Linear(8, 8), example)
        model.copied = copy.deepcopy(torch.jit.script(nn.Linear(8, 8)))
        self.assertEqual(regional_aoti._find_marked_regions(model), [])

        for target in (
            model.scripted,
            model.traced,
            model.copied,
            torch.jit.ScriptModule,
        ):
            with self.subTest(target=target):
                with self.assertRaisesRegex(TypeError, "TorchScript"):
                    regional_aoti.mark_region(target)

        model.good = GoodBlock()
        names = [r.name for r in regional_aoti._find_marked_regions(model)]
        self.assertEqual(names, ["good"])

    def test_discovery_visits_each_shared_module_once(self):
        # 2**30 paths lead to the leaf, but only 32 distinct modules exist.
        node = PlainBlock()
        for _ in range(30):
            parent = nn.Module()
            parent.left = node
            parent.right = node
            node = parent
        model = nn.Module()
        model.tower = node
        model.head = regional_aoti.mark_region(PlainBlock())
        with mock.patch.object(
            regional_aoti,
            "_get_region_config",
            wraps=regional_aoti._get_region_config,
        ) as get_config:
            regions = regional_aoti._find_marked_regions(model)
        self.assertEqual([r.name for r in regions], ["head"])
        self.assertEqual(get_config.call_count, len(list(model.modules())))

    def test_instance_mark_survives_pickle(self):
        dim_spec = {"x": {0: torch.export.Dim("batch")}}
        model = regional_aoti.mark_region(
            PlainBlock(), config=RegionConfig(rtol=1e-2, dynamic_shapes=dim_spec)
        )
        (region,) = regional_aoti._find_marked_regions(
            pickle.loads(pickle.dumps(model))
        )
        self.assertEqual(region.config.rtol, 1e-2)
        self.assertEqual(list(region.config.dynamic_shapes["x"]), [0])

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
