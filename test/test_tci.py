# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)
from torch.testing._internal.tci import _test, platform, schedule, size, test


ALL = frozenset(platform)
A100 = platform.linux_x86_cuda_a100
H100 = platform.linux_x86_cuda_h100


@instantiate_parametrized_tests
@test.default
class TestTci(TestCase):
    def test_default_config(self):
        self.assertEqual(test.default.platforms, ALL)
        self.assertEqual(test.default.schedule, schedule.pull)
        self.assertEqual(test.default.size, size.small)

    def test_markers_sorted_and_complete(self):
        # pin the platform set so the golden stays stable as platforms are added;
        # pass them out of order to exercise the deterministic sort
        config = test.default.on([H100, A100]).periodic.large
        self.assertEqual(
            config._markers(),
            [
                "tci.platform:linux_x86_cuda_a100",
                "tci.platform:linux_x86_cuda_h100",
                "tci.schedule:periodic",
                "tci.size:large",
            ],
        )

    def test_on_replaces(self):
        self.assertEqual(test.default.on([H100]).platforms, frozenset({H100}))

    def test_include_adds(self):
        config = test.default.on([H100]).include([A100])
        self.assertEqual(config.platforms, frozenset({A100, H100}))

    def test_exclude_removes(self):
        self.assertEqual(test.default.exclude([A100]).platforms, ALL - {A100})

    def test_only_filters_by_vendor(self):
        for vendor in ("cuda", "rocm", "xpu", "cpu"):
            expected = frozenset(p for p in platform if p.vendor == vendor)
            self.assertEqual(test.default.only(vendor).platforms, expected)
        self.assertEqual(test.default.only("cuda").platforms, frozenset({A100, H100}))

    @parametrize("member", [subtest(m, name=m.name) for m in schedule])
    def test_schedule_sugar(self, member):
        config = getattr(test.default, member.name)
        self.assertEqual(config.schedule, member)
        self.assertEqual(config.size, size.small)  # other fields untouched
        self.assertEqual(config.platforms, ALL)

    @parametrize("member", [subtest(m, name=m.name) for m in size])
    def test_size_sugar(self, member):
        config = getattr(test.default, member.name)
        self.assertEqual(config.size, member)
        self.assertEqual(config.schedule, schedule.pull)  # other fields untouched
        self.assertEqual(config.platforms, ALL)

    @parametrize("member", [subtest(p, name=p.name) for p in platform])
    def test_vendor(self, member):
        # name is linux_<cpu>_<vendor>_<product>; derive vendor independently
        self.assertEqual(member.vendor, member.name.split("_")[2])

    @parametrize(
        "transform",
        [
            subtest(lambda c: c.periodic, name="periodic"),
            subtest(lambda c: c.large, name="large"),
            subtest(lambda c: c.only_cuda, name="only_cuda"),
            subtest(lambda c: c.on([H100]), name="on"),
            subtest(lambda c: c.include([A100]), name="include"),
            subtest(lambda c: c.exclude([A100]), name="exclude"),
        ],
    )
    def test_transform_is_immutable(self, transform):
        base = test.default
        result = transform(base)
        self.assertIsInstance(result, _test)
        self.assertIsNot(result, base)
        # the shared preset must never be mutated by a transform
        self.assertEqual(base.platforms, ALL)
        self.assertEqual(base.schedule, schedule.pull)
        self.assertEqual(base.size, size.small)

    def test_presets_derive_from_default(self):
        self.assertEqual(test.cuda, test.default.only_cuda)
        self.assertEqual(test.rocm, test.default.only_rocm)
        self.assertEqual(test.xpu, test.default.only_xpu)
        self.assertEqual(test.cpu, test.default.only_cpu)

    def test_config_is_hashable(self):
        self.assertEqual(hash(test.cuda), hash(test.default.only_cuda))
        # equal configs dedupe; a differing field makes a distinct config
        self.assertEqual(len({test.default, test.default, test.default.periodic}), 2)

    def test_empty_selection_raises(self):
        def fn():
            pass

        with self.assertRaisesRegex(ValueError, "selects no platform"):
            test.default.on([])(fn)

    def test_apply_markers_to_function(self):
        @test.default.only_cuda.periodic
        def fn():
            pass

        names = {mark.name for mark in fn.pytestmark}
        self.assertEqual(
            names,
            {
                "tci.platform:linux_x86_cuda_a100",
                "tci.platform:linux_x86_cuda_h100",
                "tci.schedule:periodic",
                "tci.size:small",
            },
        )

    def test_apply_markers_to_class(self):
        @test.default
        class Decorated:
            pass

        names = {mark.name for mark in Decorated.pytestmark}
        self.assertEqual(names, set(test.default._markers()))


if __name__ == "__main__":
    run_tests()
