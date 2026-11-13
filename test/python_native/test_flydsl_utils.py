# Owner(s): ["module: dsl-native-ops"]

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._native import flydsl_utils
from torch._vendor.packaging.version import Version
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


_QUERY_RAISES = object()


class TestFlyDSLArchResolution(TestCase):
    """What the kernel gets compiled for.

    Compiling for the wrong arch is silent -- a wave64 reduction on a wave32
    device produces wrong numbers rather than an error -- so each source of the
    answer is pinned here.
    """

    def setUp(self):
        super().setUp()
        flydsl_utils._resolve_rocm_arch.cache_clear()
        flydsl_utils._get_flydsl_device_arch.cache_clear()
        self.addCleanup(flydsl_utils._resolve_rocm_arch.cache_clear)
        self.addCleanup(flydsl_utils._get_flydsl_device_arch.cache_clear)

    def _resolve(self, *, flydsl_arch="", hsa="", props=_QUERY_RAISES):
        """Resolve with the environment and the device query both controlled.

        ``props=None`` means no CUDA device; the default means the query itself
        raises.
        """
        env = {"FLYDSL_GPU_ARCH": flydsl_arch, "HSA_OVERRIDE_GFX_VERSION": hsa}
        query = (
            RuntimeError("no device") if props is _QUERY_RAISES else None,
            props,
        )
        with (
            patch.dict(os.environ, env),
            patch.object(torch.cuda, "is_available", return_value=props is not None),
            patch.object(
                torch.cuda,
                "get_device_properties",
                side_effect=query[0],
                return_value=query[1],
            ),
        ):
            return flydsl_utils._resolve_rocm_arch(0)

    def test_explicit_arch_env_wins_and_is_stripped(self):
        # Set on both, so this also pins the precedence over the HSA override.
        self.assertEqual(
            self._resolve(flydsl_arch="gfx950:sramecc+", hsa="9.0.10"), "gfx950"
        )

    def test_hsa_override_gfx_form_is_stripped(self):
        self.assertEqual(self._resolve(hsa="gfx950:sramecc+"), "gfx950")

    def test_hsa_override_stepping_is_hexadecimal(self):
        # 9.0.10 is gfx90a, not gfx9010 -- the one rule here that is easy to
        # get wrong by reading the format as decimal.
        self.assertEqual(self._resolve(hsa="9.0.10"), "gfx90a")

    @parametrize("hsa", ("9.0", "9.0.x", "not-a-version"))
    def test_unusable_hsa_override_falls_back_to_the_device(self, hsa):
        props = SimpleNamespace(gcnArchName="gfx942:xnack-")
        self.assertEqual(self._resolve(hsa=hsa, props=props), "gfx942")

    def test_device_properties_are_stripped(self):
        props = SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        self.assertEqual(self._resolve(props=props), "gfx950")

    def test_no_cuda_device_returns_none(self):
        self.assertIsNone(self._resolve(props=None))

    def test_device_query_failure_returns_none(self):
        # get_device_properties raising must decline rather than propagate:
        # this runs inside the dispatcher predicate.
        self.assertIsNone(self._resolve())

    def test_missing_gcn_arch_name_returns_none(self):
        self.assertIsNone(self._resolve(props=SimpleNamespace()))

    def test_resolution_is_cached_per_device(self):
        props = SimpleNamespace(gcnArchName="gfx942")
        self.assertEqual(self._resolve(props=props), "gfx942")
        self.assertEqual(self._resolve(flydsl_arch="gfx950"), "gfx942")
        flydsl_utils._resolve_rocm_arch.cache_clear()
        self.assertEqual(self._resolve(flydsl_arch="gfx950"), "gfx950")


class TestFlyDSLSharedPredicates(TestCase):
    def setUp(self):
        super().setUp()
        flydsl_utils._is_supported_arch.cache_clear()
        self.addCleanup(flydsl_utils._is_supported_arch.cache_clear)

    @parametrize(
        "arch,supported_arches,expected",
        (
            ("gfx950", ("gfx950",), True),
            ("gfx942", ("gfx950",), False),
            ("gfx942", ("gfx942", "gfx950"), True),
            (None, ("gfx950",), False),
        ),
    )
    def test_is_supported_arch(self, arch, supported_arches, expected):
        with patch.object(flydsl_utils, "_resolve_rocm_arch", return_value=arch):
            self.assertEqual(
                flydsl_utils._is_supported_arch(0, supported_arches), expected
            )

    def test_is_supported_arch_is_cached(self):
        with patch.object(
            flydsl_utils,
            "_resolve_rocm_arch",
            side_effect=("gfx950", "gfx942"),
        ) as resolve:
            self.assertTrue(flydsl_utils._is_supported_arch(0, ("gfx950",)))
            self.assertTrue(flydsl_utils._is_supported_arch(0, ("gfx950",)))
            self.assertEqual(resolve.call_count, 1)

    @parametrize(
        "rows_m,n,itemsize,expected",
        (
            (2048, 114688, 4, True),
            (16383, 65536, 4, True),
            (16384, 65536, 4, False),
            (16385, 65536, 4, False),
            (1 << 31, 1, 1, False),
            (1, 1 << 31, 1, False),
        ),
    )
    def test_fits_int32_buffer_span(self, rows_m, n, itemsize, expected):
        self.assertEqual(
            flydsl_utils._fits_int32_buffer_span(rows_m, n, itemsize), expected
        )


class TestFlyDSLVersionGate(TestCase):
    """The gate that decides whether FlyDSL overrides register at all.

    Every case has to clear the cache twice: ``_version_is_ok`` is
    ``functools.cache``d, so a stale verdict would leak into the next case and
    a real verdict from this machine's install would leak into the first.
    """

    def setUp(self):
        super().setUp()
        flydsl_utils._version_is_ok.cache_clear()
        self.addCleanup(flydsl_utils._version_is_ok.cache_clear)

    def _with_version(self, version):
        return patch.object(
            flydsl_utils,
            "_check_runtime_available",
            return_value=(True, version),
        )

    @parametrize("version", ("0.3.0", "0.3.1", "0.3.5", "0.3.0.post1"))
    def test_stable_supported_releases_are_accepted(self, version):
        with self._with_version(Version(version)):
            self.assertTrue(flydsl_utils._version_is_ok())

    @parametrize(
        "version",
        ("0.2.9", "0.3.0.dev765", "0.3.0rc1", "0.4.0", "1.3.0"),
    )
    def test_other_versions_are_rejected(self, version):
        with self._with_version(Version(version)):
            self.assertFalse(flydsl_utils._version_is_ok())

    def test_missing_version_is_rejected(self):
        with (
            self._with_version(None),
            self.assertLogs("torch._native.flydsl_utils", level="INFO") as logs,
        ):
            self.assertFalse(flydsl_utils._version_is_ok())

        self.assertIn("version None is not supported", "\n".join(logs.output))

    def test_unsupported_version_reports_the_version(self):
        with (
            self._with_version(Version("0.4.0")),
            self.assertLogs("torch._native.flydsl_utils", level="INFO") as logs,
        ):
            self.assertFalse(flydsl_utils._version_is_ok())

        self.assertIn("0.4.0 is not supported", "\n".join(logs.output))

    def test_skip_check_overrides_unsupported_version(self):
        with (
            self._with_version(Version("0.4.0")),
            patch.object(flydsl_utils, "check_native_version_skip", return_value=True),
        ):
            self.assertTrue(flydsl_utils._version_is_ok())

    def test_skip_check_overrides_missing_version(self):
        with (
            self._with_version(None),
            patch.object(flydsl_utils, "check_native_version_skip", return_value=True),
        ):
            self.assertTrue(flydsl_utils._version_is_ok())


instantiate_parametrized_tests(TestFlyDSLArchResolution)
instantiate_parametrized_tests(TestFlyDSLSharedPredicates)
instantiate_parametrized_tests(TestFlyDSLVersionGate)


if __name__ == "__main__":
    run_tests()
