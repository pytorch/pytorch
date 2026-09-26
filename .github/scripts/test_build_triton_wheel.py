#!/usr/bin/env python3
from unittest import main, SkipTest, TestCase

from build_triton_wheel import PYTORCH_LOCAL_VERSION, wheel_version_suffix


class TestWheelVersionSuffix(TestCase):
    def test_release_cuda_wheel_gets_local_version(self) -> None:
        self.assertEqual(
            wheel_version_suffix(package_name="triton", release=True),
            f"+{PYTORCH_LOCAL_VERSION}",
        )

    def test_nightly_unchanged(self) -> None:
        self.assertEqual(wheel_version_suffix(package_name="triton", release=False), "")

    def test_rocm_xpu_unchanged(self) -> None:
        for package_name in ("triton-rocm", "triton-xpu"):
            with self.subTest(package_name=package_name):
                self.assertEqual(
                    wheel_version_suffix(package_name=package_name, release=True), ""
                )

    def test_local_version_matches_torch_pin(self) -> None:
        # torch pins triton==X.Y.Z
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version
        except ImportError as e:
            raise SkipTest("packaging is not installed") from e

        suffix = wheel_version_suffix(package_name="triton", release=True)
        built = Version(f"3.5.0{suffix}")
        self.assertEqual(built.local, PYTORCH_LOCAL_VERSION)
        self.assertIn(built, SpecifierSet("==3.5.0"))
        self.assertIn(built, SpecifierSet(">=3.4,<3.6"))


if __name__ == "__main__":
    main()
