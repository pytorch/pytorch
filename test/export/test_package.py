# Owner(s): ["oncall: export"]
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._inductor.package import load_package
from torch.export import Dim
from torch.export.experimental import _ExportPackage
from torch.export.pt2_archive._package import _load_aoti
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestPackage(TestCase):
    def test_basic(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        self.assertEqual(
            package._exporter("fn", fn)(x),
            fn(x),
        )
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].fallbacks), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 0)

    def test_more_than_once(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn)
        exporter(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export .* more than once",
        ):
            exporter(x)

    def test_error(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="error")
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export fallback .* when fallback policy is set to 'error'",
        ):
            exporter(x)

    def test_overloads(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] == 4:
                    return x + 1
                elif x.shape[0] == 3:
                    return x - 1
                else:
                    return x + 2

        fn = Module()
        x = torch.randn(3, 2)
        x2 = torch.randn(4, 2)
        x3 = torch.randn(5, 2)

        def spec(self, x):
            assert x.shape[0] == 3  # noqa: S101

        def spec2(self, x):
            assert x.shape[0] == 4  # noqa: S101

        def spec3(self, x):
            assert x.shape[0] >= 5  # noqa: S101
            return {"x": (Dim("batch", min=5), Dim.STATIC)}

        package = _ExportPackage()
        exporter = (
            package._exporter("fn", fn)
            ._define_overload("spec", spec)
            ._define_overload("spec2", spec2)
            ._define_overload("spec3", spec3)
        )
        self.assertEqual(exporter(x), x - 1)
        self.assertEqual(exporter(x2), x2 + 1)
        self.assertEqual(exporter(x3), x3 + 2)
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 3)


class TestAOTIPackageDeviceValidation(TestCase):
    def test_aoti_load_uses_cpp_device_validation_before_device_info(self):
        class FakeAOTIModelPackageLoader:
            @staticmethod
            def load_metadata_from_package(file, model_name):
                return {"AOTI_DEVICE_KEY": "cuda"}

            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "Cannot load AOTInductor package for device 'cuda' because "
                    "CUDA is not available in this process."
                )

        with (
            mock.patch.object(
                torch._C._aoti, "AOTIModelPackageLoader", FakeAOTIModelPackageLoader
            ),
            mock.patch("torch._inductor.codecache.get_device_information") as get_info,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot load AOTInductor package.*CUDA is not available",
            ):
                _load_aoti("model.pt2", "model", False, 1, -1)

            get_info.assert_not_called()

    def test_load_package_does_not_fallback_on_device_validation_error(self):
        class FakeAOTIModelPackageLoader:
            def __init__(self, *args, **kwargs):
                raise AssertionError("AOTI fallback loader should not be constructed")

        with (
            mock.patch(
                "torch._inductor.package.package.load_pt2",
                side_effect=RuntimeError(
                    "Cannot load AOTInductor package for device 'cuda' because "
                    "CUDA is not available in this process."
                ),
            ),
            mock.patch.object(
                torch._C._aoti, "AOTIModelPackageLoader", FakeAOTIModelPackageLoader
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot load AOTInductor package.*CUDA is not available",
            ):
                load_package("model.pt2")

    @unittest.skipIf(
        torch.cuda.is_available(), "requires CUDA to be unavailable in this process"
    )
    def test_cpp_aoti_package_loader_validates_cuda_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "data" / "aotinductor" / "model"
            model_dir.mkdir(parents=True)
            extension = ".pyd" if sys.platform == "win32" else ".so"
            (model_dir / f"model{extension}").touch()
            (model_dir / "model_metadata.json").write_text(
                json.dumps({"AOTI_DEVICE_KEY": "cuda"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot load AOTInductor package.*(CUDA|ROCm) is not available",
            ):
                torch._C._aoti.AOTIModelPackageLoader(str(tmp), "model", False, 1, -1)

    def test_aoti_load_does_not_invoke_compiler_isa_probe(self):
        from torch._inductor import config as inductor_config, cpu_vec_isa

        class FakeAOTIModelPackageLoader:
            @staticmethod
            def load_metadata_from_package(file, model_name):
                return {"AOTI_DEVICE_KEY": "cpu", "AOTI_CPU_ISA": "AVX2"}

            def __init__(self, *args, **kwargs):
                pass

        def clear_isa_caches():
            cpu_vec_isa.valid_vec_isa_list.cache_clear()
            cpu_vec_isa.cpuinfo_vec_isa_list.cache_clear()
            cpu_vec_isa.VecISA._VecISA__bool__impl.cache_clear()
            cpu_vec_isa.VecAVX512.__bool__.cache_clear()
            cpu_vec_isa.VecAVX512VNNI.__bool__.cache_clear()
            cpu_vec_isa.VecAMX.__bool__.cache_clear()

        clear_isa_caches()
        try:
            with (
                inductor_config.patch({"cpp.vec_isa_ok": None, "cpp.simdlen": None}),
                mock.patch.object(
                    torch._C._aoti, "AOTIModelPackageLoader", FakeAOTIModelPackageLoader
                ),
                mock.patch.object(
                    cpu_vec_isa.VecISA,
                    "check_build",
                    side_effect=AssertionError(
                        "the load path must not dry-compile ISA probe programs"
                    ),
                ),
            ):
                _load_aoti("model.pt2", "model", False, 1, -1)
        finally:
            clear_isa_caches()

    def test_aoti_load_still_warns_on_isa_mismatch(self):
        class FakeAOTIModelPackageLoader:
            @staticmethod
            def load_metadata_from_package(file, model_name):
                # An ISA string no host can satisfy, so the mismatch warning
                # must fire regardless of the machine running the test.
                return {
                    "AOTI_DEVICE_KEY": "cpu",
                    "AOTI_CPU_ISA": "AVX512 AVX512_VNNI AMX_TILE NONEXISTENT_ISA",
                }

            def __init__(self, *args, **kwargs):
                pass

        with (
            mock.patch.object(
                torch._C._aoti, "AOTIModelPackageLoader", FakeAOTIModelPackageLoader
            ),
            mock.patch("torch.export.pt2_archive._package.logger.warning") as mock_warn,
        ):
            _load_aoti("model.pt2", "model", False, 1, -1)

        mock_warn.assert_called_once()
        self.assertIn("Device information mismatch", mock_warn.call_args[0][0])

    def test_cpuinfo_vec_isa_list_covers_valid_vec_isa_list(self):
        from torch._inductor import cpu_vec_isa

        # The cpuinfo-detected list may only ever be a (non-strict) superset
        # of the toolchain-validated list: the dry-compile probe can remove
        # ISAs the compiler cannot build for, never add them.
        valid = {str(isa) for isa in cpu_vec_isa.valid_vec_isa_list()}
        detected = {str(isa) for isa in cpu_vec_isa.cpuinfo_vec_isa_list()}
        self.assertTrue(
            valid.issubset(detected), f"{valid=} not a subset of {detected=}"
        )


if __name__ == "__main__":
    run_tests()
