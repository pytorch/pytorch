import importlib
import tempfile
import sys
from pathlib import Path
from torch._dynamo.test_case import TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

module_name = "torch._dynamo.einops_patch"


def import_einops_patch():
    for module in list(sys.modules):
        if module == module_name:
            sys.modules.pop(module, None)

    importlib.invalidate_caches()

    return importlib.import_module(module_name)


def fake_einops_package(
    temp_path: Path, version: str, *, missing_uncached=False, missing_wrapped=False
):
    package_directory = temp_path / "einops"
    package_directory.mkdir()

    (package_directory / "__init__.py").write_text(f"__version__ = {version!r}\n")
    einops_py = package_directory / "einops.py"

    lines = []
    lines += [
        "def A(): return 'cached'\n",
        "def B(): return 'uncached'\n",
    ]
    lines += ["_reconstruct_from_shape = A\n"]
    lines += [
        "def _inner(): return 'inner'\n",
        "def _outer(): return 'outer'\n",
    ]
    lines += ["_prepare_transformation_recipe = _outer\n"]

    if not missing_uncached:
        lines += ["_reconstruct_from_shape_uncached = B\n"]
    if not missing_wrapped:
        lines += ["_outer.__wrapped__ = _inner\n"]

    einops_py.write_text("".join(lines))
    return package_directory.parent


class TestEinopsPatch(TestCase):
    def setUp(self):
        super().setUp()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.einops_patch_module = import_einops_patch()

    def tearDown(self):
        for module in list(sys.modules):
            if module == "einops" or module.startswith("einops."):
                sys.modules.pop(module, None)

        importlib.invalidate_caches()

        super().tearDown()

    def test_no_eager_import(self):
        self.einops_patch_module._patch_einops()

        assert "einops" not in sys.modules

    @parametrize("ver", ["0.6.1", "0.6.2rc0", "0.7.0rc1"])
    def test_target_versions_are_patched(self, ver):
        sys.path.insert(0, str(fake_einops_package(self.temp_path, ver)))

        self.einops_patch_module._patch_einops()

        importlib.import_module("einops")  # triggers _EinopsImportInterceptor
        einops_module = importlib.import_module("einops.einops")

        assert (
            einops_module._reconstruct_from_shape
            is einops_module._reconstruct_from_shape_uncached
        )
        assert (
            getattr(einops_module._prepare_transformation_recipe, "__name__", "")
            == "_inner"
        )
        assert not any(
            isinstance(f, self.einops_patch_module._EinopsImportInterceptor)
            for f in sys.meta_path
        )
        assert self.einops_patch_module._EINOPS_NEEDS_PATCH is False

    def test_non_target_version_unaffected(self):
        sys.path.insert(0, str(fake_einops_package(self.temp_path, "0.7.0rc2")))

        self.einops_patch_module._patch_einops()

        importlib.import_module("einops")
        einops_module = importlib.import_module("einops.einops")

        assert einops_module._reconstruct_from_shape.__name__ == "A"
        assert (
            getattr(einops_module, "_reconstruct_from_shape_uncached", None).__name__
            == "B"
        )
        assert (
            getattr(einops_module._prepare_transformation_recipe, "__name__", "")
            == "_outer"
        )
        assert not any(
            isinstance(f, self.einops_patch_module._EinopsImportInterceptor)
            for f in sys.meta_path
        )
        assert self.einops_patch_module._EINOPS_NEEDS_PATCH is False

    def test_already_imported_target_patches_immediately(self):
        sys.path.insert(0, str(fake_einops_package(self.temp_path, "0.6.1")))
        importlib.import_module("einops")  # import before patch

        self.einops_patch_module._patch_einops()
        einops_module = importlib.import_module("einops.einops")

        assert (
            einops_module._reconstruct_from_shape
            is einops_module._reconstruct_from_shape_uncached
        )
        assert (
            getattr(einops_module._prepare_transformation_recipe, "__name__", "")
            == "_inner"
        )
        assert not any(
            isinstance(f, self.einops_patch_module._EinopsImportInterceptor)
            for f in sys.meta_path
        )

    def test_already_imported_non_target_unaffected(self):
        sys.path.insert(0, str(fake_einops_package(self.temp_path, "0.8.0")))
        importlib.import_module("einops")

        self.einops_patch_module._patch_einops()
        einops_module = importlib.import_module("einops.einops")

        assert einops_module._reconstruct_from_shape.__name__ == "A"
        assert (
            getattr(einops_module._prepare_transformation_recipe, "__name__", "")
            == "_outer"
        )
        assert not any(
            isinstance(f, self.einops_patch_module._EinopsImportInterceptor)
            for f in sys.meta_path
        )

    def test_idempotent(self):
        sys.path.insert(0, str(fake_einops_package(self.temp_path, "0.6.1")))

        self.einops_patch_module._patch_einops()
        self.einops_patch_module._patch_einops()

        assert (
            sum(
                isinstance(f, self.einops_patch_module._EinopsImportInterceptor)
                for f in sys.meta_path
            )
            == 1
        )

        importlib.import_module("einops")

        einops_module = importlib.import_module("einops.einops")
        first = (
            einops_module._reconstruct_from_shape,
            einops_module._prepare_transformation_recipe,
        )

        self.einops_patch_module._patch_einops()

        einops_module_2 = importlib.import_module("einops.einops")
        second = (
            einops_module_2._reconstruct_from_shape,
            einops_module_2._prepare_transformation_recipe,
        )

        assert first == second
        assert not any(
            isinstance(f, self.einops_patch_module._EinopsImportInterceptor)
            for f in sys.meta_path
        )

    @parametrize(
        "kwargs",
        [
            dict(missing_uncached=True, missing_wrapped=False),
            dict(missing_uncached=False, missing_wrapped=True),
        ],
    )
    def test_patch_even_if_some_parts_are_missing(self, kwargs):
        sys.path.insert(0, str(fake_einops_package(self.temp_path, "0.6.1", **kwargs)))

        self.einops_patch_module._patch_einops()

        importlib.import_module("einops")
        einops_module = importlib.import_module("einops.einops")

        if not kwargs.get("missing_uncached"):
            assert (
                einops_module._reconstruct_from_shape
                is einops_module._reconstruct_from_shape_uncached
            )

        if not kwargs.get("missing_wrapped"):
            assert (
                getattr(einops_module._prepare_transformation_recipe, "__name__", "")
                == "_inner"
            )

        assert not any(
            isinstance(f, self.einops_patch_module._EinopsImportInterceptor)
            for f in sys.meta_path
        )
        assert self.einops_patch_module._EINOPS_NEEDS_PATCH is False

    def test_cleanup_interceptor_even_on_exception(self):
        package_directory = self.temp_path / "einops"
        package_directory.mkdir()
        (package_directory / "__init__.py").write_text("__version__='0.6.1'\n")
        (package_directory / "einops.py").write_text("raise RuntimeError('Error')\n")
        sys.path.insert(0, str(self.temp_path))
        self.einops_patch_module._patch_einops()

        importlib.import_module("einops")
        assert not any(
            isinstance(f, self.einops_patch_module._EinopsImportInterceptor)
            for f in sys.meta_path
        )
        assert self.einops_patch_module._EINOPS_NEEDS_PATCH is False


instantiate_parametrized_tests(
    TestEinopsPatch,
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
