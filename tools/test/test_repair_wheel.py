# Owner(s): ["module: binaries"]

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".ci/wheel/linux/repair_wheel.py"


class TestRepairWheel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("repair_wheel", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {SCRIPT}")
        cls.repair = importlib.util.module_from_spec(spec)
        wheeltools = mock.Mock(InWheelCtx=mock.MagicMock(), add_platforms=mock.Mock())
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "repair_wheel": cls.repair,
                    "auditwheel": ModuleType("auditwheel"),
                    "auditwheel.wheeltools": wheeltools,
                },
            ),
            mock.patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]),
        ):
            spec.loader.exec_module(cls.repair)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.wheel = self.make_wheel()
        self.enterContext(mock.patch.dict(os.environ, {}, clear=True))
        self.enterContext(
            mock.patch.object(self.repair.platform, "machine", return_value="x86_64")
        )
        self.patchelf = self.enterContext(mock.patch.object(self.repair, "patchelf"))
        self.default_copier = self.enterContext(
            mock.patch.object(self.repair, "copy_runtime_dependencies", autospec=True)
        )
        self.default_copier.return_value = None
        self.contexts = []
        self.wheel_context = self.enterContext(
            mock.patch.object(self.repair, "InWheelCtx", side_effect=self.make_context)
        )
        self.add_platforms = self.enterContext(
            mock.patch.object(self.repair, "add_platforms")
        )

    def make_context(self, wheel: Path, output: Path):
        path = self.root / f"unpacked-{len(self.contexts)}"
        (path / "torch/lib").mkdir(parents=True)
        (path / "torch/_C.so").touch()
        (path / "torch/lib/libtorch.so").touch()
        ctx = mock.MagicMock(path=path, out_wheel=output)
        ctx.__enter__.return_value = ctx
        ctx.__exit__.return_value = False
        self.contexts.append(ctx)
        return ctx

    def make_wheel(self, version: str = "1.0", tag: str = "linux_x86_64") -> Path:
        wheel = self.input_dir / f"torch-{version}-cp311-cp311-{tag}.whl"
        wheel.touch()
        return wheel

    def run_main(self) -> None:
        with mock.patch.object(
            sys, "argv", [str(SCRIPT), str(self.input_dir), str(self.output_dir)]
        ):
            self.repair.main()

    def install_hook(self, body: str) -> Path:
        hook = self.root / "hook.py"
        hook.write_text(body)
        os.environ.update(
            PYTORCH_RUNTIME_DEPENDENCY_POLICY="custom",
            PYTORCH_CUSTOM_WHEEL_DEPENDENCY_HOOK=str(hook),
        )
        return hook

    def test_default_ignores_custom_configuration(self) -> None:
        os.environ.update(
            PYTORCH_CUSTOM_WHEEL_DEPENDENCY_HOOK="/nonexistent/hook.py",
            PYTORCH_CUSTOM_WHEEL_PLATFORM_TAG="invalid-tag",
        )
        with mock.patch.object(self.repair.runpy, "run_path") as load_hook:
            self.run_main()
        load_hook.assert_not_called()
        self.default_copier.assert_called_once()
        self.add_platforms.assert_called_once_with(
            self.contexts[0],
            ["manylinux_2_28_x86_64"],
            remove_platforms=["linux_x86_64"],
        )
        self.assertIsNotNone(self.contexts[0].out_wheel)

    def test_custom_platform_tags(self) -> None:
        self.install_hook("def copy_runtime_dependencies(*args):\n    pass\n")
        for tag in ("linux_x86_64", "manylinux_2_28_x86_64"):
            with self.subTest(tag=tag):
                os.environ["PYTORCH_CUSTOM_WHEEL_PLATFORM_TAG"] = f" {tag} "
                self.run_main()
                self.add_platforms.assert_called_once_with(
                    self.contexts[-1],
                    [tag],
                    remove_platforms=[] if tag == "linux_x86_64" else ["linux_x86_64"],
                )
                self.assertIsNotNone(self.contexts[-1].out_wheel)
                self.add_platforms.reset_mock()

    def test_retag_removes_other_platform_tags(self) -> None:
        self.wheel.unlink()
        self.make_wheel(tag="linux_x86_64.manylinux_2_28_x86_64")
        self.install_hook("def copy_runtime_dependencies(*args):\n    pass\n")
        os.environ["PYTORCH_CUSTOM_WHEEL_PLATFORM_TAG"] = "manylinux_2_28_x86_64"
        self.run_main()
        self.add_platforms.assert_called_once_with(
            self.contexts[0],
            ["manylinux_2_28_x86_64"],
            remove_platforms=["linux_x86_64"],
        )

    def test_invalid_platform_tag_prevents_output(self) -> None:
        self.install_hook("def copy_runtime_dependencies(*args):\n    pass\n")
        for tag in ("", " ", "linux-x86_64", "linux_x86_64.other", "linux_\u00e9"):
            with self.subTest(tag=tag):
                os.environ["PYTORCH_CUSTOM_WHEEL_PLATFORM_TAG"] = tag
                with self.assertRaisesRegex(
                    ValueError, "PYTORCH_CUSTOM_WHEEL_PLATFORM_TAG"
                ):
                    self.run_main()
                self.assertIsNone(self.contexts[-1].out_wheel)
                self.add_platforms.assert_not_called()

    def test_invalid_policy(self) -> None:
        for policy in ("", "unknown"):
            with self.subTest(policy=policy):
                os.environ["PYTORCH_RUNTIME_DEPENDENCY_POLICY"] = policy
                with self.assertRaisesRegex(
                    ValueError, "Invalid PYTORCH_RUNTIME_DEPENDENCY_POLICY"
                ):
                    self.run_main()
        self.default_copier.assert_not_called()

    def test_custom_requires_hook(self) -> None:
        os.environ["PYTORCH_RUNTIME_DEPENDENCY_POLICY"] = "custom"
        with self.assertRaisesRegex(ValueError, "DEPENDENCY_HOOK is required"):
            self.run_main()
        self.default_copier.assert_not_called()

    def test_invalid_hook_interface(self) -> None:
        cases = (
            ("copy_runtime_dependencies = None\n", "must define a callable"),
            (
                "def copy_runtime_dependencies(one):\n    pass\n",
                "must accept the positional arguments",
            ),
            (
                "def copy_runtime_dependencies(*args, required):\n    pass\n",
                "must accept the positional arguments",
            ),
        )
        for body, message in cases:
            with self.subTest(body=body):
                self.install_hook(body)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_main()
                self.wheel_context.assert_not_called()
        self.default_copier.assert_not_called()

    def test_hook_failure_prevents_output(self) -> None:
        for statement, error, message in (
            ("return False", TypeError, "must return None"),
            ("raise RuntimeError('copy failed')", RuntimeError, "copy failed"),
        ):
            with self.subTest(statement=statement):
                self.install_hook(
                    f"def copy_runtime_dependencies(*args):\n    {statement}\n"
                )
                with self.assertRaisesRegex(error, message):
                    self.run_main()
                self.assertIsNone(self.contexts[-1].out_wheel)
        self.default_copier.assert_not_called()

    def test_rpath_failure_prevents_output(self) -> None:
        self.install_hook("def copy_runtime_dependencies(*args):\n    pass\n")
        self.patchelf.side_effect = RuntimeError("patchelf failed")
        with self.assertRaisesRegex(RuntimeError, "patchelf failed"):
            self.run_main()
        self.assertIsNone(self.contexts[0].out_wheel)

    def test_failure_after_retag_prevents_output(self) -> None:
        def fail_after_retag(ctx, platforms, remove_platforms):
            ctx.out_wheel = (
                self.output_dir / "torch-1.0-cp311-cp311-manylinux_2_28_x86_64.whl"
            )
            raise RuntimeError("retag failed")

        with mock.patch.object(
            self.repair, "add_platforms", side_effect=fail_after_retag
        ):
            with self.assertRaisesRegex(RuntimeError, "retag failed"):
                self.run_main()
        self.assertIsNone(self.contexts[0].out_wheel)


if __name__ == "__main__":
    unittest.main()
