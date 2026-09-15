# Owner(s): ["module: dsl-native-ops"]
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from tools.native_aot import build_stage2, dependencies, export, gen_aot_lib
from tools.test import test_native_aot_tools as aot_fixtures

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torchgen.native_aot_decl import load_by_path


REPO = Path(__file__).resolve().parents[1]
reuse_old_whl = load_by_path(
    "reuse_old_whl", str(REPO / ".github/actions/reuse-old-whl/reuse_old_whl.py")
)


class TestNativeAotDependencies(TestCase):
    kernel = "torch/_native/ops/norm/kernel.py"
    vendor = "torch/_vendor/quack/helper.py"
    declaration = "torch/_native/ops/norm/aot.py"

    def setUp(self) -> None:
        super().setUp()
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.repo = Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        self.sources: dict[str, str] = {}
        for path in (self.kernel, self.vendor, self.declaration):
            source = self.repo / path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("VALUE = 1\n")
            self.sources[path] = dependencies.file_hash(source)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.repo)
        for name, value in (
            ("get_merge_base", "base"),
            ("get_head_sha", "head"),
            ("check_changed_files", True),
            ("check_labels_for_pr", False),
            ("check_issue_open", False),
            ("get_workflow_id", "workflow"),
            ("find_old_whl", True),
        ):
            self.context.enter_context(
                mock.patch.object(reuse_old_whl, name, return_value=value)
            )
        self.args = argparse.Namespace(
            github_ref="refs/pull/1/merge",
            run_id="run",
            build_environment="linux-cuda13.2-py3.12",
        )

    def candidate(self, manifest: bytes | None = None) -> bytes:
        if manifest is None:
            manifest = json.dumps({"version": 1, "sources": self.sources}).encode()
        with zipfile.ZipFile("artifacts.zip", "w") as archive:
            archive.writestr("dist/torch-2.0.0+githead.whl", b"old binary")
            if manifest:
                archive.writestr(str(dependencies.CI_MANIFEST), manifest)
        return manifest

    @parametrize(
        "relative,deleted",
        [(kernel, False), (vendor, True)],
    )
    def test_changed_dependency_rejects_candidate(
        self, relative: str, deleted: bool
    ) -> None:
        self.candidate()
        source = self.repo / relative
        if deleted:
            source.unlink()
        else:
            source.write_text("VALUE = 2\n")
        with (
            mock.patch.object(reuse_old_whl, "parse_args", return_value=self.args),
            mock.patch.object(reuse_old_whl, "emit_metric") as metric,
            mock.patch.object(
                reuse_old_whl, "unzip_artifact_and_replace_files"
            ) as overlay,
            mock.patch.object(reuse_old_whl, "set_output") as output,
        ):
            reuse_old_whl.main()
        result = metric.call_args.args[1]
        self.assertFalse(result["reuse_whl"])
        self.assertIn(relative, result["reason"])
        overlay.assert_not_called()
        output.assert_not_called()
        self.assertFalse(Path("artifacts.zip").exists())

    @parametrize(
        "manifest",
        [b"", b'{"version": 2, "sources": {}}'],
    )
    def test_missing_or_invalid_metadata_rebuilds(self, manifest: bytes) -> None:
        self.candidate(manifest)
        self.assertFalse(reuse_old_whl.can_reuse_whl(self.args)[0])
        self.assertFalse(Path("artifacts.zip").exists())

    def test_non_cuda_reuse_does_not_require_metadata(self) -> None:
        self.args.build_environment = "linux-cpu"
        self.candidate(b"")
        self.assertTrue(reuse_old_whl.can_reuse_whl(self.args)[0])

    def test_generation_records_only_selected_specs_and_arches(self) -> None:
        ops = self.repo / "ops/fakeop"
        ops.mkdir(parents=True)
        (ops / "aot.py").touch()
        artifacts = self.repo / "build/native_aot"
        for arch, point, source in (
            ("sm_90a", 1024, self.kernel),
            ("sm_100a", 1024, self.vendor),
            ("sm_100a", 2048, self.declaration),
            ("sm_100", 1024, "torch/_vendor/shadowed.py"),
            ("sm_80", 1024, "torch/_vendor/excluded.py"),
        ):
            tree = artifacts / arch / "fakeop"
            tree.mkdir(parents=True, exist_ok=True)
            prefix = f"k_{arch}_{point}"
            aot_fixtures._touch_artifacts(str(tree), prefix)
            sc = dict(
                aot_fixtures.SIDECAR,
                version=export.SIDECAR_VERSION,
                prefix=prefix,
                arch=arch,
                spec={"N": point, "K": 8},
                sources={source: self.sources.get(source, "0" * 16)},
                runtimes=export.runtime_versions("cutedsl"),
            )
            (tree / f"{prefix}.json").write_text(json.dumps(sc))
        with (
            mock.patch.object(export, "REPO", str(self.repo)),
            aot_fixtures._patched_generation(str(ops.parent)),
        ):
            gen_aot_lib.main(
                [
                    "--artifacts-dir",
                    str(artifacts),
                    "--archs",
                    "sm_90a",
                    "sm_100",
                    "sm_100a",
                ]
            )
        manifest = (artifacts / dependencies.MANIFEST).read_bytes()
        self.assertEqual(dependencies.read_manifest(manifest), self.sources)
        self.candidate(manifest)
        (self.repo / "torch/_vendor/unrelated.py").write_text("UNRELATED = True\n")
        self.assertTrue(reuse_old_whl.can_reuse_whl(self.args)[0])
        (self.repo / self.vendor).write_text("VALUE = 2\n")
        reusable, reason = reuse_old_whl.can_reuse_whl(self.args)
        self.assertFalse(reusable)
        self.assertIn(self.vendor, reason)

    def test_conflicting_dependencies_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, self.kernel):
            dependencies.merge_sources([self.sources, {self.kernel: "0" * 16}])

    def test_kernel_free_build_replaces_old_manifest_with_empty_dependencies(
        self,
    ) -> None:
        manifest = self.repo / dependencies.CI_MANIFEST
        dependencies.write_manifest(manifest, self.sources)
        wheel = self.repo / "torch.whl"
        wheel.touch()
        with (
            mock.patch.object(build_stage2, "REPO", str(self.repo)),
            mock.patch.object(build_stage2, "_opted_out", return_value=False),
            mock.patch.object(build_stage2, "_refuse_editable_rebuild"),
            mock.patch.object(build_stage2, "_torch_probe", return_value=True),
            mock.patch.object(build_stage2, "should_run", return_value=False),
            mock.patch.object(build_stage2, "_invalidate_stale_include"),
            mock.patch.object(build_stage2, "_torch_value", return_value="False"),
        ):
            self.assertEqual(build_stage2.main(["--wheel", str(wheel)]), 0)
        self.assertEqual(dependencies.read_manifest(manifest.read_bytes()), {})
        self.candidate(manifest.read_bytes())
        self.assertTrue(reuse_old_whl.can_reuse_whl(self.args)[0])

    def test_opt_out_removes_previous_manifest_without_probing(self) -> None:
        manifest = self.repo / dependencies.CI_MANIFEST
        dependencies.write_manifest(manifest, self.sources)
        with (
            mock.patch.object(build_stage2, "REPO", str(self.repo)),
            mock.patch.object(build_stage2, "_opted_out", return_value=True),
            mock.patch.object(build_stage2, "_torch_probe") as probe,
        ):
            self.assertEqual(build_stage2.main(["--wheel", "torch.whl"]), 0)
        self.assertFalse(manifest.exists())
        probe.assert_not_called()

    def test_installed_source_paths_and_hashes(self) -> None:
        package_path = self.repo / "venv/torch"
        source = package_path / "_vendor/quack/helper.py"
        source.parent.mkdir(parents=True)
        source.write_text("VALUE = 'installed, different from checkout'\n")
        package = types.ModuleType("torch")
        package.__file__ = str(package_path / "__init__.py")
        module = types.ModuleType("torch._vendor.quack.helper")
        module.__file__ = str(source)
        with (
            mock.patch.object(export, "REPO", str(self.repo)),
            mock.patch.object(export, "_HERE", str(self.repo / "tools/native_aot")),
            mock.patch.object(export, "_CLOSURE_PREFIXES", (module.__name__,)),
            mock.patch.dict(sys.modules, {"torch": package, module.__name__: module}),
        ):
            sources = export.source_closure()
        self.assertEqual(sources, {self.vendor: dependencies.file_hash(source)})
        self.assertEqual(dependencies.changed_source(sources, self.repo), self.vendor)


instantiate_parametrized_tests(TestNativeAotDependencies)


if __name__ == "__main__":
    run_tests()
