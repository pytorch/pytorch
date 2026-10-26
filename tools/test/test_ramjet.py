from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import shutil
import sys
import tarfile
import tempfile
import textwrap
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import ramjet

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIRST = "test/test_first.py"
SECOND = "test/nested/test_second.py"
MARKED = "Suite::test_marked"
SPECIFIC = "Another::test_specific"


@instantiate_parametrized_tests
class TestRamjet(TestCase):
    root: Path

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-tests-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        (cls.root / "test/nested").mkdir(parents=True)
        for path in ("pytest.ini", "test/conftest.py", "test/pytest_shard_custom.py"):
            shutil.copyfile(REPO_ROOT / path, cls.root / path)
        fixture = textwrap.dedent(
            """\
            from torch.testing._internal import tci
            from torch.testing._internal.common_utils import run_tests, TestCase


            @tci.test(platform=())
            class Suite(TestCase):
                @tci.test(platform=tci.platform.linux_x86_64_cuda_t4)
                def test_marked(self):
                    self.assertEqual(__name__, "__main__")

                def test_excluded(self):
                    self.fail("Ramjet executed an excluded test")


            class Another(TestCase):
                @tci.test(platform=tci.platform.linux_x86_64_cuda_t4)
                def test_specific(self):
                    self.assertEqual(__name__, "__main__")


            class Defaults(TestCase):
                def test_inherited(self):
                    self.assertEqual(__name__, "__main__")

                @tci.test(exclude=tci.platform.linux_x86_64)
                def test_cuda_only(self):
                    self.assertEqual(__name__, "__main__")

                @tci.test(exclude=tci.platform.linux_x86_64_cuda_l4)
                def test_cpu_only(self):
                    self.assertEqual(__name__, "__main__")


            if __name__ == "__main__":
                run_tests()
            """
        )
        for path in (FIRST, SECOND):
            (cls.root / path).write_text(fixture)
        (cls.root / "outside.py").write_text(fixture)

    @parametrize(
        "target", [FIRST, f"{FIRST}::Suite", f"{FIRST}::{SPECIFIC}", "test/nested"]
    )
    def test_normalize_targets(self, target):
        self.assertEqual(
            ramjet.normalize_targets([target], self.root, cwd=self.root), [target]
        )
        self.assertEqual(
            ramjet.normalize_targets([f"{self.root}/{target}"], self.root), [target]
        )

    @parametrize("targets", [[], ["."]])
    @parametrize(
        "directory,expected",
        [(".", "test"), ("test", "test"), ("test/nested", "test/nested")],
    )
    def test_current_directory(self, targets, directory, expected):
        self.assertEqual(
            ramjet.normalize_targets(targets, self.root, cwd=self.root / directory),
            [expected],
        )

    def test_relative_selector_from_nested_directory(self):
        target = f"test_second.py::{SPECIFIC}"
        self.assertEqual(
            ramjet.normalize_targets(
                [target], self.root, cwd=self.root / "test/nested"
            ),
            [f"{SECOND}::{SPECIFIC}"],
        )

    def test_current_directory_outside_repository(self):
        with self.assertRaises(ValueError):
            ramjet.normalize_targets([], self.root, cwd=self.root.parent)

    @parametrize(
        "target",
        ["test/missing.py", "outside.py", "test/nested::Suite", f"{FIRST}::"],
    )
    def test_invalid_paths(self, target):
        with self.assertRaises(ValueError):
            ramjet.normalize_targets([FIRST, target], self.root, cwd=self.root)

    def test_collection_groups_overlapping_targets_by_file(self):
        targets = [
            f"{FIRST}::Suite",
            "test",
            f"{FIRST}::Suite",
            f"{SECOND}::{SPECIFIC}",
        ]
        self.assertCountEqual(
            ramjet.collection_targets(targets, root=str(self.root)),
            [
                [FIRST, f"{FIRST}::Suite"],
                [SECOND, f"{SECOND}::{SPECIFIC}"],
            ],
        )

    def test_collection_preserves_explicit_selectors(self):
        targets = [f"{FIRST}::{SPECIFIC}", f"{FIRST}::MissingSuite"]
        self.assertEqual(
            ramjet.collection_targets(targets, root=str(self.root)),
            [sorted(targets)],
        )

    def test_collection_rejects_empty_directory(self):
        empty = self.root / "test/empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No test files"):
            ramjet.collection_targets(["test/empty"], root=str(self.root))

    @parametrize(
        "targets,expected",
        [
            ([FIRST], [f"{FIRST}::{MARKED}", f"{FIRST}::{SPECIFIC}"]),
            (
                [f"{FIRST}::Suite", f"{SECOND}::{SPECIFIC}"],
                [f"{FIRST}::{MARKED}", f"{SECOND}::{SPECIFIC}"],
            ),
        ],
    )
    def test_collect_selections(self, targets, expected):
        markers = ramjet.collect_markers(targets, root=str(self.root))
        self.assertEqual(ramjet.select_tests(markers, ("t4",))["t4"], sorted(expected))

    def test_overlapping_targets_execute_only_marked_tests(self):
        targets = [FIRST, f"{FIRST}::Suite", f"{FIRST}::{MARKED}", "test/nested"]
        markers = ramjet.collect_markers(targets, root=str(self.root))
        selected = ramjet.select_tests(markers, ("t4",))["t4"]
        expected = sorted(
            f"{file}::{selector}"
            for file in (FIRST, SECOND)
            for selector in (MARKED, SPECIFIC)
        )
        self.assertEqual(selected, expected)

        result = ramjet.execute_tests(
            selected, root=str(self.root), platform=ramjet.T4_PLATFORM
        )
        self.assertEqual(result["returncode"], 0, result["stdout"] + result["stderr"])
        cases = ET.fromstring(result["junit"]).findall(".//testcase")
        self.assertEqual(len(cases), len(expected))
        self.assertEqual(
            sorted(case.attrib["name"] for case in cases),
            ["test_marked", "test_marked", "test_specific", "test_specific"],
        )
        for case in cases:
            self.assertIsNone(case.find("failure"))
            self.assertIsNone(case.find("error"))
            self.assertIsNone(case.find("skipped"))

    def test_invalid_selector_does_not_return_partial_collection(self):
        with self.assertRaisesRegex(RuntimeError, "Collection failed"):
            ramjet.collect_markers(
                [FIRST, f"{SECOND}::MissingSuite"], root=str(self.root)
            )

    def test_empty_platform_selection_is_rejected(self):
        markers = ramjet.collect_markers(
            [f"{FIRST}::Suite::test_excluded"], root=str(self.root)
        )
        with self.assertRaisesRegex(ValueError, "No tests"):
            ramjet.select_tests(markers)

    def test_inherited_defaults_and_exclusions(self):
        markers = ramjet.collect_markers([f"{FIRST}::Defaults"], root=str(self.root))
        self.assertEqual(
            ramjet.select_tests(markers),
            {
                "cpu": [
                    f"{FIRST}::Defaults::test_cpu_only",
                    f"{FIRST}::Defaults::test_inherited",
                ],
                "l4": [
                    f"{FIRST}::Defaults::test_cuda_only",
                    f"{FIRST}::Defaults::test_inherited",
                ],
            },
        )

    def test_cpu_execution_receives_explicit_platform(self):
        node = f"{FIRST}::Defaults::test_cpu_only"
        result = ramjet.execute_tests(
            [node], root=str(self.root), platform=ramjet.CPU_PLATFORM
        )
        self.assertEqual(result["returncode"], 0, result["stdout"] + result["stderr"])
        case = ET.fromstring(result["junit"]).find(".//testcase")
        self.assertIsNotNone(case)
        self.assertIsNone(case.find("skipped"))


def worker_result(nodes, **overrides):
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite", tests=str(len(nodes)))
    for node in nodes:
        file, _, name = node.partition("::")
        ET.SubElement(suite, "testcase", classname=file, name=name)
    return {
        "returncode": 0,
        "stdout": "worker output",
        "stderr": "worker diagnostics",
        "junit": ET.tostring(suites, encoding="unicode"),
        **overrides,
    }


@instantiate_parametrized_tests
class TestRamjetCLI(TestCase):
    @parametrize("platforms", [(), ("t4",)])
    @parametrize("jobs,gpu_jobs", [(None, None), (2, 1), (None, 1)])
    def test_collection_uses_cpu_and_runtime_files_avoid_image_builds(
        self, platforms, jobs, gpu_jobs
    ):
        node = f"{FIRST}::{MARKED}"
        modal = mock.MagicMock()
        modal.App.return_value.app_id = "ap-ramjet-cli-test"
        test = mock.Mock()
        test.with_options.side_effect = lambda **options: mock.Mock(
            map=lambda batches, **kwargs: [worker_result(nodes) for nodes in batches]
        )
        functions = {
            "collection_targets": mock.Mock(remote=mock.Mock(return_value=[[FIRST]])),
            "collect_markers": mock.Mock(
                map=mock.Mock(
                    return_value=[
                        {node: [value[0] for value in ramjet.PLATFORMS.values()]}
                    ]
                )
            ),
            "run_tests": test,
        }
        options_by_function = {}
        image = mock.MagicMock()

        def prepare(root, snapshots, base_image, volume, plan, clean):
            plan.update(
                build_cache_hit=True,
                source_delta_bytes=0,
                build_image_id="im-cached-pytorch",
            )
            return image

        def drained(workers):
            test.with_options.assert_not_called()
            if functions["collection_targets"] in workers.values():
                functions["collect_markers"].map.assert_not_called()
            else:
                functions["collect_markers"].map.assert_called_once()

        def register(**options):
            def decorate(function):
                options_by_function[function.__name__] = options
                return functions[function.__name__]

            return decorate

        modal.App.return_value.function.side_effect = register
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            snapshots = {
                "build": (output / "build.json", "build-hash"),
                "tests": (output / "tests.tar.gz", "test-hash"),
                "bootstrap": (output / "bootstrap", "bootstrap-hash"),
            }
            args = ["ramjet", "test/test_jit.py"]
            for platform in platforms:
                args.extend(("--platform", platform))
            if jobs is not None:
                args.extend(("--jobs", str(jobs)))
            if gpu_jobs is not None:
                args.extend(("--gpu-jobs", str(gpu_jobs)))
            with (
                mock.patch.dict(sys.modules, {"modal": modal}),
                mock.patch.dict(ramjet.os.environ, {}, clear=True),
                mock.patch.object(sys, "argv", args),
                mock.patch.object(ramjet, "normalize_targets", return_value=[FIRST]),
                mock.patch.object(ramjet, "create_run_directory", return_value=output),
                mock.patch.object(ramjet, "snapshot_source", return_value=snapshots),
                mock.patch.object(ramjet, "prepare_image", side_effect=prepare),
                mock.patch.object(ramjet, "save_build"),
                mock.patch.object(
                    ramjet, "drain_workers", side_effect=drained
                ) as drain,
                mock.patch("builtins.print"),
            ):
                ramjet.main()
            plan = json.loads((output / "plan.json").read_text())
            expected = platforms or ("cpu", "l4")
            self.assertEqual(list(plan["platforms"]), list(expected))
            self.assertEqual(list(plan["shards"]), list(expected))
            self.assertEqual(plan["jobs"], jobs)
            self.assertEqual(plan["gpu_jobs"], gpu_jobs)
            self.assertEqual(
                plan["worker_waves"],
                ramjet.allocate_workers(plan["shards"], jobs, gpu_jobs),
            )
            if jobs is None:
                drain.assert_not_called()
            else:
                self.assertEqual(
                    [list(call.args[0].values()) for call in drain.call_args_list],
                    [[functions["collection_targets"]], [functions["collect_markers"]]],
                )
        mounts = [
            call for call in image.mock_calls if call[0].endswith("add_local_file")
        ]
        self.assertEqual(len(mounts), 1 + len(ramjet.BOOTSTRAP))
        self.assertCountEqual(
            [call.args[0] for call in mounts],
            [
                snapshots["tests"][0],
                *(snapshots["bootstrap"][0] / path for path in ramjet.BOOTSTRAP),
            ],
        )
        self.assertCountEqual(
            [call.args[1] for call in mounts],
            [
                "/tmp/ramjet-tests.tar.gz",
                *(f"{ramjet.REMOTE_ROOT}/{path}" for path in ramjet.BOOTSTRAP),
            ],
        )
        self.assertTrue(all(not call.kwargs.get("copy", False) for call in mounts))
        modal.Image.from_registry.assert_not_called()
        for name in ("collection_targets", "collect_markers"):
            self.assertIsNone(options_by_function[name].get("gpu"))
            self.assertEqual(options_by_function[name]["buffer_containers"], 0)
            self.assertEqual(options_by_function[name]["scaledown_window"], 2)
        for options in options_by_function.values():
            self.assertEqual(options["min_containers"], 0)
        self.assertCountEqual(
            [call.kwargs["gpu"] for call in test.with_options.call_args_list],
            [ramjet.PLATFORMS[name][1] for name in expected],
        )

    @parametrize(
        "environment,flags,expected",
        [
            ({}, [], (None, None)),
            ({}, ["--clean-build"], (None, None)),
            ({"RAMJET_JOBS": "4", "RAMJET_GPU_JOBS": "1"}, [], (4, 1)),
            (
                {"RAMJET_JOBS": "4", "RAMJET_GPU_JOBS": "1"},
                ["-j", "2", "--gpu-jobs", "0", "--platform", "cpu"],
                (2, 0),
            ),
        ],
    )
    def test_worker_budget_defaults_and_overrides(self, environment, flags, expected):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            snapshots = {
                "build": (output / "build.json", "build-hash"),
                "tests": (output / "tests.tar.gz", "test-hash"),
                "bootstrap": (output / "bootstrap", "bootstrap-hash"),
            }
            with (
                mock.patch.dict(ramjet.os.environ, environment, clear=True),
                mock.patch.object(sys, "argv", ["ramjet", "--dry-run", *flags]),
                mock.patch.object(ramjet, "normalize_targets", return_value=[FIRST]),
                mock.patch.object(ramjet, "create_run_directory", return_value=output),
                mock.patch.object(ramjet, "snapshot_source", return_value=snapshots),
                mock.patch.object(ramjet, "prepare_image") as prepare,
                mock.patch("builtins.print"),
            ):
                ramjet.main()
            plan = json.loads((output / "plan.json").read_text())
            self.assertEqual((plan["jobs"], plan["gpu_jobs"]), expected)
            self.assertEqual(plan["clean_build"], "--clean-build" in flags)
            prepare.assert_not_called()

    @parametrize(
        "environment,flags",
        [
            ({}, ["-j", "0"]),
            ({}, ["-j", "-1"]),
            ({}, ["--gpu-jobs", "-1"]),
            ({"RAMJET_JOBS": "0"}, []),
            ({"RAMJET_GPU_JOBS": "-1"}, []),
            ({"RAMJET_JOBS": "invalid"}, []),
            ({"RAMJET_GPU_JOBS": "invalid"}, []),
        ],
    )
    def test_invalid_worker_budgets_fail_before_snapshot(self, environment, flags):
        with (
            mock.patch.dict(ramjet.os.environ, environment, clear=True),
            mock.patch.object(sys, "argv", ["ramjet", *flags]),
            mock.patch.object(ramjet, "snapshot_source") as snapshot,
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit) as error,
        ):
            ramjet.main()
        self.assertEqual(error.exception.code, 2)
        snapshot.assert_not_called()


@instantiate_parametrized_tests
class TestRamjetImages(TestCase):
    root: Path
    bootstrap: Path
    manifest: Path
    delta: Path
    snapshots: dict
    plan: dict
    modal: mock.MagicMock
    image: mock.MagicMock
    volume: mock.MagicMock
    create_delta: mock.Mock

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-image-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.joinpath("tools").mkdir()
        self.root.joinpath("tools/ramjet_build.py").write_text("builder implementation")
        self.bootstrap = self.root / "bootstrap"
        self.bootstrap.joinpath("tools").mkdir(parents=True)
        for path in ramjet.BOOTSTRAP:
            self.bootstrap.joinpath(path).write_text(f"frozen {path}")
        self.manifest = self.root / "build.json"
        self.manifest.write_text('{"version.txt": {"sha256": "version"}}')
        self.delta = self.root / "delta.tar.gz"
        self.delta.write_bytes(b"source delta")
        self.snapshots = {
            "build": (self.manifest, "source-hash"),
            "bootstrap": (self.bootstrap, "bootstrap-hash"),
        }
        self.plan = {"cache_volume": "ramjet-test-cache", "build_sha256": "source-hash"}
        self.modal = mock.MagicMock()
        self.image = mock.MagicMock(object_id="im-pytorch")
        self.modal.Image.from_registry.return_value = self.image
        self.modal.Image.from_id.return_value = self.image
        for method in (
            "dockerfile_commands",
            "apt_install",
            "env",
            "workdir",
            "add_local_file",
            "add_local_dir",
            "run_function",
        ):
            getattr(self.image, method).return_value = self.image
        self.volume = mock.MagicMock()
        self.volume.read_file.side_effect = FileNotFoundError
        modules = mock.patch.dict(sys.modules, {"modal": self.modal})
        modules.start()
        self.addCleanup(modules.stop)
        delta = mock.patch.object(
            ramjet, "create_source_delta", return_value=self.delta
        )
        self.create_delta = delta.start()
        self.addCleanup(delta.stop)

    def test_cold_build_uses_cpu_and_fixed_build_configuration(self):
        image = ramjet.prepare_image(
            self.root, self.snapshots, ramjet.BASE_IMAGE, self.volume, self.plan, False
        )

        self.assertIs(image, self.image)
        self.modal.Image.from_registry.assert_called_once_with(ramjet.BASE_IMAGE)
        self.modal.Image.from_id.assert_not_called()
        self.create_delta.assert_called_once_with(
            self.root, self.manifest, {}, self.manifest.parent
        )
        context_call = self.image.add_local_dir.call_args
        self.assertEqual(context_call.args[1:], ("/",))
        self.assertEqual(context_call.kwargs, {"copy": True})
        context = Path(context_call.args[0])
        expected = {
            "tmp/ramjet-source-delta.tar.gz": b"source delta",
            "opt/ramjet/tools/__init__.py": b"frozen tools/__init__.py",
            "opt/ramjet/tools/ramjet_build.py": b"frozen tools/ramjet_build.py",
        }
        self.assertEqual(
            {
                path.relative_to(context).as_posix(): path.read_bytes()
                for path in context.rglob("*")
                if path.is_file()
            },
            expected,
        )
        self.image.add_local_file.assert_not_called()
        self.image.workdir.assert_called_once_with(ramjet.REMOTE_ROOT)
        environment = self.image.env.call_args.args[0]
        self.assertEqual(environment["PYTHONPATH"], ramjet.REMOTE_ROOT)
        self.assertEqual(environment["TORCH_CUDA_ARCH_LIST"], "7.5;8.9")
        self.assertNotIn("PYTORCH_BUILD_VERSION", environment)
        build = self.image.run_function.call_args
        self.assertEqual(build.args, (ramjet.build_wheel,))
        self.assertEqual(build.kwargs["cpu"], 32)
        self.assertEqual(build.kwargs["memory"], 131072)
        self.assertIsNone(build.kwargs["gpu"])
        self.assertFalse(build.kwargs["include_source"])
        self.assertFalse(build.kwargs["force_build"])
        self.assertEqual(build.kwargs["volumes"], {"/cache/ccache": self.volume})
        self.assertEqual(build.kwargs["kwargs"]["build_sha256"], "source-hash")
        self.assertFalse(self.plan["build_cache_hit"])
        self.assertEqual(self.plan["source_delta_bytes"], len(b"source delta"))

    def test_exact_cache_hit_does_not_prepare_or_upload_build_inputs(self):
        self.volume.read_file.side_effect = None
        self.volume.read_file.return_value = [
            json.dumps(
                {"image_id": "im-cached", "build_sha256": "source-hash"}
            ).encode()
        ]

        image = ramjet.prepare_image(
            self.root, self.snapshots, ramjet.BASE_IMAGE, self.volume, self.plan, False
        )

        self.assertIs(image, self.image)
        self.modal.Image.from_id.assert_called_once_with("im-cached")
        self.modal.Image.from_registry.assert_not_called()
        self.create_delta.assert_not_called()
        self.image.add_local_file.assert_not_called()
        self.image.add_local_dir.assert_not_called()
        self.image.run_function.assert_not_called()
        self.volume.read_file.assert_called_once()
        self.assertTrue(self.plan["build_cache_hit"])
        self.assertEqual(self.plan["source_delta_bytes"], 0)
        self.assertEqual(self.plan["build_image_id"], "im-cached")

    def test_incremental_build_uses_parent_image_and_its_source_manifest(self):
        previous = {"version.txt": {"sha256": "old-version"}}
        serialized = json.dumps(previous).encode()
        digest = hashlib.sha256(serialized).hexdigest()
        parent = {"image_id": "im-parent", "build_sha256": digest}
        self.volume.read_file.side_effect = [
            FileNotFoundError(),
            [json.dumps(parent).encode()],
            [gzip.compress(serialized)],
        ]

        ramjet.prepare_image(
            self.root, self.snapshots, ramjet.BASE_IMAGE, self.volume, self.plan, False
        )

        self.modal.Image.from_id.assert_called_once_with("im-parent")
        self.modal.Image.from_registry.assert_not_called()
        self.create_delta.assert_called_once_with(
            self.root, self.manifest, previous, self.manifest.parent
        )
        self.assertEqual(
            self.volume.read_file.call_args.args[0],
            f"/ramjet/manifests/{digest}.json.gz",
        )
        self.assertEqual(self.plan["incremental_parent_image_id"], "im-parent")
        self.assertFalse(self.plan["build_cache_hit"])
        self.assertFalse(self.image.run_function.call_args.kwargs["force_build"])

    @parametrize("stage", ["record", "manifest"])
    def test_rejects_inconsistent_shared_cache_before_reusing_image(self, stage):
        record = {"image_id": "im-corrupt", "build_sha256": "wrong-hash"}
        encoded = json.dumps(record).encode()
        self.volume.read_file.side_effect = (
            [[encoded]]
            if stage == "record"
            else [FileNotFoundError(), [encoded], [gzip.compress(b"{}")]]
        )
        with self.assertRaisesRegex(RuntimeError, "inconsistent; use --clean-build"):
            ramjet.prepare_image(
                self.root,
                self.snapshots,
                ramjet.BASE_IMAGE,
                self.volume,
                self.plan,
                False,
            )
        self.modal.Image.from_id.assert_not_called()
        self.create_delta.assert_not_called()

    def test_clean_build_bypasses_exact_cache_and_incremental_parent(self):
        ramjet.prepare_image(
            self.root, self.snapshots, ramjet.BASE_IMAGE, self.volume, self.plan, True
        )

        self.volume.read_file.assert_not_called()
        self.modal.Image.from_registry.assert_called_once_with(ramjet.BASE_IMAGE)
        self.modal.Image.from_id.assert_not_called()
        self.create_delta.assert_called_once_with(
            self.root, self.manifest, {}, self.manifest.parent
        )
        self.assertFalse(self.plan["build_cache_hit"])
        self.assertTrue(self.image.run_function.call_args.kwargs["force_build"])

    @parametrize(
        "change,invalidates",
        [
            ("builder", True),
            ("base", True),
            ("cuda", True),
            ("runner", False),
            ("checkout_builder", False),
        ],
    )
    def test_recipe_changes_invalidate_native_cache(self, change, invalidates):
        ramjet.prepare_image(
            self.root, self.snapshots, ramjet.BASE_IMAGE, self.volume, self.plan, True
        )
        original = self.plan["build_recipe_sha256"]
        if change == "builder":
            self.bootstrap.joinpath("tools/ramjet_build.py").write_text(
                "new builder implementation"
            )
        elif change == "runner":
            self.bootstrap.joinpath("tools/ramjet.py").write_text(
                "new runner implementation"
            )
        elif change == "checkout_builder":
            self.root.joinpath("tools/ramjet_build.py").write_text(
                "checkout changed after snapshot"
            )
        base = (
            "registry/image@sha256:different" if change == "base" else ramjet.BASE_IMAGE
        )
        arches = "8.9" if change == "cuda" else ramjet.CUDA_ARCHS
        with mock.patch.object(ramjet, "CUDA_ARCHS", arches):
            ramjet.prepare_image(
                self.root, self.snapshots, base, self.volume, self.plan, True
            )
        self.assertEqual(self.plan["build_recipe_sha256"] != original, invalidates)

    def test_successful_build_publishes_immutable_image_and_source_manifest(self):
        digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        recipe = hashlib.sha256(b"build recipe").hexdigest()
        self.plan.update(
            build_cache_hit=False,
            build_recipe_sha256=recipe,
            build_sha256=digest,
            cache_volume="v" * 64,
        )

        ramjet.save_build(self.image, self.volume, self.manifest, self.plan)

        tag = hashlib.sha256(f"{recipe}:{digest}".encode()).hexdigest()
        self.image.publish.assert_called_once_with(f"ramjet-build:{tag}")
        name = self.image.publish.call_args.args[0]
        self.assertRegex(name, r"^ramjet-build:[0-9a-f]{64}$")
        self.assertTrue(all(len(part) <= 64 for part in name.split(":")))
        self.volume.batch_upload.assert_called_once_with(force=True)
        upload = self.volume.batch_upload.return_value.__enter__.return_value
        written = {
            call.args[1]: call.args[0].getvalue()
            for call in upload.put_file.call_args_list
        }
        self.assertEqual(
            json.loads(gzip.decompress(written[f"/ramjet/manifests/{digest}.json.gz"])),
            json.loads(self.manifest.read_text()),
        )
        expected = {"image_id": "im-pytorch", "build_sha256": digest}
        self.assertEqual(
            json.loads(written[f"/ramjet/{recipe}/{digest}.json"]), expected
        )
        self.assertEqual(json.loads(written[f"/ramjet/{recipe}/latest.json"]), expected)
        self.assertEqual(self.plan["build_image_id"], "im-pytorch")

    def test_cache_hit_does_not_republish_or_update_shared_cache(self):
        self.plan["build_cache_hit"] = True
        ramjet.save_build(self.image, self.volume, self.manifest, self.plan)
        self.image.publish.assert_not_called()
        self.volume.batch_upload.assert_not_called()

    def test_runtime_preparation_uses_installed_wheel_and_preserves_bootstrap(self):
        source = self.root / "source"
        source.mkdir()
        for name in ("torch", "torchgen", "functorch", "build", "aten", "tools"):
            source.joinpath(name).mkdir()
        source.joinpath("version.txt").write_text("version")
        runtime = self.root / "runtime"
        runtime.joinpath("tools").mkdir(parents=True)
        bootstrap = runtime / "tools/ramjet.py"
        bootstrap.write_text("bootstrap")
        snapshot = self.root / "runtime.tar.gz"
        with tarfile.open(snapshot, "w:gz") as archive:
            for name, text in {
                "ramjet-source.json": "{}",
                "test/test_new.py": "test",
                "tools/helper.py": "helper",
            }.items():
                content = text.encode()
                entry = tarfile.TarInfo(name)
                entry.size = len(content)
                archive.addfile(entry, io.BytesIO(content))
        open_archive = tarfile.open

        def open_runtime(path):
            self.assertEqual(path, "/tmp/ramjet-tests.tar.gz")
            return open_archive(snapshot)

        def remote_path(path):
            return source if path == "/opt/pytorch" else Path(path)

        with (
            mock.patch.object(ramjet, "Path", side_effect=remote_path),
            mock.patch.object(
                ramjet.tarfile, "open", side_effect=open_runtime
            ) as opened,
        ):
            ramjet.prepare_runtime(str(runtime))
            ramjet.prepare_runtime(str(runtime))
        opened.assert_called_once()
        self.assertEqual(bootstrap.read_text(), "bootstrap")
        self.assertEqual(runtime.joinpath("test/test_new.py").read_text(), "test")
        self.assertEqual(runtime.joinpath("tools/helper.py").read_text(), "helper")
        self.assertTrue(runtime.joinpath("aten").is_symlink())
        self.assertTrue(runtime.joinpath("version.txt").is_symlink())
        for name in ("torch", "torchgen", "functorch", "build"):
            self.assertFalse(runtime.joinpath(name).exists())


@instantiate_parametrized_tests
class TestRamjetReports(TestCase):
    modal: mock.MagicMock

    def setUp(self):
        super().setUp()
        self.modal = mock.MagicMock()
        module_patch = mock.patch.dict(sys.modules, {"modal": self.modal})
        module_patch.start()
        self.addCleanup(module_patch.stop)

    @parametrize("stage", ["success", "build", "collection", "tests", "interrupt"])
    def test_main_preserves_outcome_and_records_run(self, stage):
        app_id = "ap-ramjet-report-test"
        app = self.modal.App.return_value
        app.app_id = app_id
        app.run.return_value.__exit__.return_value = False
        node = f"{FIRST}::{MARKED}"
        functions = {
            "collection_targets": mock.Mock(remote=mock.Mock(return_value=[[FIRST]])),
            "collect_markers": mock.Mock(
                map=mock.Mock(return_value=[{node: [ramjet.CPU_PLATFORM]}])
            ),
            "run_tests": mock.Mock(),
        }
        app.function.side_effect = lambda **options: lambda function: functions[
            function.__name__
        ]
        failure = None
        if stage == "build":
            failure = RuntimeError("CPU wheel build failed")
            app.run.return_value.__enter__.side_effect = failure
        elif stage == "collection":
            failure = RuntimeError("collection failed")
            functions["collect_markers"].map.return_value = [failure]
        elif stage == "tests":
            failure = RuntimeError("Test run failed")
        elif stage == "interrupt":
            failure = KeyboardInterrupt("user cancelled")
            functions["collection_targets"].remote.side_effect = failure
        worker = worker_result(
            [node], returncode=int(stage == "tests"), platform=ramjet.CPU_PLATFORM
        )

        def prepare(root, snapshots, base_image, volume, plan, clean):
            plan.update(
                build_cache_hit=False,
                source_delta_bytes=123,
                build_image_id="im-built-pytorch",
            )
            return mock.MagicMock()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            snapshots = {
                "build": (output / "build.json", "build-hash"),
                "tests": (output / "tests.tar.gz", "test-hash"),
                "bootstrap": (output / "bootstrap", "bootstrap-hash"),
            }

            with (
                mock.patch.dict(ramjet.os.environ, {}, clear=True),
                mock.patch.object(sys, "argv", ["ramjet", "--platform", "cpu", FIRST]),
                mock.patch.object(ramjet, "normalize_targets", return_value=[FIRST]),
                mock.patch.object(ramjet, "create_run_directory", return_value=output),
                mock.patch.object(ramjet, "snapshot_source", return_value=snapshots),
                mock.patch.object(ramjet, "prepare_image", side_effect=prepare),
                mock.patch.object(ramjet, "save_build"),
                mock.patch.object(ramjet, "run_shards", return_value=[worker]),
                mock.patch("builtins.print") as printed,
            ):
                if failure is None:
                    ramjet.main()
                else:
                    with self.assertRaisesRegex(type(failure), str(failure)):
                        ramjet.main()
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["returncode"], int(stage != "success"))
            self.assertEqual(result["app_id"], app_id)
            self.assertEqual(result["dashboard_url"], f"https://modal.com/id/{app_id}")
            self.assertEqual(result["test_sha256"], "test-hash")
            self.assertNotIn("cost", result)
            if stage in ("success", "tests"):
                self.assertEqual(result["stdout"], worker["stdout"])
                self.assertEqual(result["shards"][0]["nodes"], [node])
                self.assertEqual((output / "junit.xml").read_text(), result["junit"])
            if failure is not None:
                self.assertIn(str(failure), result["error"])
            if stage != "build":
                app.run.return_value.__exit__.assert_called_once()
            self.modal.Workspace.from_context.assert_not_called()
            messages = [
                str(call.args[0]) for call in printed.call_args_list if call.args
            ]
            self.assertFalse(any("Modal cost" in message for message in messages))
            self.assertEqual(messages[-1], f"Modal run: https://modal.com/id/{app_id}")


@instantiate_parametrized_tests
class TestRamjetParallel(TestCase):
    @parametrize(
        "demand,jobs,gpu_jobs,expected",
        [
            ({"cpu": 100, "l4": 100}, 100, 10, [{"cpu": 90, "l4": 10}]),
            ({"cpu": 100, "l4": 100}, 4, None, [{"cpu": 2, "l4": 2}]),
            ({"cpu": 1, "l4": 100}, 4, None, [{"cpu": 1, "l4": 3}]),
            ({"cpu": 2, "l4": 1}, 100, 10, [{"cpu": 2, "l4": 1}]),
            (
                {"cpu": 10, "l4": 10, "t4": 10},
                8,
                3,
                [{"cpu": 5, "l4": 2, "t4": 1}],
            ),
            (
                {"cpu": 2, "l4": 2, "t4": 2},
                1,
                None,
                [{"cpu": 1}, {"l4": 1}, {"t4": 1}],
            ),
            (
                {"cpu": 2, "l4": 2, "t4": 2},
                None,
                1,
                [{"cpu": None, "l4": 1}, {"t4": 1}],
            ),
            (
                {"cpu": 2, "l4": 2, "t4": 2},
                None,
                None,
                [{"cpu": None, "l4": None, "t4": None}],
            ),
            ({"cpu": 2}, 4, 0, [{"cpu": 2}]),
        ],
    )
    def test_worker_allocation_respects_shared_budgets(
        self, demand, jobs, gpu_jobs, expected
    ):
        shards = {
            name: [[f"{name}-{index}"] for index in range(count)]
            for name, count in demand.items()
        }
        waves = ramjet.allocate_workers(shards, jobs, gpu_jobs)
        self.assertEqual(waves, expected)
        self.assertEqual([name for wave in waves for name in wave], list(demand))
        for wave in waves:
            if jobs is not None:
                self.assertLessEqual(
                    sum(cap for cap in wave.values() if cap is not None), jobs
                )
            if gpu_jobs is not None:
                self.assertLessEqual(
                    sum(
                        cap
                        for name, cap in wave.items()
                        if name != "cpu" and cap is not None
                    ),
                    gpu_jobs,
                )
            for name, cap in wave.items():
                if cap is not None:
                    self.assertGreater(cap, 0)
                    self.assertLessEqual(cap, demand[name])

    @parametrize("jobs,gpu_jobs", [(0, None), (-1, None), (None, -1), (None, 0)])
    def test_invalid_shared_budgets(self, jobs, gpu_jobs):
        shards = {"cpu": [["cpu"]], "l4": [["gpu"]]}
        with self.assertRaises(ValueError):
            ramjet.allocate_workers(shards, jobs, gpu_jobs)

    @parametrize("counter", ["num_total_runners", "backlog", "num_running_inputs"])
    def test_drain_waits_for_workers_and_queued_inputs(self, counter):
        idle = dict(num_total_runners=0, backlog=0, num_running_inputs=0)
        active = {**idle, counter: 1}
        worker = mock.Mock()
        worker.get_current_stats.side_effect = [
            SimpleNamespace(**active),
            SimpleNamespace(**idle),
        ]
        with mock.patch.object(ramjet, "time") as clock:
            clock.monotonic.return_value = 0
            ramjet.drain_workers({"l4": worker})
        self.assertEqual(worker.get_current_stats.call_count, 2)
        clock.sleep.assert_called_once_with(2)

    def test_drain_has_finite_timeout(self):
        worker = mock.Mock()
        worker.get_current_stats.return_value = mock.Mock(
            num_total_runners=1, backlog=0, num_running_inputs=0
        )
        with mock.patch.object(ramjet, "time") as clock:
            clock.monotonic.side_effect = range(0, 100, 2)
            with self.assertRaisesRegex(RuntimeError, "Timed out.*l4"):
                ramjet.drain_workers({"l4": worker}, timeout=4)
        self.assertLessEqual(worker.get_current_stats.call_count, 3)

    def test_single_worker_drains_exact_pool_before_starting_next(self):
        shards = {
            "cpu": [["cpu-first"], ["cpu-second"]],
            "l4": [["l4-test"]],
            "t4": [["t4-test"]],
        }
        events = []
        workers = {}

        def configured(**options):
            platforms = {gpu: name for name, (_, gpu) in ramjet.PLATFORMS.items()}
            name = platforms[options["gpu"]]
            self.assertNotIn(name, workers)
            self.assertEqual(options["max_containers"], 1)
            self.assertEqual(options["buffer_containers"], 0)
            self.assertEqual(options["scaledown_window"], 2)

            def mapped(batches, **kwargs):
                events.append(("run", name))
                return [worker_result(nodes) for nodes in batches]

            workers[name] = mock.Mock(map=mapped)
            return workers[name]

        def drained(functions):
            self.assertEqual(len(functions), 1)
            for name, worker in functions.items():
                self.assertIs(worker, workers[name])
                events.append(("drain", name))

        with mock.patch.object(ramjet, "drain_workers", side_effect=drained):
            results = ramjet.run_shards(mock.Mock(with_options=configured), shards, 1)
        self.assertEqual(
            events,
            [
                ("run", "cpu"),
                ("drain", "cpu"),
                ("run", "l4"),
                ("drain", "l4"),
                ("run", "t4"),
            ],
        )
        expected = [
            ramjet.PLATFORMS[name][0]
            for name, batches in shards.items()
            for _ in batches
        ]
        self.assertEqual([result["platform"] for result in results], expected)

    def test_gpu_budget_preserves_result_order_across_waves(self):
        shards = {"l4": [["l4-test"]], "t4": [["t4-test"]], "cpu": [["cpu-test"]]}
        workers = {}

        def configured(**options):
            worker = mock.Mock(
                map=lambda batches, **kwargs: [
                    worker_result(nodes) for nodes in batches
                ]
            )
            workers[options["gpu"]] = worker
            return worker

        with mock.patch.object(ramjet, "drain_workers") as drain:
            results = ramjet.run_shards(
                mock.Mock(with_options=configured), shards, None, gpu_jobs=1
            )
        self.assertEqual(
            drain.call_args.args[0], {"l4": workers["L4"], "cpu": workers[None]}
        )
        self.assertEqual(
            [result["platform"] for result in results],
            [ramjet.L4_PLATFORM, ramjet.T4_PLATFORM, ramjet.CPU_PLATFORM],
        )

    def test_drain_timeout_keeps_completed_reports_and_fails_remaining_shards(self):
        shards = {
            "cpu": [["cpu-first"], ["cpu-second"]],
            "l4": [["l4-test"]],
            "t4": [["t4-test"]],
        }
        started = []

        def configured(**options):
            def mapped(batches, **kwargs):
                started.append(options["gpu"])
                return [worker_result(nodes) for nodes in batches]

            return mock.Mock(map=mapped)

        with mock.patch.object(
            ramjet,
            "drain_workers",
            side_effect=TimeoutError("CPU workers did not stop"),
        ):
            results = ramjet.run_shards(mock.Mock(with_options=configured), shards, 1)
        batches = [batch for platform in shards.values() for batch in platform]
        combined = ramjet.merge_results(batches, results)
        self.assertEqual(started, [None])
        self.assertEqual([result["returncode"] for result in results], [0, 0, 1, 1])
        self.assertEqual([shard["nodes"] for shard in combined["shards"]], batches)
        self.assertIn("CPU workers did not stop", combined["stderr"])
        self.assertEqual(
            len(ET.fromstring(combined["junit"]).findall(".//testcase")), 2
        )

    def test_queue_warning_does_not_fail_successful_shards(self):
        nodes = ["gpu-test"]
        warning = "Modal container limit reached; inputs are queued"
        worker = mock.Mock()
        worker.map.return_value = [worker_result(nodes, stderr=warning)]
        test = mock.Mock(with_options=mock.Mock(return_value=worker))
        results = ramjet.run_shards(test, {"l4": [nodes]}, None)
        combined = ramjet.merge_results([nodes], results)
        self.assertEqual(combined["returncode"], 0)
        self.assertIn(warning, combined["stderr"])

    @parametrize("requested", [ramjet.CPU_PLATFORM, ramjet.L4_PLATFORM])
    def test_cpu_worker_validates_platform_without_requesting_cuda(self, requested):
        from torch.testing._internal import tci

        nodes = ["test"]
        manifest = json.dumps({"build_sha256": "abc123"})
        torch = mock.Mock(
            __file__="/venv/lib/python3.11/site-packages/torch/__init__.py",
            __version__="2.12.0+ramjet.abc123",
        )
        with (
            mock.patch.dict(sys.modules, {"torch": torch, "modal": mock.Mock()}),
            mock.patch.object(
                tci.platform, "current", return_value=tci.platform.linux_x86_64
            ),
            mock.patch.object(Path, "read_text", return_value=manifest),
            mock.patch.object(ramjet, "prepare_runtime") as prepare,
            mock.patch.object(
                ramjet, "execute_tests", return_value=worker_result(nodes)
            ) as execute,
        ):
            if requested == ramjet.CPU_PLATFORM:
                result = ramjet.run_tests(nodes, requested)
                self.assertIsNone(result["gpu"])
                execute.assert_called_once_with(nodes, platform=ramjet.CPU_PLATFORM)
            else:
                with self.assertRaisesRegex(RuntimeError, "Expected tci.platform"):
                    ramjet.run_tests(nodes, requested)
                execute.assert_not_called()
        torch.cuda.get_device_name.assert_not_called()
        prepare.assert_called_once_with()

    def test_unmarked_tests_are_excluded(self):
        markers = {
            "unmarked": [],
            "excluded": ["tci.size:small", "tci.schedule:pull"],
            "default": [ramjet.CPU_PLATFORM, ramjet.L4_PLATFORM],
        }
        self.assertEqual(
            ramjet.select_tests(markers), {"cpu": ["default"], "l4": ["default"]}
        )

    @parametrize("platform", ["cpu", "l4", "t4"])
    def test_platform_selection(self, platform):
        markers = {
            name: [marker, "tci.size:small", "tci.schedule:pull"]
            for name, (marker, _) in ramjet.PLATFORMS.items()
        }
        self.assertEqual(
            ramjet.select_tests(markers, (platform,)), {platform: [platform]}
        )

    @parametrize(
        "requirement", ["tci.size:large", "tci.schedule:periodic", "tci.unknown:x"]
    )
    def test_unsupported_requirements_are_rejected(self, requirement):
        with self.assertRaisesRegex(ValueError, "Unsupported TCI"):
            ramjet.select_tests({"test": [ramjet.L4_PLATFORM, requirement]})

    def test_variant_metadata_does_not_exclude_cpu_tests_on_l4(self):
        markers = {
            "test": [
                ramjet.CPU_PLATFORM,
                ramjet.L4_PLATFORM,
                "tci.device:cpu",
                "tci.exclude_device:cuda",
                "tci.size:small",
                "tci.schedule:pull",
            ]
        }
        self.assertEqual(
            ramjet.select_tests(markers), {"cpu": ["test"], "l4": ["test"]}
        )

    @parametrize("jobs", [None, 2])
    def test_platform_workers_run_concurrently(self, jobs):
        shards = {"cpu": [["cpu-test"]], "l4": [["l4-test"]]}
        started = threading.Barrier(2, timeout=10)
        calls = []

        def configured(**options):
            def mapped(batches, **kwargs):
                calls.append((options, kwargs))
                started.wait()
                return [worker_result(nodes) for nodes in batches]

            return mock.Mock(map=mapped)

        test = mock.Mock(with_options=configured)
        results = ramjet.run_shards(test, shards, jobs)
        self.assertEqual(
            [result["platform"] for result in results],
            [ramjet.CPU_PLATFORM, ramjet.L4_PLATFORM],
        )
        self.assertTrue(all(result["returncode"] == 0 for result in results))
        self.assertCountEqual([options["gpu"] for options, _ in calls], [None, "L4"])
        for options, kwargs in calls:
            self.assertEqual(options["max_containers"], None if jobs is None else 1)
            self.assertEqual(kwargs["order_outputs"], True)
            self.assertEqual(kwargs["return_exceptions"], True)
            self.assertIn(
                kwargs["kwargs"]["platform"], (ramjet.CPU_PLATFORM, ramjet.L4_PLATFORM)
            )

    @parametrize("completed_gpu_shards", [0, 1])
    def test_platform_map_failure_retains_completed_shards(self, completed_gpu_shards):
        shards = {"cpu": [["cpu-test"]], "l4": [["l4-first"], ["l4-second"]]}

        def configured(**options):
            def mapped(batches, **kwargs):
                if options["gpu"]:
                    for nodes in batches[:completed_gpu_shards]:
                        yield worker_result(nodes)
                    raise TimeoutError("L4 capacity unavailable")
                yield from (worker_result(nodes) for nodes in batches)

            return mock.Mock(map=mapped)

        results = ramjet.run_shards(mock.Mock(with_options=configured), shards, None)
        combined = ramjet.merge_results([*shards["cpu"], *shards["l4"]], results)
        self.assertNotEqual(combined["returncode"], 0)
        self.assertIn("L4 capacity unavailable", combined["stderr"])
        self.assertEqual(combined["shards"][1]["platform"], ramjet.L4_PLATFORM)
        self.assertEqual(
            len(ET.fromstring(combined["junit"]).findall(".//testcase")),
            1 + completed_gpu_shards,
        )

    @parametrize(
        "file_counts,jobs",
        [
            ((1,), 1),
            ((8,), 3),
            ((9,), 8),
            ((2,), 16),
            ((5, 2, 9), 4),
            ((1, 1, 1, 1, 1), 2),
        ],
    )
    def test_shards_cover_distinct_tests_in_single_file_batches(
        self, file_counts, jobs
    ):
        nodes = [
            f"test/test_{file}.py::Suite::test_{index}"
            for index in range(max(file_counts))
            for file, count in enumerate(file_counts)
            if index < count
        ]
        shards = ramjet.shard_tests(nodes + nodes[::2], jobs)
        flattened = [node for shard in shards for node in shard]
        self.assertEqual(sorted(flattened), sorted(nodes))
        self.assertGreaterEqual(len(shards), min(jobs, len(nodes)))
        for shard in shards:
            self.assertGreater(len(shard), 0)
            self.assertLessEqual(len(shard), math.ceil(len(nodes) / jobs))
            self.assertEqual(len({node.split("::", 1)[0] for node in shard}), 1)

    @parametrize("jobs", [0, -1])
    def test_invalid_jobs(self, jobs):
        with self.assertRaises(ValueError):
            ramjet.shard_tests([f"{FIRST}::{MARKED}"], jobs)

    def test_default_jobs_exposes_all_tests_to_autoscaling(self):
        nodes = [f"{FIRST}::Suite::test_{index}" for index in range(17)]
        shards = ramjet.shard_tests(nodes + nodes[:2])
        self.assertEqual(len(shards), 17)
        self.assertEqual(
            sorted(node for shard in shards for node in shard), sorted(nodes)
        )
        self.assertTrue(all(len(shard) == 1 for shard in shards))

    @parametrize("jobs", [None, 4])
    def test_empty_selection_has_no_shards(self, jobs):
        self.assertEqual(ramjet.shard_tests([], jobs), [])

    def test_merge_preserves_all_reports_and_worker_metadata(self):
        shards = [
            [f"{FIRST}::{MARKED}", f"{FIRST}::{SPECIFIC}"],
            [f"{SECOND}::{MARKED}"],
        ]
        metadata = {
            "torch_path": "/site-packages/torch/__init__.py",
            "torch_version": "2.12.0+ramjet.example",
            "gpu": "Tesla T4",
            "source": {"build_sha256": "example"},
            "build": {"build_sha256": "example", "wheel_build_seconds": 1},
        }
        workers = [
            worker_result(
                nodes,
                stdout=f"output {index}",
                stderr=f"diagnostics {index}",
                worker=f"container-{index}",
                **metadata,
            )
            for index, nodes in enumerate(shards)
        ]
        result = ramjet.merge_results(shards, workers)
        self.assertEqual(result["returncode"], 0)
        cases = ET.fromstring(result["junit"]).findall(".//testcase")
        self.assertEqual(len(cases), 3)
        self.assertEqual(
            sorted((case.attrib["classname"], case.attrib["name"]) for case in cases),
            sorted(tuple(node.split("::", 1)) for shard in shards for node in shard),
        )
        for key, value in metadata.items():
            if key != "gpu":
                self.assertEqual(result[key], value)
        for index, (nodes, worker) in enumerate(zip(shards, workers)):
            self.assertIn(worker["stdout"], result["stdout"])
            self.assertIn(worker["stderr"], result["stderr"])
            self.assertEqual(result["shards"][index]["nodes"], nodes)
            self.assertEqual(result["shards"][index]["worker"], worker["worker"])

    def test_junit_distinguishes_platforms_for_the_same_test(self):
        nodes = [f"{FIRST}::{MARKED}"]
        results = [
            worker_result(nodes, platform=platform)
            for platform in (ramjet.CPU_PLATFORM, ramjet.L4_PLATFORM)
        ]
        combined = ramjet.merge_results([nodes, nodes], results)
        suites = ET.fromstring(combined["junit"]).findall("testsuite")
        self.assertEqual(len(suites), 2)
        self.assertIn(ramjet.CPU_PLATFORM, suites[0].attrib["name"])
        self.assertIn(ramjet.L4_PLATFORM, suites[1].attrib["name"])

    @parametrize("tag", ["failure", "error", "skipped"])
    def test_junit_failures_override_successful_worker_exit(self, tag):
        shards = [[f"{FIRST}::{MARKED}"], [f"{SECOND}::{MARKED}"]]
        workers = [worker_result(nodes) for nodes in shards]
        report = ET.fromstring(workers[1]["junit"])
        case = report.find(".//testcase")
        self.assertIsNotNone(case)
        ET.SubElement(case, tag, message=f"test {tag}")
        workers[1]["junit"] = ET.tostring(report, encoding="unicode")
        result = ramjet.merge_results(shards, workers)
        self.assertNotEqual(result["returncode"], 0)
        combined = ET.fromstring(result["junit"])
        self.assertEqual(len(combined.findall(".//testcase")), 2)
        self.assertEqual(len(combined.findall(f".//testcase/{tag}")), 1)

    @parametrize(
        "report", ["", "<testsuites>", "<testsuites><testsuite /></testsuites>"]
    )
    def test_missing_or_invalid_report_keeps_other_shards(self, report):
        shards = [[f"{FIRST}::{MARKED}"], [f"{SECOND}::{MARKED}"]]
        workers = [worker_result(shards[0]), worker_result(shards[1], junit=report)]
        result = ramjet.merge_results(shards, workers)
        self.assertNotEqual(result["returncode"], 0)
        cases = ET.fromstring(result["junit"]).findall(".//testcase")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].attrib["classname"], FIRST)
        self.assertEqual(len(result["shards"]), 2)

    def test_report_with_extra_cases_fails(self):
        nodes = [f"{FIRST}::{MARKED}"]
        worker = worker_result([*nodes, f"{FIRST}::{SPECIFIC}"])
        result = ramjet.merge_results([nodes], [worker])
        self.assertNotEqual(result["returncode"], 0)
        self.assertEqual(len(ET.fromstring(result["junit"]).findall(".//testcase")), 2)

    def test_single_suite_report(self):
        nodes = [f"{FIRST}::{MARKED}"]
        worker = worker_result(nodes)
        suite = ET.fromstring(worker["junit"])[0]
        worker["junit"] = ET.tostring(suite, encoding="unicode")
        result = ramjet.merge_results([nodes], [worker])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len(ET.fromstring(result["junit"]).findall(".//testcase")), 1)

    def test_invalid_report_root(self):
        nodes = [f"{FIRST}::{MARKED}"]
        worker = worker_result(
            nodes, junit='<unrelated><testcase name="test" /></unrelated>'
        )
        result = ramjet.merge_results([nodes], [worker])
        self.assertNotEqual(result["returncode"], 0)

    def test_worker_exit_failure_keeps_valid_reports(self):
        nodes = [f"{FIRST}::{MARKED}"]
        result = ramjet.merge_results([nodes], [worker_result(nodes, returncode=2)])
        self.assertNotEqual(result["returncode"], 0)
        self.assertEqual(len(ET.fromstring(result["junit"]).findall(".//testcase")), 1)

    def test_worker_exception_keeps_completed_shards(self):
        shards = [[f"{FIRST}::{MARKED}"], [f"{SECOND}::{MARKED}"]]
        result = ramjet.merge_results(
            shards, [TimeoutError("worker timed out"), worker_result(shards[1])]
        )
        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("worker timed out", result["stderr"])
        cases = ET.fromstring(result["junit"]).findall(".//testcase")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].attrib["classname"], SECOND)
        self.assertEqual([shard["nodes"] for shard in result["shards"]], shards)

    def test_merge_rejects_missing_results(self):
        with self.assertRaises(ValueError):
            ramjet.merge_results([[f"{FIRST}::{MARKED}"]], [])

    def test_merge_rejects_empty_run(self):
        with self.assertRaises(ValueError):
            ramjet.merge_results([], [])


if __name__ == "__main__":
    run_tests()
