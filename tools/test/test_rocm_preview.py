import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).parents[2]


class TestRocmPreviewWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workflow = REPO_ROOT / ".github/workflows/generated-linux-binary-manywheel-nightly.yml"
        cls.jobs = yaml.safe_load(workflow.read_text())["jobs"]
        resolver = cls.jobs["get-preview-docker-tag"]
        cls.resolver_script = next(
            step["run"] for step in resolver["steps"] if step.get("id") == "calc"
        )

    def run_resolver(
        self,
        statuses: list[str],
        *,
        github_ref: str,
        ref_type: str,
        wait_seconds: int = 3600,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_path = tmp_path / "bin"
            bin_path.mkdir()
            status_file = tmp_path / "statuses"
            status_file.write_text("\n".join(statuses))
            output_file = tmp_path / "output"
            scripts = {
                "git": "#!/bin/sh\nprintf treehash\n",
                "sleep": "#!/bin/sh\nexit 0\n",
                "curl": """#!/usr/bin/env python3
import os
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
statuses = path.read_text().splitlines()
print(statuses[0], end="")
path.write_text("\\n".join(statuses[1:]))
""",
            }
            for name, content in scripts.items():
                path = bin_path / name
                path.write_text(content)
                path.chmod(0o755)
            env = os.environ | {
                "PATH": f"{bin_path}:{os.environ['PATH']}",
                "STATUS_FILE": str(status_file),
                "GITHUB_OUTPUT": str(output_file),
                "GITHUB_REF": github_ref,
                "REF_TYPE": ref_type,
                "PREVIEW_IMAGE_WAIT_SECONDS": str(wait_seconds),
                "PREVIEW_IMAGE_POLL_SECONDS": "1",
            }
            result = subprocess.run(
                ["bash", "-c", self.resolver_script],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            outputs = {}
            if output_file.exists():
                outputs = dict(
                    line.split("=", 1) for line in output_file.read_text().splitlines()
                )
            return result, outputs

    def test_nightly_waits_for_content_addressed_image(self) -> None:
        result, outputs = self.run_resolver(
            ["404", "200"],
            github_ref="refs/heads/nightly",
            ref_type="branch",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs["docker_image_suffix"], "-treehash")
        self.assertEqual(outputs["image_available"], "true")

    def test_nightly_missing_image_times_out(self) -> None:
        result, outputs = self.run_resolver(
            ["404"],
            github_ref="refs/heads/nightly",
            ref_type="branch",
            wait_seconds=0,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(outputs, {})

    def test_non_nightly_missing_image_fails(self) -> None:
        result, outputs = self.run_resolver(
            ["404"],
            github_ref="refs/heads/main",
            ref_type="branch",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(outputs, {})

    def test_ciflow_missing_image_skips_preview(self) -> None:
        result, outputs = self.run_resolver(
            ["404"],
            github_ref="refs/tags/ciflow/binaries/123",
            ref_type="tag",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs["docker_image_suffix"], "-treehash")
        self.assertEqual(outputs["image_available"], "false")

    def test_preview_graph_is_isolated_and_consistent(self) -> None:
        expected_needs = {
            "manywheel-build-rocm-preview": {"get-label-type", "get-preview-docker-tag"},
            "manywheel-test-rocm-preview": {
                "get-label-type",
                "get-preview-docker-tag",
                "manywheel-build-rocm-preview",
            },
            "manywheel-rocm-preview-upload": {
                "get-preview-docker-tag",
                "manywheel-build-rocm-preview",
                "manywheel-test-rocm-preview",
            },
            "libtorch-extract-rocm-preview": {
                "get-preview-docker-tag",
                "manywheel-build-rocm-preview",
            },
            "libtorch-upload-rocm-preview": {
                "get-preview-docker-tag",
                "libtorch-extract-rocm-preview",
            },
        }
        suffix = "${{ needs.get-preview-docker-tag.outputs.docker_image_suffix }}"
        for name, needs in expected_needs.items():
            job = self.jobs[name]
            self.assertEqual(set(job["needs"]), needs)
            self.assertIn("!startsWith(github.ref, 'refs/tags/v')", job["if"])
            self.assertIn("image_available == 'true'", job["if"])

        self.assertTrue(
            self.jobs["manywheel-build-rocm-preview"]["with"][
                "DOCKER_IMAGE_TAG_PREFIX"
            ].endswith(suffix)
        )
        self.assertTrue(
            self.jobs["manywheel-test-rocm-preview"]["with"][
                "DOCKER_IMAGE_TAG_PREFIX"
            ].endswith(suffix)
        )
        self.assertTrue(
            self.jobs["manywheel-rocm-preview-upload"]["with"][
                "DOCKER_IMAGE_TAG_PREFIX"
            ].endswith(suffix)
        )
        self.assertTrue(
            self.jobs["libtorch-extract-rocm-preview"]["container"]["image"].endswith(
                suffix
            )
        )

    def test_preview_indexes_are_explicit(self) -> None:
        self.assertEqual(
            self.jobs["manywheel-test-rocm-preview"]["with"]["INDEX_SUBFOLDER"],
            "rocm-preview",
        )
        self.assertEqual(
            self.jobs["manywheel-rocm-preview-upload"]["with"]["UPLOAD_SUBFOLDER"],
            "rocm-preview",
        )
        self.assertEqual(
            self.jobs["libtorch-upload-rocm-preview"]["with"]["UPLOAD_SUBFOLDER"],
            "rocm-preview",
        )

    def test_preview_builder_failure_does_not_block_stable_images(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/build-manywheel-images.yml").read_text()
        )
        self.assertEqual(
            workflow["jobs"]["build"]["continue-on-error"],
            "${{ matrix.tag == 'rocm-preview' }}",
        )


class TestRocmPreviewScripts(unittest.TestCase):
    def test_wheel_version_uses_preview_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pytorch"
            pins = root / ".ci/docker/ci_commit_pins"
            pins.mkdir(parents=True)
            (root / "version.txt").write_text((REPO_ROOT / "version.txt").read_text())
            for name in ("rocm-preview.txt", "triton.txt"):
                (pins / name).write_text(
                    (REPO_ROOT / ".ci/docker/ci_commit_pins" / name).read_text()
                )
            triton_version = REPO_ROOT / ".ci/docker/triton_version.txt"
            (root / ".ci/docker/triton_version.txt").write_text(
                triton_version.read_text()
            )
            subprocess.run(
                ["git", "init", str(root)],
                check=True,
                capture_output=True,
                text=True,
            )
            env_file = Path(tmp) / "env"
            env = os.environ | {
                "BINARY_ENV_FILE": str(env_file),
                "PYTORCH_ROOT": str(root),
                "PACKAGE_TYPE": "libtorch",
                "DESIRED_CUDA": "rocmpreview",
                "DESIRED_PYTHON": "3.11",
                "GPU_ARCH_TYPE": "rocm",
            }
            subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / ".ci/pytorch/binary_populate_env.sh"),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            pin = (
                REPO_ROOT / ".ci/docker/ci_commit_pins/rocm-preview.txt"
            ).read_text().strip()
            self.assertIn(f'+rocm{pin}"', env_file.read_text())

    def test_test_index_override_and_fallback(self) -> None:
        script = REPO_ROOT / ".ci/pytorch/binary_linux_test.sh"
        with tempfile.TemporaryDirectory() as tmp:
            for subfolder, expected in (
                ("rocm-preview", "rocm-preview"),
                (None, "rocmpreview"),
            ):
                output = Path(tmp) / (subfolder or "fallback")
                env = os.environ | {
                    "OUTPUT_SCRIPT": str(output),
                    "PACKAGE_TYPE": "manywheel",
                    "DESIRED_CUDA": "rocmpreview",
                    "DESIRED_PYTHON": "3.11",
                    "BUILD_ENVIRONMENT": "linux-binary-manywheel",
                }
                if subfolder is not None:
                    env["INDEX_SUBFOLDER"] = subfolder
                else:
                    env.pop("INDEX_SUBFOLDER", None)
                subprocess.run(
                    ["bash", str(script)],
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertIn(
                    f'/whl/${{CHANNEL}}/{expected}"',
                    output.read_text(),
                )

    def test_repair_wheel_preview_version(self) -> None:
        auditwheel = types.ModuleType("auditwheel")
        wheeltools = types.ModuleType("auditwheel.wheeltools")
        setattr(wheeltools, "add_platforms", lambda *args, **kwargs: None)
        setattr(wheeltools, "InWheelCtx", object)
        build_env_setup = types.ModuleType("build_env_setup")
        setattr(build_env_setup, "PLATFORM_TAGS", {})
        script = REPO_ROOT / ".ci/wheel/linux/repair_wheel.py"
        spec = importlib.util.spec_from_file_location("repair_wheel_test", script)
        if spec is None or spec.loader is None:
            raise AssertionError("unable to load repair_wheel.py")
        module = importlib.util.module_from_spec(spec)
        modules = {
            "auditwheel": auditwheel,
            "auditwheel.wheeltools": wheeltools,
            "build_env_setup": build_env_setup,
            spec.name: module,
        }
        with mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)

        self.assertFalse(module.is_pre_rocm_10("preview"))
        self.assertFalse(module.is_pre_rocm_10("10.0"))
        self.assertTrue(module.is_pre_rocm_10("9.2"))
        self.assertTrue(module.is_pre_rocm_10(""))


if __name__ == "__main__":
    unittest.main()
