# Owner(s): ["module: rocm"]

import importlib.util
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import yaml


REPO_ROOT = Path(__file__).parents[2]
PREVIEW_PIN = (
    (REPO_ROOT / ".ci/docker/ci_commit_pins/rocm-preview.txt").read_text().strip()
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestRocmPreviewDocker(unittest.TestCase):
    def run_builder(
        self,
        script: Path,
        image: str,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_path = tmp_path / "bin"
            bin_path.mkdir()
            docker_log = tmp_path / "docker.log"
            scripts = {
                "git": f"""#!/bin/sh
if [ "$#" -ne 2 ] || [ "$1" != rev-parse ] || [ "$2" != --show-toplevel ]; then
  exit 64
fi
printf '%s\\n' '{REPO_ROOT}'
""",
                "docker": """#!/bin/sh
printf '%s\\n' "$@" >> "$DOCKER_LOG"
""",
            }
            for name, content in scripts.items():
                path = bin_path / name
                path.write_text(content)
                path.chmod(0o755)
            env = os.environ | {
                "PATH": f"{bin_path}:{os.environ['PATH']}",
                "DOCKER_LOG": str(docker_log),
                "REMOTE_BUILDKIT": "1",
                "REMOTE_BUILDKIT_CONNECT_ATTEMPTS": "1",
                "WITH_PUSH": "true",
            }
            result = subprocess.run(
                ["bash", str(script), image],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            return result, docker_log.read_text().splitlines()

    def test_docker_builders_use_preview_pin_and_index(self) -> None:
        nightly_index = "https://nightly.repo.amd.com/rocm/core/whl-next/"
        builders = (
            (
                REPO_ROOT / ".ci/docker/build.sh",
                "pytorch-linux-noble-rocm-preview-py3.11",
            ),
            (
                REPO_ROOT / ".ci/docker/manywheel/build.sh",
                "manylinux2_28-builder:rocm-preview",
            ),
        )
        for script, image in builders:
            with self.subTest(script=script):
                result, docker_args = self.run_builder(script, image)
                self.assertEqual(result.returncode, 0, result.stderr)
                build_args = set(itertools.pairwise(docker_args))
                self.assertIn(
                    ("--build-arg", f"ROCM_VERSION={PREVIEW_PIN}"), build_args
                )
                self.assertIn(
                    ("--build-arg", f"THEROCK_INDEX_URL={nightly_index}"),
                    build_args,
                )
                if "manywheel" in str(script):
                    matrix = load_module(
                        "generate_binary_build_matrix",
                        REPO_ROOT / ".github/scripts/generate_binary_build_matrix.py",
                    )
                    self.assertIn(
                        (
                            "--build-arg",
                            f"PYTORCH_ROCM_ARCH={matrix.ROCM_PREVIEW_DEVICES}",
                        ),
                        build_args,
                    )

        stable, docker_args = self.run_builder(
            REPO_ROOT / ".ci/docker/manywheel/build.sh",
            "manylinux2_28-builder:rocm10.1",
        )
        self.assertEqual(stable.returncode, 0, stable.stderr)
        build_args = set(itertools.pairwise(docker_args))
        self.assertIn(("--build-arg", "ROCM_VERSION=10.1"), build_args)
        self.assertIn(
            (
                "--build-arg",
                "THEROCK_INDEX_URL=https://stable.repo.amd.com/rocm/whl-next/",
            ),
            build_args,
        )


class TestRocmPreviewWheels(unittest.TestCase):
    def test_nightly_preview_is_opt_in(self) -> None:
        nightly = load_module("nightly", REPO_ROOT / "tools/nightly.py")
        if nightly.Version is None:
            self.skipTest("packaging is not installed")

        def source(name: str) -> object:
            return nightly.PipSource(name, f"https://x/{name}", {"Linux"}, "rocm")

        sources = {n: source(n) for n in ("rocm-10.0", "rocm-10.1", "rocm-preview")}

        def select(requested: str | None, srcs: dict = sources) -> str:
            return nightly._select_accelerator_source(
                "ROCm", requested, platform="Linux", sources=srcs
            ).name

        self.assertEqual(select(None), "rocm-10.1")
        self.assertEqual(select("preview"), "rocm-preview")
        self.assertEqual(select("10.0"), "rocm-10.0")
        with self.assertRaisesRegex(ValueError, "10.1, preview"):
            select("9.9")
        with self.assertRaisesRegex(ValueError, "No numeric ROCm"):
            select(None, {"rocm-preview": sources["rocm-preview"]})

    def test_preview_matrix_routes_to_preview_index(self) -> None:
        matrix = load_module(
            "generate_binary_build_matrix",
            REPO_ROOT / ".github/scripts/generate_binary_build_matrix.py",
        )
        wheels = matrix.generate_wheels_matrix("linux")
        preview = [c for c in wheels if c["gpu_arch_version"] == "preview"]
        self.assertTrue(preview)
        for config in preview:
            self.assertEqual(config["desired_cuda"], "rocmpreview")
            self.assertEqual(config["upload_subfolder"], "rocm-preview")
            self.assertEqual(config["container_image_tag_prefix"], "rocm-preview")
            devices = ",".join(
                f"device-{gfx}" for gfx in matrix.ROCM_PREVIEW_DEVICES.split(";")
            )
            # Pinned at build time so pin bumps leave generated YAML alone.
            self.assertEqual(
                config["pytorch_extra_install_requirements"],
                f"rocm[libraries,{devices}]",
            )
        self.assertEqual(
            [c["python_version"] for c in preview], matrix.ROCM_PREVIEW_PYTHON_VERSIONS
        )
        self.assertFalse(
            [
                c
                for c in wheels
                if c["gpu_arch_version"] != "preview" and "upload_subfolder" in c
            ]
        )
        libtorch = matrix.generate_libtorch_extraction_configs("linux", wheels)
        (lt_preview,) = [c for c in libtorch if c["gpu_arch_version"] == "preview"]
        self.assertEqual(lt_preview["upload_subfolder"], "rocm-preview")
        self.assertEqual(
            lt_preview["source_wheel_build_name"], "manywheel-py3_12-rocmpreview"
        )
        (lt_stable,) = [c for c in libtorch if c["gpu_arch_version"] == "10.1"]
        self.assertEqual(
            lt_stable["source_wheel_build_name"], "manywheel-py3_11-rocm10_1"
        )

    def populate_env(
        self, tmp: str, desired_cuda: str, requirement: str
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        root = Path(tmp) / "pytorch"
        (root / ".git").mkdir(parents=True)
        for name in (".ci", "version.txt"):
            (root / name).symlink_to(REPO_ROOT / name)
        envfile = Path(tmp) / "env"
        result = subprocess.run(
            ["bash", str(REPO_ROOT / ".ci/pytorch/binary_populate_env.sh")],
            env=os.environ
            | {
                "PYTORCH_ROOT": str(root),
                "PACKAGE_TYPE": "manywheel",
                "DESIRED_CUDA": desired_cuda,
                "GPU_ARCH_TYPE": "rocm",
                "BINARY_ENV_FILE": str(envfile),
                "PYTORCH_EXTRA_INSTALL_REQUIREMENTS": requirement,
            },
            capture_output=True,
            text=True,
        )
        return result, envfile

    def test_preview_build_version_records_pin(self) -> None:
        matrix = load_module(
            "generate_binary_build_matrix",
            REPO_ROOT / ".github/scripts/generate_binary_build_matrix.py",
        )
        requirements = matrix.PYTORCH_EXTRA_INSTALL_REQUIREMENTS
        preview_req = requirements["preview"]
        for desired_cuda, local, requirement, pinned in (
            (
                "rocmpreview",
                f"rocm{PREVIEW_PIN}",
                preview_req,
                f"{preview_req}=={PREVIEW_PIN}",
            ),
            ("rocm10.1", "rocm10.1", requirements["10.1"], requirements["10.1"]),
        ):
            with (
                self.subTest(desired_cuda=desired_cuda),
                tempfile.TemporaryDirectory() as tmp,
            ):
                result, envfile = self.populate_env(tmp, desired_cuda, requirement)
                self.assertEqual(result.returncode, 0, result.stderr)
                (line,) = [
                    env_line
                    for env_line in envfile.read_text().splitlines()
                    if env_line.startswith("export PYTORCH_BUILD_VERSION=")
                ]
                self.assertTrue(line.endswith(f'+{local}"'), line)
                (reqs,) = [
                    env_line
                    for env_line in envfile.read_text().splitlines()
                    if env_line.startswith("export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=")
                ]
                self.assertIn(f'="{pinned} | ', reqs)

    def test_preview_build_fails_when_pin_cannot_apply(self) -> None:
        # A marker or version on the rocm term must not ship an unpinned wheel.
        for requirement in (
            "rocm[libraries,device-gfx942]; platform_system == 'Linux'",
            "rocm[libraries,device-gfx942]==10.2.0a20261014",
        ):
            with (
                self.subTest(requirement=requirement),
                tempfile.TemporaryDirectory() as tmp,
            ):
                result, _ = self.populate_env(tmp, "rocmpreview", requirement)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"could not pin rocm=={PREVIEW_PIN}", result.stderr)

    def test_workflow_isolates_preview_lane(self) -> None:
        workflow = (
            REPO_ROOT / ".github/workflows/generated-linux-binary-manywheel-nightly.yml"
        )
        # A pin bump must not require regenerating workflows.
        self.assertNotIn(PREVIEW_PIN, workflow.read_text())
        jobs = yaml.safe_load(workflow.read_text())["jobs"]
        preview = {n for n in jobs if "preview" in n}
        self.assertIn("get-preview-docker-tag", preview)

        def needs(job: dict) -> set[str]:
            value = job.get("needs", [])
            return {value} if isinstance(value, str) else set(value)

        for name, job in jobs.items():
            with self.subTest(job=name):
                if name in preview:
                    self.assertLessEqual(needs(job), preview | {"get-label-type"})
                else:
                    self.assertFalse(needs(job) & preview)

        tag_outputs = "needs.get-preview-docker-tag.outputs."
        build = jobs["manywheel-build-rocm-preview"]
        self.assertEqual(
            build["with"]["DOCKER_IMAGE_REGISTRY_HOST"],
            "${{ " + tag_outputs + "docker_registry_host }}",
        )
        self.assertIn(
            tag_outputs + "docker_registry_host",
            jobs["libtorch-extract-rocm-preview"]["container"]["image"],
        )
        # ROCm test runners cannot pull ECR images, so tests gate separately.
        for name in ("manywheel-test-rocm-preview", "manywheel-rocm-preview-upload"):
            self.assertIn(tag_outputs + "test_available == 'true'", jobs[name]["if"])
        self.assertEqual(
            jobs["manywheel-test-rocm-preview"]["with"]["INDEX_SUBFOLDER"],
            "rocm-preview",
        )
        for name in ("manywheel-rocm-preview-upload", "libtorch-upload-rocm-preview"):
            self.assertEqual(jobs[name]["with"]["UPLOAD_SUBFOLDER"], "rocm-preview")

    def test_mirror_check_runs_on_ciflow_docker(self) -> None:
        # Pin-bump PRs are auto-labeled ciflow/docker, not ciflow/binaries.
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/build-manywheel-images.yml").read_text()
        )
        self.assertIn("ciflow/docker/*", workflow[True]["push"]["tags"])
        job = workflow["jobs"]["rocm-preview-mirror-check"]
        self.assertEqual(job["permissions"], {"contents": "read"})
        for release_ref in ("refs/heads/release/", "refs/tags/v"):
            self.assertIn(f"!startsWith(github.ref, '{release_ref}')", job["if"])


if __name__ == "__main__":
    unittest.main()
