import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]


class TestRocmPreviewDocker(unittest.TestCase):
    def run_builder(
        self,
        script: Path,
        image: str,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
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
printf '%s\\n' "$*" >> "$DOCKER_LOG"
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
            return result, docker_log.read_text()

    def test_docker_builders_use_preview_pin_and_index(self) -> None:
        pin = (
            (REPO_ROOT / ".ci/docker/ci_commit_pins/rocm-preview.txt")
            .read_text()
            .strip()
        )
        nightly_index = "https://nightly.repo.amd.com/rocm/core/whl-next/"
        builders = (
            (
                REPO_ROOT / ".ci/docker/build.sh",
                "pytorch-linux-noble-rocm-preview-py3.12",
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
                self.assertIn(f"--build-arg ROCM_VERSION={pin}", docker_args)
                self.assertIn(
                    f"--build-arg THEROCK_INDEX_URL={nightly_index}",
                    docker_args,
                )

        stable, docker_args = self.run_builder(
            REPO_ROOT / ".ci/docker/manywheel/build.sh",
            "manylinux2_28-builder:rocm10.1",
        )
        self.assertEqual(stable.returncode, 0, stable.stderr)
        self.assertIn("--build-arg ROCM_VERSION=10.1", docker_args)
        self.assertIn(
            "--build-arg THEROCK_INDEX_URL=https://stable.repo.amd.com/rocm/whl-next/",
            docker_args,
        )


if __name__ == "__main__":
    unittest.main()
