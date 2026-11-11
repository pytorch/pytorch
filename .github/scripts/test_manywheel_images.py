import re
from pathlib import Path
from unittest import main, TestCase

import generate_binary_build_matrix
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / ".ci/docker/manywheel/images.txt"
IMAGES_WORKFLOW = REPO_ROOT / ".github/workflows/build-manywheel-images.yml"

HOW_TO_FIX = (
    "A manywheel builder image variant has to be listed in "
    ".ci/docker/manywheel/images.txt and in the matrix of "
    ".github/workflows/build-manywheel-images.yml together. The first changes the "
    ".ci/docker tree hash that tags the images a ciflow/binaries run pulls, so "
    "the new image is built for the PR instead of being looked up on Docker Hub, "
    "where main has not published it for a tree that already exists."
)


def listed_images() -> set[str]:
    lines = MANIFEST.read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and line[0] != "#"}


def built_images() -> set[str]:
    workflow = yaml.safe_load(IMAGES_WORKFLOW.read_text())
    matrix = workflow["jobs"]["build"]["strategy"]["matrix"]["include"]
    return {f"{entry['name']}:{entry['tag']}" for entry in matrix}


def used_images(os: str) -> set[str]:
    configs = generate_binary_build_matrix.generate_wheels_matrix(os)
    return {
        f"{config['container_image']}:{config['container_image_tag_prefix']}"
        for config in configs
    }


class TestManywheelImages(TestCase):
    def test_manifest_matches_the_images_workflow(self) -> None:
        listed, built = listed_images(), built_images()
        self.assertEqual(
            listed,
            built,
            f"only in images.txt: {sorted(listed - built)}; only in "
            f"build-manywheel-images.yml: {sorted(built - listed)}. {HOW_TO_FIX}",
        )

    def test_binary_matrices_only_use_listed_images(self) -> None:
        for os in ("linux", "linux-aarch64"):
            with self.subTest(os):
                missing = used_images(os) - listed_images()
                self.assertFalse(
                    missing,
                    f"the {os} binary build matrix uses {sorted(missing)}, which "
                    f"images.txt does not list. {HOW_TO_FIX}",
                )

    def test_manifest_entries_are_name_and_tag(self) -> None:
        for image in listed_images():
            with self.subTest(image):
                self.assertRegex(
                    image, re.compile(r"^[a-z0-9_]+-builder:[A-Za-z0-9._-]+$")
                )


if __name__ == "__main__":
    main()
