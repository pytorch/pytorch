import argparse
import subprocess
from typing import Dict

import generate_binary_build_matrix


def tag_image(
    image: str,
    default_tag: str,
    release_version: str,
    dry_run: str,
    tagged_images: Dict[str, bool],
) -> None:
    if image in tagged_images:
        return
    release_image = image.replace(f"-{default_tag}", f"-{release_version}")
    print(f"Tagging {image} to {release_image} , dry_run: {dry_run}")

    if dry_run == "disabled":
        subprocess.check_call(["docker", "pull", image])
        subprocess.check_call(["docker", "tag", image, release_image])
        subprocess.check_call(["docker", "push", release_image])
    tagged_images[image] = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        help="Version to tag",
        type=str,
        default="2.2",
    )
    parser.add_argument(
        "--dry-run",
        help="No Runtime Error check",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
    )

    options = parser.parse_args()
    tagged_images: Dict[str, bool] = {}
    platform_images = [
        generate_binary_build_matrix.WHEEL_CONTAINER_IMAGES,
        generate_binary_build_matrix.LIBTORCH_CONTAINER_IMAGES,
        generate_binary_build_matrix.CONDA_CONTAINER_IMAGES,
    ]
    default_tag = generate_binary_build_matrix.DEFAULT_TAG

    for platform_image in platform_images:  # type: ignore[attr-defined]
        for arch in platform_image.keys():  # type: ignore[attr-defined]
            if arch == "cpu-s390x":
                continue
            tag_image(
                platform_image[arch],  # type: ignore[index]
                default_tag,
                options.version,
                options.dry_run,
                tagged_images,
            )


if __name__ == "__main__":
    main()
