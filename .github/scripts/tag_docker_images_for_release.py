
import generate_binary_build_matrix
import os
import subprocess
import argparse

def tag_image( image, default_tag, release_version, dry_run, tagged_images):
    if image not in tagged_images:
        release_image = image.replace(f'-{default_tag}', f'-{release_version}')
        print(f"Tagging {image} to {release_image} , dry_run: {dry_run}")
        if dry_run == 'disabled':
            subprocess.run(["docker", "pull", image])
            subprocess.run(["docker", "tag", image, release_image])
            subprocess.run(["docker", "push", release_image])

        tagged_images[image] = True

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        help="Version to tag",
        type=str,
    )
    parser.add_argument(
        "--dry-run",
        help="No Runtime Error check",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
    )

    options = parser.parse_args()
    tagged_images = {}
    wheel_images = generate_binary_build_matrix.WHEEL_CONTAINER_IMAGES
    libtorch_images = generate_binary_build_matrix.LIBTORCH_CONTAINER_IMAGES
    conda_images = generate_binary_build_matrix.CONDA_CONTAINER_IMAGES
    default_tag = generate_binary_build_matrix.DEFAULT_TAG

    for arch in libtorch_images.keys():
        tag_image(libtorch_images[arch], default_tag, options.version, options.dry_run, tagged_images)

    for arch in wheel_images.keys():
        tag_image(wheel_images[arch], default_tag, options.version, options.dry_run, tagged_images)

    for arch in conda_images.keys():
        tag_image(conda_images[arch], default_tag, options.version, options.dry_run, tagged_images)


if __name__ == "__main__":
    main()
