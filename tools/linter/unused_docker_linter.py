import os
import re
import yaml
from pathlib import Path

DOCKER_BUILD_SCRIPT = Path(".ci/docker/build.sh")
DOCKER_BUILDS_YML = Path("docker-builds.yml")
WORKFLOWS_DIR = Path(".github/workflows")

IMAGE_TAG_REGEX = re.compile(r"(?<=--tag )pytorch/.+?:[a-zA-Z0-9._-]+")

def extract_images_from_build_script():
    images = set()
    with open(DOCKER_BUILD_SCRIPT) as f:
        for line in f:
            match = IMAGE_TAG_REGEX.search(line)
            if match:
                images.add(match.group())
    return images

def extract_images_from_docker_builds_yml():
    if not DOCKER_BUILDS_YML.exists():
        return set()
    with open(DOCKER_BUILDS_YML) as f:
        data = yaml.safe_load(f)
    return set(data.keys()) if isinstance(data, dict) else set()

def extract_images_used_in_workflows():
    used = set()
    for file in WORKFLOWS_DIR.glob("*.yml"):
        with open(file) as f:
            content = f.read()
            for match in re.findall(r"pytorch/.+?:[a-zA-Z0-9._-]+", content):
                used.add(match)
    return used

def main():
    defined_images = (
        extract_images_from_build_script()
        | extract_images_from_docker_builds_yml()
    )
    used_images = extract_images_used_in_workflows()

    unused = defined_images - used_images

    if not unused:
        print("✅ No unused Docker images found.")
    else:
        print("⚠️ Unused Docker images detected:")
        for img in sorted(unused):
            print(f"  - {img}")
        exit(1)

if __name__ == "__main__":
    main()
