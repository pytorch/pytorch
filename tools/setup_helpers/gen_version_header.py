# Ideally, there would be a way in Bazel to parse version.txt
# and use the version numbers from there as substitutions for
# an expand_template action. Since there isn't, this silly script exists.

from __future__ import annotations

import argparse
import os
from typing import cast


Version = tuple[int, int, int]


def parse_version(version: str) -> Version:
    """
    Parses a version string into (major, minor, patch) version numbers.

    Args:
      version: Full version number string, possibly including revision / commit hash.

    Returns:
      An int 3-tuple of (major, minor, patch) version numbers.
    """
    # Extract version number part (i.e. toss any revision / hash parts).
    version_number_str = version
    for i in range(len(version)):
        c = version[i]
        if not (c.isdigit() or c == "."):
            version_number_str = version[:i]
            break

    return cast(Version, tuple([int(n) for n in version_number_str.split(".")]))


def apply_replacements(replacements: dict[str, str], text: str) -> str:
    """
    Applies the given replacements within the text.

    Args:
      replacements (dict): Mapping of str -> str replacements.
      text (str): Text in which to make replacements.

    Returns:
      Text with replacements applied, if any.
    """
    for before, after in replacements.items():
        text = text.replace(before, after)
    return text


def main(args: argparse.Namespace) -> None:
    with open(args.version_path) as f:
        version = f.read().strip()
    (major, minor, patch) = parse_version(version)

    replacements = {
        "@TORCH_VERSION_MAJOR@": str(major),
        "@TORCH_VERSION_MINOR@": str(minor),
        "@TORCH_VERSION_PATCH@": str(patch),
    }

    # Create the output dir if it doesn't exist.
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.template_path) as input:
        with open(args.output_path, "w") as output:
            for line in input:
                output.write(apply_replacements(replacements, line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate version.h from version.h.in template",
    )
    parser.add_argument(
        "--template-path",
        required=True,
        help="Path to the template (i.e. version.h.in)",
    )
    parser.add_argument(
        "--version-path",
        required=True,
        help="Path to the file specifying the version",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for expanded template (i.e. version.h)",
    )
    args = parser.parse_args()
    main(args)
