"""Generate a bundled license file for wheel distribution.

Concatenates the main PyTorch BSD license with all third-party licenses
found by third_party/build_bundled.py.  The result is written to an output
file without modifying the source LICENSE.

Called at build time by cmake/PostBuildSteps.cmake.
"""

import argparse
import pathlib
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir", type=pathlib.Path, help="Project source directory"
    )
    parser.add_argument("output", type=pathlib.Path, help="Output bundled license file")
    args = parser.parse_args()

    third_party = args.source_dir / "third_party"
    sys.path.insert(0, str(third_party))
    from build_bundled import create_bundled  # type: ignore[import-not-found]

    license_file = args.source_dir / "LICENSE"
    bsd_text = license_file.read_text(encoding="utf-8")

    # Append third-party licenses to the main license text.
    with license_file.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        create_bundled(str(third_party.resolve()), f, include_files=True)
    bundled = license_file.read_text(encoding="utf-8")

    # Restore the original LICENSE file.
    license_file.write_text(bsd_text, encoding="utf-8")

    args.output.write_text(bundled, encoding="utf-8")


if __name__ == "__main__":
    main()
