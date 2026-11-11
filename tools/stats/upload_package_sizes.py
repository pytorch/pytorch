"""Record the size of the torch wheel in a build's `dist` as a `package_size` metric.

    python -m tools.stats.upload_package_sizes dist

One metric is emitted per `torch-*.whl` file in the directory via `emit_metric`.
Other wheels a build produces, such as vision or ROCm's composable kernel, are
not recorded.
Rows land in ClickHouse's `misc.ossci_uploaded_metrics` table alongside the
`job`, `build_environment`, and `pr_number` columns, which `emit_metric`
populates from environment variables.

A commit has no row for a build environment whose build did not run, for example
when the job repacks an earlier commit's wheel because the commit only touches
Python, test, or docs files (see `.github/actions/reuse-old-whl`).
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Any

from tools.stats.upload_metrics import emit_metric


def package_sizes(wheel: Path, largest_files: int = 20) -> dict[str, Any]:
    """Sizes in bytes: the wheel itself, its unpacked contents, and its largest files.

    Only `largest_files` files are listed: a torch wheel holds over ten thousand,
    and nearly all of its size is in a few shared libraries.
    """
    with zipfile.ZipFile(wheel) as archive:
        files = archive.infolist()
    files.sort(key=lambda info: (-info.file_size, info.filename))
    return {
        "wheel": wheel.name,
        # Compressed download size of this CI build's wheel, not a release wheel.
        "wheel_bytes": wheel.stat().st_size,
        # Sum of unpacked file sizes.
        "uncompressed_bytes": sum(info.file_size for info in files),
        "largest_files": {
            info.filename: {
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
            }
            for info in files[:largest_files]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", type=Path, help="Directory holding the wheel")
    args = parser.parse_args()

    wheels = sorted(args.directory.glob("torch-*.whl"))
    if not wheels:
        print(f"No torch wheel in {args.directory}, nothing to record")
    for wheel in wheels:
        sizes = package_sizes(wheel)
        print(f"{wheel.name}: {sizes['wheel_bytes']} bytes")
        emit_metric("package_size", {"head_sha": os.environ.get("HEAD_SHA"), **sizes})


if __name__ == "__main__":
    main()
