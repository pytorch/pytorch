#!/usr/bin/env python3

"""This script writes `torch/_rocm_init.py` to initialize with rocm_sdk packages.

That file is loaded by `torch/__init__.py`. See the ROCm packaging documentation
at https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md.
If the file is not provided, torch will skip initialization with the rocm_sdk
packages and will rely on its own built in code which may expect a system
install of ROCm or some other packaging setup.

Usage examples, choosing a rocm Python package version to check for during init:

    * Use installed version if available, else the default "stable" version:

        ```
        python ./tools/amd_build/write_rocm_init.py
        ```

    * Use the default "stable" version, ignoring any installed version:

        ```
        python ./tools/amd_build/write_rocm_init.py --rocm-version stable
        ```

    * Use a specific version or pattern:

        ```
        python ./tools/amd_build/write_rocm_init.py --rocm-version 7.0.0rc20250812
        python ./tools/amd_build/write_rocm_init.py --rocm-version 7.0.0*
        ```

    * Omit the version, skipping version checks:

        ```
        python ./tools/amd_build/write_rocm_init.py --rocm-version ""
        ```

The expectation is that the https://github.com/pytorch/pytorch repo will build
"PyTorch HEAD on ROCm stable", while https://github.com/ROCm/TheRock downstream
will build "PyTorch HEAD on ROCm HEAD". However, while the ROCm Python packages
continue to mature, the "stable" version pinned in this file will be
periodically updated. Eventually a pattern like `7.1.*` may be preffered here,
allowing for changes in patch versions but warning about changes to major or
minor versions.

Available versions can be seen on the index pages at
https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-releases-using-pip.
Currently (August 2025), only 7.0.0rcYYYYMMDD versions are available, but in the
future there may be non-rc versions, possibly pushed to PyPI.

The lists of library preloads available are likely to change from ROCm version
to ROCm version, while the lists of preloads _used directly by PyTorch_ should
be dependent on code in PyTorch itself. Indirect dependencies like a ROCm
library taking a dependency on another ROCm library, are possible too and would
require changes here during a version bump, even with no other PyTorch changes.
"""

import argparse
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import textwrap

DEFAULT_ROCM_SDK_VERSION = "7.0.0rc20250812"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IS_WINDOWS = platform.system() == "Windows"

# List of library preloads for Linux to generate into _rocm_init.py
LINUX_LIBRARY_PRELOADS = [
    "amd_comgr",
    "amdhip64",
    "rocprofiler-sdk-roctx",  # Linux only for the moment.
    "roctracer64",  # Linux only for the moment.
    "roctx64",  # Linux only for the moment.
    "hiprtc",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsparse",
    "hipsolver",
    "rccl",  # Linux only for the moment.
    "hipblaslt",
    "miopen",
]

# List of library preloads for Windows to generate into _rocm_init.py
WINDOWS_LIBRARY_PRELOADS = [
    "amd_comgr",
    "amdhip64",
    "hiprtc",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsparse",
    "hipsolver",
    "hipblaslt",
    "miopen",
]


def _capture(args: list[str | Path], cwd: Path) -> str:
    args = [str(arg) for arg in args]
    print(f"++ Capture [{cwd}]$ {shlex.join(args)}")
    try:
        return subprocess.check_output(
            args, cwd=str(cwd), stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"  Error capturing output: {e}")
        print(f"  Output from the failed command:\n  {e.output}")
        return ""


def get_rocm_sdk_version() -> str:
    return _capture(
        [sys.executable, "-m", "rocm_sdk", "version"], cwd=Path.cwd()
    ).strip()


def get_rocm_init_contents(sdk_version: str):
    """Gets the contents of the _rocm_init.py file to add to the build."""
    library_preloads = (
        WINDOWS_LIBRARY_PRELOADS if IS_WINDOWS else LINUX_LIBRARY_PRELOADS
    )
    library_preloads_formatted = ", ".join(f"'{s}'" for s in library_preloads)

    # Notes:
    #   * We could also set `fail_on_version_mismatch=True` here.
    #   * We could pass a version regex pattern instead of a string too.

    return textwrap.dedent(
        f"""
        def initialize():
            import rocm_sdk
            rocm_sdk.initialize_process(
                preload_shortnames=[{library_preloads_formatted}],
                check_version='{sdk_version}')
        """
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        prog="write_rocm_init.py", formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--rocm-version",
        default="auto",
        help=f"""The rocm version to check for at initialization time. Options are:
  * 'auto' (the default)
  * 'stable' (currently {DEFAULT_ROCM_SDK_VERSION})
  * an explicit version (or pattern) like '7.0.0rc20250812' or '7.0.*'
The 'auto' option will use the currently installed version or will fallback to 'stable'""",
    )
    # TODO: add argument (or other handling) for version regex?
    # TODO: add argument for explicit list of library preloads?

    args = p.parse_args()

    if args.rocm_version == "auto":
        print("Requested --rocm-version is 'auto', checking `rocm-sdk version`")
        rocm_version = get_rocm_sdk_version()
        if not rocm_version:
            print("Could not find installed rocm_sdk package, falling back to 'stable'")
            rocm_version = DEFAULT_ROCM_SDK_VERSION
    elif args.rocm_version == "stable":
        rocm_version = DEFAULT_ROCM_SDK_VERSION
    else:
        rocm_version = args.rocm_version
    print(f"Using rocm_version '{rocm_version}'")

    rocm_init_file = REPO_ROOT / "torch" / "_rocm_init.py"
    rocm_init_contents = get_rocm_init_contents(rocm_version)
    print(f"Writing to '{rocm_init_file}'")
    rocm_init_file.write_text(rocm_init_contents)
