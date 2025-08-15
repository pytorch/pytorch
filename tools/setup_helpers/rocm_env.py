import os
from pathlib import Path


def check_if_rocm() -> bool:
    if os.path.exists("/opt/rocm"):
        return True
    if os.environ.get("ROCM_PATH") is not None:
        return True
    # Only used for Windows builds
    if os.environ.get("USE_ROCM"):
        return True
    return False


IS_ROCM = check_if_rocm()

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent.parent


# CK pin is read in a similar way that triton commit is
def read_ck_pin() -> str:
    """
    Reads the CK (Composable Kernel) commit hash.
    The hash is pinned to a known stable version of CK.

    Returns:
        str: The commit hash read from 'rocm-composable-kernel.txt'.
    """
    ck_file = "rocm-composable-kernel.txt"
    with open(REPO_DIR / ".ci" / "docker" / "ci_commit_pins" / ck_file) as f:
        return f.read().strip()


# Prepares a dependency string for install_requires in setuptools
# in specific PEP 508 URL format
def get_ck_dependency_string() -> str:
    """
    Generates a PEP 508-compliant dependency string for the ROCm Composable Kernel
    to be used in setuptools' install_requires.

    The returned string is in the format:
        " @ git+<repo_url>@<commit_hash>#egg=rocm-composable-kernel"
    where:
        - <repo_url> is the GitHub repository URL for ROCm Composable Kernel
        - <commit_hash> is read from the commit pin file
        - "#egg=rocm-composable-kernel" specifies the package name for setuptools

    Returns:
        str: The formatted dependency string for use in install_requires.
    """
    prefix = " @ git+"
    repo_address = "https://github.com/ROCm/composable_kernel.git"
    commit_pin = "@" + read_ck_pin()
    # The dependency doesn't get resolved without version number in the end
    # Since we use commit hash for versioning it will always be 1.0.0
    egg_name = "#egg=rocm-composable-kernel"
    return prefix + repo_address + commit_pin + egg_name
