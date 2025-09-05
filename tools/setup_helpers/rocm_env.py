import os
from pathlib import Path


def check_if_rocm() -> bool:
    # If user defines USE_ROCM during PyTorch build, respect their intention
    use_rocm_env = os.environ.get("USE_ROCM")
    if use_rocm_env:
        return bool(use_rocm_env)
    # otherwise infer existence of ROCm installation as indication of ROCm build
    rocm_path_env = os.environ.get("ROCM_PATH", "/opt/rocm")
    if rocm_path_env and os.path.exists(rocm_path_env):
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

    The returned string is EITHER in the format:
        " @ git+<repo_url>@<commit_hash>#egg=rocm-composable-kernel"
    where:
        - <repo_url> is the URL for ROCm Composable Kernel
        - <commit_hash> is read from the commit pin file
        - "#egg=rocm-composable-kernel" specifies the package name for setuptools
    OR an empty string, making use of the existing rocm-composable-kernel installation.

    Returns:
        str: The formatted dependency string for use in install_requires.
    """
    egg_name = "#egg=rocm-composable-kernel"
    commit_pin = f"@{read_ck_pin()}"
    if os.getenv("TORCHINDUCTOR_CK_DIR"):
        # we take non-empty env as an indicator that the package has already been installed and doesn't need to be re-installed
        # this comes with a caveat that the pinned version is known to work while the preinstalled version might not
        return ""
    return f"@ git+https://github.com/ROCm/composable_kernel.git{commit_pin}{egg_name}"
