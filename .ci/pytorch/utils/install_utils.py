"""
Package installation utilities.

This module provides utilities for installing packages and dependencies
that are commonly used in PyTorch CI tests.
"""

import logging
from typing import List, Optional, Dict, Any

from .shell_utils import run_command


def pip_install(packages: List[str], extra_args: Optional[List[str]] = None) -> bool:
    """
    Install packages using pip.
    
    Args:
        packages: List of package names to install
        extra_args: Additional arguments to pass to pip
    
    Returns:
        True if installation succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if not packages:
        return True
    
    cmd_parts = ["python", "-m", "pip", "install"]
    
    if extra_args:
        cmd_parts.extend(extra_args)
    
    cmd_parts.extend(packages)
    command = " ".join(cmd_parts)
    
    logger.info(f"Installing packages: {', '.join(packages)}")
    return run_command(command, check=False)


def install_torchvision() -> bool:
    """Install torchvision package."""
    logger = logging.getLogger(__name__)
    logger.info("Installing torchvision")
    
    # Use the same logic as the original shell script
    return run_command(
        "pip_install torchvision",
        check=False
    )


def install_torchaudio() -> bool:
    """Install torchaudio package."""
    logger = logging.getLogger(__name__)
    logger.info("Installing torchaudio")
    
    return run_command(
        "pip_install torchaudio",
        check=False
    )


def install_torchao() -> bool:
    """Install torchao package."""
    logger = logging.getLogger(__name__)
    logger.info("Installing torchao")
    
    return run_command(
        "pip_install torchao",
        check=False
    )


def install_monkeytype() -> bool:
    """Install monkeytype package."""
    logger = logging.getLogger(__name__)
    logger.info("Installing monkeytype")
    
    return pip_install(["MonkeyType"])


def install_torchrec_and_fbgemm() -> bool:
    """Install torchrec and fbgemm packages."""
    logger = logging.getLogger(__name__)
    logger.info("Installing torchrec and fbgemm")
    
    # This is a complex installation that may need special handling
    # For now, delegate to the shell function
    return run_command(
        "install_torchrec_and_fbgemm",
        check=False
    )


def install_numpy_2_compatibility() -> bool:
    """Install NumPy 2.0 and compatible packages."""
    logger = logging.getLogger(__name__)
    logger.info("Installing NumPy 2.0 compatibility packages")
    
    # Check if pandas is installed and get version
    success, pandas_version, _ = run_command_with_output(
        'python -c "import pandas; print(pandas.__version__)" 2>/dev/null'
    )
    
    packages = ["numpy==2.0.2", "scipy==1.13.1", "numba==0.60.0"]
    extra_args = ["--pre", "--force-reinstall"]
    
    if success and pandas_version.strip():
        packages.append(f"pandas=={pandas_version.strip()}")
    
    return pip_install(packages, extra_args)


def install_opencv() -> bool:
    """Install OpenCV with specific version for compatibility."""
    logger = logging.getLogger(__name__)
    logger.info("Installing OpenCV")
    
    # Use specific version as in original script
    return pip_install(["opencv-python==4.8.0.74"])


def install_pandas() -> bool:
    """Install pandas package."""
    logger = logging.getLogger(__name__)
    logger.info("Installing pandas")
    
    return pip_install(["pandas"])


# Import the run_command_with_output function
from .shell_utils import run_command_with_output


def install_pip_dependencies(dependencies: List[str]) -> bool:
    """
    Install a list of pip dependencies.
    
    Args:
        dependencies: List of package names or requirements to install
    
    Returns:
        True if installation succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if not dependencies:
        logger.info("No dependencies to install")
        return True
    
    logger.info(f"Installing pip dependencies: {', '.join(dependencies)}")
    return pip_install(dependencies)
