"""
Shell command execution utilities.

This module provides utilities for executing shell commands and handling
the transition from shell-based test functions to Python.
"""

import os
import subprocess
import logging
import shlex
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class ShellCommandError(Exception):
    """Exception raised when a shell command fails."""
    
    def __init__(self, command: str, returncode: int, stdout: str, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"Command '{command}' failed with return code {returncode}")


def run_command(
    command: str,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    check: bool = True,
    capture_output: bool = True,
    shell: bool = True
) -> bool:
    """
    Execute a shell command and return success status.
    
    Args:
        command: The command to execute
        cwd: Working directory for the command
        env: Environment variables (merged with os.environ)
        timeout: Timeout in seconds
        check: Whether to raise exception on failure
        capture_output: Whether to capture stdout/stderr
        shell: Whether to use shell execution
    
    Returns:
        True if command succeeded, False otherwise
    
    Raises:
        ShellCommandError: If check=True and command fails
    """
    logger = logging.getLogger(__name__)
    
    # Prepare environment
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    
    # Log the command being executed
    logger.info(f"Executing: {command}")
    if cwd:
        logger.debug(f"Working directory: {cwd}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=command_env,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            shell=shell
        )
        
        if capture_output:
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.debug(f"STDERR: {result.stderr}")
        
        success = result.returncode == 0
        
        if not success:
            logger.error(f"Command failed with return code {result.returncode}")
            if check:
                raise ShellCommandError(
                    command=command,
                    returncode=result.returncode,
                    stdout=result.stdout if capture_output else "",
                    stderr=result.stderr if capture_output else ""
                )
        
        return success
        
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        if check:
            raise
        return False
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        if check:
            raise
        return False


def run_command_with_output(
    command: str,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None
) -> Tuple[bool, str, str]:
    """
    Execute a shell command and return success status with output.
    
    Args:
        command: The command to execute
        cwd: Working directory for the command
        env: Environment variables (merged with os.environ)
        timeout: Timeout in seconds
    
    Returns:
        Tuple of (success, stdout, stderr)
    """
    logger = logging.getLogger(__name__)
    
    # Prepare environment
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    
    logger.info(f"Executing: {command}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=command_env,
            timeout=timeout,
            capture_output=True,
            text=True,
            shell=True
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return False, "", str(e)


def source_and_run(script_path: str, function_name: str, cwd: Optional[str] = None) -> bool:
    """
    Source a shell script and run a function from it.
    
    This is a transitional utility to call existing shell functions
    while we migrate to Python implementations.
    
    Args:
        script_path: Path to the shell script to source
        function_name: Name of the function to call
        cwd: Working directory for the command
    
    Returns:
        True if function succeeded, False otherwise
    """
    command = f"source {script_path} && {function_name}"
    return run_command(command, cwd=cwd, check=False)


def get_pytorch_root() -> Path:
    """Get the PyTorch repository root directory."""
    # Assume we're in .ci/pytorch/ and go up two levels
    current_dir = Path(__file__).parent.parent.parent.parent
    return current_dir.resolve()


def get_ci_dir() -> Path:
    """Get the CI directory (.ci/pytorch/)."""
    return get_pytorch_root() / ".ci" / "pytorch"


def setup_test_environment() -> Dict[str, str]:
    """Set up common environment variables for tests."""
    pytorch_root = get_pytorch_root()
    
    env_vars = {
        'PYTORCH_ROOT': str(pytorch_root),
        'PYTHONPATH': str(pytorch_root),
        'TORCH_INSTALL_DIR': f"{subprocess.check_output(['python', '-c', 'import site; print(site.getsitepackages()[0])'], text=True).strip()}/torch",
    }
    
    # Set additional paths
    torch_install_dir = env_vars['TORCH_INSTALL_DIR']
    env_vars.update({
        'TORCH_BIN_DIR': f"{torch_install_dir}/bin",
        'TORCH_LIB_DIR': f"{torch_install_dir}/lib",
        'TORCH_TEST_DIR': f"{torch_install_dir}/test",
    })
    
    return env_vars
