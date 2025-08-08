import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml
from cli.lib.common.utils import run_cmd


logger = logging.getLogger(__name__)


def force_create_dir(path: str):
    """
    Ensures that the given directory path is freshly created.

    If the directory already exists, it will be removed along with all its contents.
    Then a new, empty directory will be created at the same path.
    """
    remove_dir(path)
    ensure_dir_exists(path)


def ensure_dir_exists(path: str):
    """
    Ensure the directory exists. Create it if it doesn't exist.
    """
    if not os.path.exists(path):
        logger.info(f"[INFO] Creating directory: {path}")
        os.makedirs(path, exist_ok=True)
    else:
        logger.info(f"Directory already exists: {path}")


def remove_dir(path: str):
    """
    Remove a directory if it exists.
    """
    if os.path.exists(path):
        logger.info(f"Removing directory: {path}")
        shutil.rmtree(path)
    else:
        logger.info(f"Directory not found (skipped): {path}")


def get_abs_path(path: str):
    """
    Get the absolute path of the given path.
    """
    if not path:
        return ""
    return os.path.abspath(path)


def get_existing_abs_path(path: str) -> str:
    """
    Get and validate the absolute path of the given path.
    Raises an exception if the path does not exist.
    """

    path = get_abs_path(path)
    if is_path_exist(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def is_path_exist(path: str) -> bool:
    """
    Check if a path exists.
    """
    if not path:
        return False
    return os.path.exists(path)


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file with environment variable substitution.

    Supports replacing environment variables in the form of $VAR or ${VAR}.
    Logs any missing variables and removes unresolved placeholders.

    Args
    - file_path[str]: Local Path to the YAML file

    Returns:
    - dict[optionally]: Parsed YAML content as a dictionary.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - ValueError: If the YAML content is invalid or not a dictionary.
    - RuntimeError: For other unexpected errors during parsing.
    """
    p = get_abs_path(file_path)

    if not os.path.exists(p):
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    try:
        with open(p, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Find all $VAR and ${VAR}
        pattern = re.compile(r"\$(\w+)|\$\{([^}]+)\}")
        missing_vars = set()

        expanded_content = os.path.expandvars(raw_content)

        # Then remove any remaining unresolved $VAR or ${VAR} patterns
        for match in pattern.finditer(expanded_content):
            if match.group(1):
                missing_vars.add(match.group(1))
            else:
                missing_vars.add(match.group(2))
        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")

        # remove $VAR or ${VAR} if it does not exist in the yml file
        cleaned = re.sub(r"\$(\w+)|\$\{[^}]+\}", "", expanded_content)
        # remove the env_var holder ${ENV_VAR} if it does not exist in the env
        data = yaml.safe_load(cleaned)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML content must be a dictionary, got {type(data).__name__}"
            )
        return data
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file '{file_path}': {e}") from e
    except ValueError as e:
        raise ValueError(f"Failed to parse YAML file '{file_path}': {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error while reading YAML file '{file_path}': {e}"
        ) from e


def local_image_exists(image_name: str):
    """
    Check if a local Docker image exists.
    image name format: <image_name>:<tag>
    """
    try:
        run_cmd(f"docker image inspect {image_name}", log_cmd=False)
        return True
    except subprocess.CalledProcessError:
        return False


def clone_external_repo(target: str, repo: str, cwd: str):
    logger.info(f"cloning {target}....")
    commit = get_post_build_pinned_commit(target)

    # delete the directory if it exists
    remove_dir(cwd)

    # Clone the repo & checkout commit
    run_cmd(f"git clone {repo}")
    run_cmd(f"git checkout {commit}", cwd=cwd)
    run_cmd("git submodule update --init --recursive", cwd=cwd)


def get_post_build_pinned_commit(name: str, prefix=".github/ci_commit_pins") -> str:
    abs_path = get_abs_path(prefix)
    path = Path(abs_path) / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Pin file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def update_file_with_torch_whls(
    target_file: str = "requirements/test.in",
    pkgs_to_remove=["torch", "torchvision", "torchaudio", "xformers", "mamba_ssm"],
):
    # Read current requirements
    target_path = Path(target_file)
    lines = target_path.read_text().splitlines()

    # Remove lines starting with the package names (==, @, >=) — case-insensitive
    pattern = re.compile(rf"^({'|'.join(pkgs_to_remove)})\s*(==|@|>=)", re.IGNORECASE)
    kept_lines = [line for line in lines if not pattern.match(line)]

    # Get local torch/vision/audio installs from pip freeze
    pip_freeze = subprocess.check_output(["pip", "freeze"], text=True)
    header_lines = [
        line
        for line in pip_freeze.splitlines()
        if re.match(
            r"^(torch|torchvision|torchaudio)\s*@\s*file://", line, re.IGNORECASE
        )
    ]

    # Write back: header_lines + blank + kept_lines
    out = "\n".join(header_lines + [""] + kept_lines) + "\n"
    target_path.write_text(out)

    print(f"[INFO] Updated {target_file}")
