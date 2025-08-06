import logging
import os
import re
from dataclasses import fields
from textwrap import indent

import yaml


logger = logging.getLogger(__name__)


def get_abs_path(path: str):
    return os.path.abspath(path)


def generate_dataclass_help(cls) -> str:
    """Auto-generate help text for dataclass default values."""
    lines = []
    for field in fields(cls):
        default = field.default
        if default is not None and default != "":
            lines.append(f"{field.name:<22} = {repr(default)}")
        else:
            lines.append(f"{field.name:<22} = ''")
    return indent("\n".join(lines), "    ")


def get_existing_abs_path(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def read_yaml_file(file_path: str, app=None) -> dict:
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
