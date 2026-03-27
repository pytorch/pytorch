"""Remote execution support — submit commands to run on RE infrastructure."""

from __future__ import annotations

import logging
from pathlib import Path

from re_cli.core.script_builder import RunnerScriptBuilder


logger = logging.getLogger(__name__)

SCRIPT_MODULES_DIR = Path(__file__).resolve().parent / "script_modules"

DEFAULT_IMAGE = "ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930"
DEFAULT_BOOTSTRAP = ["git_clone", "setup_uv", "check_python", "install_lumen"]


def _load_script_module(name: str) -> str:
    """Load a bash script from script_modules/ by name (without .sh)."""
    path = SCRIPT_MODULES_DIR / f"{name}.sh"
    if not path.exists():
        raise RuntimeError(f"script module '{name}' not found at {path}")
    template = path.read_text()
    return "\n".join(
        line
        for line in template.splitlines()
        if not line.startswith("#") and not line.startswith("set -")
    ).strip()


class LumenScriptBuilder(RunnerScriptBuilder):
    """RE script builder for lumen."""

    def _add_script(self, module_name: str, script_name: str) -> LumenScriptBuilder:
        body = _load_script_module(script_name)
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: {module_name}\n# {'=' * 44}\n" + body
        )
        return self

    def add_git_clone(self) -> LumenScriptBuilder:
        return self._add_script("git_clone", "git_clone")

    def add_setup_uv(self) -> LumenScriptBuilder:
        return self._add_script("setup_uv", "setup_uv")

    def add_check_python(self) -> LumenScriptBuilder:
        return self._add_script("check_python", "check_python")

    def add_install_lumen(self) -> LumenScriptBuilder:
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: install_lumen\n# {'=' * 44}\n"
            "uv pip install -e .ci/lumen_cli"
        )
        return self
