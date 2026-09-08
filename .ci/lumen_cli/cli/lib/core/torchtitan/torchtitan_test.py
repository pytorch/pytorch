import logging
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import working_directory
from cli.lib.core.torchtitan.lib import (
    clone_torchtitan,
    load_torchtitan_test_library,
    run_test_plan,
)


logger = logging.getLogger(__name__)

# generate_binary_build_matrix.py is not importable as a package (it lives in
# .github/scripts, outside the cli package), so add that dir to sys.path the
# same way .github/scripts/get_ci_variable.py does.
_SCRIPTS_DIR = Path(__file__).resolve().parents[6] / ".github" / "scripts"


def _nightly_index_url() -> str:
    # torchao and torchcomms nightlies must match the CUDA toolchain of the
    # build, so reuse CUDA_STABLE from generate_binary_build_matrix.py (the
    # single source of truth for the stable CUDA version, e.g. "13.0" -> cu130)
    # rather than hardcoding the wheel channel here.
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from generate_binary_build_matrix import CUDA_STABLE

    return f"https://download.pytorch.org/whl/nightly/cu{CUDA_STABLE.replace('.', '')}"


class TorchtitanTestRunner(BaseRunner):
    def __init__(self, args: Any):
        self.work_directory = "torchtitan"
        self.test_plan = args.test_plan

    def prepare(self):
        clone_torchtitan(dst=self.work_directory)
        # torchao and torchcomms nightlies are required by torchtitan
        pip_install_packages(
            packages=[
                "--pre",
                "torchao",
                "torchcomms",
                "--index-url",
                _nightly_index_url(),
            ],
        )
        pip_install_packages(packages=["helion"])
        with working_directory(self.work_directory):
            pip_install_packages(packages=["-e", "."])
            pip_install_packages(packages=["pytest", "pytest-cov"])
            torch_constraint = Path("torch-constraint.txt")
            torch_constraint.write_text(
                f"torch=={version('torch')}\n", encoding="utf-8"
            )
            # This path is relative to the cloned TorchTitan checkout.
            pip_install_packages(
                requirements=".ci/docker/requirements-vlm.txt",
                constraints=str(torch_constraint),
            )

    def run(self):
        self.prepare()
        with working_directory(self.work_directory):
            run_test_plan(self.test_plan, load_torchtitan_test_library())
