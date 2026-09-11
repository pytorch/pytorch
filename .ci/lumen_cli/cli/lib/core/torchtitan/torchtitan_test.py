import logging
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.pip_helper import first_matching_pkg, pip_install_packages
from cli.lib.common.utils import working_directory
from cli.lib.core.torchtitan.lib import (
    clone_torchtitan,
    load_torchtitan_test_library,
    run_test_plan,
)


logger = logging.getLogger(__name__)


def _install_built_torchtitan_dependency_wheels() -> None:
    wheels = [
        first_matching_pkg("dist/ao/torchao*.whl"),
        first_matching_pkg("dist/torchcomms/torchcomms*.whl"),
    ]
    # These wheels are built against the PyTorch wheel under test. Avoid
    # dependency resolution so pip cannot replace that PyTorch wheel.
    pip_install_packages(packages=["--no-index", "--no-deps", *wheels])


class TorchtitanTestRunner(BaseRunner):
    def __init__(self, args: Any):
        self.work_directory = "torchtitan"
        self.test_plan = args.test_plan

    def prepare(self):
        clone_torchtitan(dst=self.work_directory)
        _install_built_torchtitan_dependency_wheels()
        pip_install_packages(packages=["helion"])
        with working_directory(self.work_directory):
            pip_install_packages(packages=["-e", "."])
            pip_install_packages(packages=["pytest", "pytest-cov"])

    def run(self):
        self.prepare()
        with working_directory(self.work_directory):
            run_test_plan(self.test_plan, load_torchtitan_test_library())
