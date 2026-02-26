import logging
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


class TorchtitanTestRunner(BaseRunner):
    def __init__(self, args: Any):
        self.work_directory = "torchtitan"
        self.test_plan = args.test_plan

    def prepare(self):
        clone_torchtitan(dst=self.work_directory)
        # torchao nightly is required by torchtitan
        pip_install_packages(
            packages=[
                "--pre",
                "torchao",
                "--index-url",
                "https://download.pytorch.org/whl/nightly/cu129",
            ],
        )
        with working_directory(self.work_directory):
            pip_install_packages(packages=["-e", "."])
            pip_install_packages(packages=["pytest", "pytest-cov"])

    def run(self):
        self.prepare()
        with working_directory(self.work_directory):
            run_test_plan(self.test_plan, load_torchtitan_test_library())
