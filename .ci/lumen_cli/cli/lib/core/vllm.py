import logging

from cli.lib.common.cli_helper import BaseRunner


logger = logging.getLogger(__name__)


class VllmBuildRunner(BaseRunner):
    """
    Build vllm whels in ci
    """

    def run(self):
        logger.info("Running vllm build")
