import logging

from cli.lib.common.type import BaseRunner


logger = logging.getLogger(__name__)


class VllmBuildRunner(BaseRunner):
    def run(self):
        logger.info("Running vllm build")
