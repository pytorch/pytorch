import logging

from cli.lib.common.file_utils import read_yaml_file


logger = logging.getLogger(__name__)


def build_vllm(config_path: str = ""):
    config_map = {}
    if config_path:
        logger.info("use config file user provided")
        logger.info("Reading config yaml file ...")
        config_map = read_yaml_file(config_path)
        logger.info(f"config_map: {config_map}")
    else:
        logger.info("please input config file, otherwise use default config")
    logger.info("running vllm build ....")
    # TODO(elainewy): implement vllm build
