import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cli.lib.common.file_utils import read_yaml_file
from cli.lib.common.utils import get_env


logger = logging.getLogger(__name__)


@dataclass
class LinuxExternalBuildBaseConfig:
    """
    Base configuration for external builds, derived from environment variables.
    These values are fetched at instance creation time.
    """

    cuda: str = field(default_factory=lambda: get_env("CUDA_VERSION", "12.8.1"))
    py: str = field(default_factory=lambda: get_env("PYTHON_VERSION", "3.12"))
    max_jobs: str = field(default_factory=lambda: get_env("MAX_JOBS", "64"))
    sccache_bucket: str = field(default_factory=lambda: get_env("SCCACHE_BUCKET", ""))
    sccache_region: str = field(default_factory=lambda: get_env("SCCACHE_REGION", ""))
    torch_cuda_arch_list: str = field(
        default_factory=lambda: get_env("TORCH_CUDA_ARCH_LIST", "8.0 9.0")
    )


class BuildRunner(ABC):
    """
    Base class for defining a build runner.

    Users should subclass this and implement the `run()` methods.
    """

    def __init__(self, config_path: str = ""):
        """
        Initialize the build runner.

        Args:
            config_path (str): Optional path to a config file (YAML, JSON, etc).
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """
        load config from file.
        Override this if you want custom parsing logic.
        """
        config = {}
        if self.config_path:
            logger.info("use config file user provided")
            logger.info("Reading config yaml file ...")
            config = read_yaml_file(self.config_path)
        else:
            logger.info("did not find the config file, use default behaviour")
        return config

    def get_external_build_config(self):
        return self.config.get("external_build", {})

    def get_build_config(self):
        return self.config.get("build", {})

    @abstractmethod
    def run(self):
        """
        Run the build or task.
        Must be implemented by subclasses.
        """
        pass
