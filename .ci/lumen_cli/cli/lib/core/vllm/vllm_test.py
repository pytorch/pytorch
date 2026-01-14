import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.envs_helper import env_path_field, env_str_field, get_env
from cli.lib.common.path_helper import copy, get_path, remove_dir
from cli.lib.common.pip_helper import (
    pip_install_first_match,
    pip_install_packages,
    pkg_exists,
    run_python,
)
from cli.lib.common.utils import run_command, working_directory
from cli.lib.core.vllm.lib import clone_vllm, run_test_plan, sample_vllm_test_library


logger = logging.getLogger(__name__)


@dataclass
class VllmTestParameters:
    """
    Parameters defining the vllm external test input

    !!!DO NOT ADD SECRETS IN THIS CLASS!!!
    you can put environment variable name in VllmTestParameters if it's not the same as the secret one
    fetch secrests directly from env variables during runtime
    """

    torch_whls_path: Path = env_path_field("WHEELS_PATH", "./dist")

    vllm_whls_path: Path = env_path_field(
        "VLLM_WHEELS_PATH", "./dist/external/vllm/wheels"
    )

    torch_cuda_arch_list: str = env_str_field("TORCH_CUDA_ARCH_LIST", "8.9")

    cleaning_script: Path = env_path_field(
        "cleaning_script", ".github/ci_configs/vllm/use_existing_torch.py"
    )

    def __post_init__(self):
        if not self.torch_whls_path.exists():
            raise ValueError("missing torch_whls_path")
        if not self.vllm_whls_path.exists():
            raise ValueError("missing vllm_whls_path")


class TestInpuType(Enum):
    TEST_PLAN = "test_plan"
    UNKNOWN = "unknown"


class VllmTestRunner(BaseRunner):
    def __init__(self, args: Any):
        self.work_directory = "vllm"
        self.test_plan = ""
        self.test_type = TestInpuType.UNKNOWN

        self.shard_id = args.shard_id
        self.num_shards = args.num_shards

        if args.test_plan:
            self.test_plan = args.test_plan
            self.test_type = TestInpuType.TEST_PLAN

        # Matches the structeur in the artifacts.zip from torcb build
        self.TORCH_WHL_PATH_REGEX = "torch*.whl"
        self.TORCH_WHL_EXTRA = "opt-einsum"
        self.TORCH_ADDITIONAL_WHLS_REGEX = [
            "vision/torchvision*.whl",
            "audio/torchaudio*.whl",
        ]

        # Match the structure of the artifacts.zip from vllm external build
        self.VLLM_TEST_WHLS_REGEX = [
            "vllm/vllm*.whl",
        ]

    def prepare(self):
        """
        prepare test environment for vllm. This includes clone vllm repo, install all wheels, test dependencies and set env
        """
        params = VllmTestParameters()
        logger.info("Display VllmTestParameters %s", params)
        self._set_envs(params)

        clone_vllm(dst=self.work_directory)
        self.cp_torch_cleaning_script(params)
        with working_directory(self.work_directory):
            remove_dir(Path("vllm"))
            self._install_wheels(params)
            self._install_dependencies()
        # verify the torches are not overridden by test dependencies

        check_versions()

    def run(self):
        """
        main function to run vllm test
        """
        self.prepare()
        try:
            with working_directory(self.work_directory):
                if self.test_type == TestInpuType.TEST_PLAN:
                    if self.num_shards > 1:
                        run_test_plan(
                            self.test_plan,
                            "vllm",
                            sample_vllm_test_library(),
                            self.shard_id,
                            self.num_shards,
                        )
                    else:
                        run_test_plan(
                            self.test_plan, "vllm", sample_vllm_test_library()
                        )
                else:
                    raise ValueError(f"Unknown test type {self.test_type}")
        finally:
            # double check the torches are not overridden by other packages
            check_versions()

    def cp_torch_cleaning_script(self, params: VllmTestParameters):
        script = get_path(params.cleaning_script, resolve=True)
        vllm_script = Path(f"./{self.work_directory}/use_existing_torch.py")
        copy(script, vllm_script)

    def _install_wheels(self, params: VllmTestParameters):
        logger.info("Running vllm test with inputs: %s", params)
        if not pkg_exists("torch"):
            # install torch from local whls if it's not installed yet.
            torch_p = f"{str(params.torch_whls_path)}/{self.TORCH_WHL_PATH_REGEX}"
            pip_install_first_match(torch_p, self.TORCH_WHL_EXTRA)

        torch_whls_path = [
            f"{str(params.torch_whls_path)}/{whl_path}"
            for whl_path in self.TORCH_ADDITIONAL_WHLS_REGEX
        ]
        for torch_whl in torch_whls_path:
            pip_install_first_match(torch_whl)
        logger.info("Done. Installed torch and other torch-related wheels ")

        logger.info("Installing vllm wheels")
        vllm_whls_path = [
            f"{str(params.vllm_whls_path)}/{whl_path}"
            for whl_path in self.VLLM_TEST_WHLS_REGEX
        ]
        for vllm_whl in vllm_whls_path:
            pip_install_first_match(vllm_whl)
        logger.info("Done. Installed vllm wheels")

    def _install_test_dependencies(self):
        """
        This method replaces torch dependencies with local torch wheel info in
        requirements/test.in file from vllm repo. then generates the test.txt
        in runtime
        """
        logger.info("generate test.txt from requirements/test.in with local torch whls")
        preprocess_test_in()
        copy("requirements/test.txt", "snapshot_constraint.txt")

        run_command(
            f"{sys.executable} -m uv pip compile requirements/test.in "
            "-o test.txt "
            "--index-strategy unsafe-best-match "
            "--constraint snapshot_constraint.txt "
            "--torch-backend cu128"
        )
        pip_install_packages(requirements="test.txt", prefer_uv=True)
        logger.info("Done. installed requirements for test dependencies")

    def _install_dependencies(self):
        pip_install_packages(packages=["-e", "tests/vllm_test_utils"], prefer_uv=True)
        pip_install_packages(packages=["hf_transfer"], prefer_uv=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # using script from vllm repo to remove all torch packages from requirements txt
        run_python("use_existing_torch.py")

        # install common packages
        for requirements in ["requirements/common.txt", "requirements/build.txt"]:
            pip_install_packages(
                requirements=requirements,
                prefer_uv=True,
            )
        # install test packages
        self._install_test_dependencies()

    def _set_envs(self, inputs: VllmTestParameters):
        os.environ["TORCH_CUDA_ARCH_LIST"] = inputs.torch_cuda_arch_list
        if not validate_cuda(get_env("TORCH_CUDA_ARCH_LIST")):
            logger.warning(
                "Missing supported TORCH_CUDA_ARCH_LIST. "
                "Currently support TORCH_CUDA_ARCH_LIST env var "
                "with supported arch [8.0, 8.9, 9.0]"
            )

        os.environ["HF_TOKEN"] = os.getenv("VLLM_TEST_HUGGING_FACE_TOKEN", "")
        if not get_env("HF_TOKEN"):
            raise ValueError(
                "missing required HF_TOKEN, please set VLLM_TEST_HUGGING_FACE_TOKEN env var"
            )
        if not get_env("TORCH_CUDA_ARCH_LIST"):
            raise ValueError(
                "missing required TORCH_CUDA_ARCH_LIST, please set TORCH_CUDA_ARCH_LIST env var"
            )
        # HF_HOME is absolutely needed on CI to avoid rate limit to HF, so explicitly fail
        # vLLM jobs when it's not set so that we know when it's missing
        if get_env("CI") and not get_env("HF_HOME"):
            raise ValueError(
                "missing required HF_HOME when running on CI, please set HF_HOME env var"
            )


def preprocess_test_in(
    target_file: str = "requirements/test.in", additional_packages: Iterable[str] = ()
):
    """
    This modifies the target_file file in place in vllm work directory.
    It removes torch and unwanted packages in target_file and replace with local torch whls
    package with format "$WHEEL_PACKAGE_NAME @ file://<LOCAL_PATH>"
    """
    additional_package_to_move = list(additional_packages or ())
    pkgs_to_remove = [
        "torch",
        "torchvision",
        "torchaudio",
        "mamba_ssm",
    ] + additional_package_to_move
    # Read current requirements
    target_path = Path(target_file)
    lines = target_path.read_text().splitlines()

    pkgs_to_add = []

    # Remove lines starting with the package names (==, @, >=) — case-insensitive
    pattern = re.compile(rf"^({'|'.join(pkgs_to_remove)})\s*(==|@|>=)", re.IGNORECASE)
    kept_lines = [line for line in lines if not pattern.match(line)]

    # Get local installed torch/vision/audio from pip freeze
    # This is hacky, but it works
    pip_freeze = subprocess.check_output(["pip", "freeze"], text=True)
    header_lines = [
        line
        for line in pip_freeze.splitlines()
        if re.match(
            r"^(torch|torchvision|torchaudio)\s*@\s*file://", line, re.IGNORECASE
        )
    ]

    # Write back: header_lines + blank + kept_lines
    out_lines = header_lines + [""] + kept_lines
    if pkgs_to_add:
        out_lines += [""] + pkgs_to_add

    out = "\n".join(out_lines) + "\n"
    target_path.write_text(out)
    logger.info("[INFO] Updated %s", target_file)


def validate_cuda(value: str) -> bool:
    VALID_VALUES = {"8.0", "8.9", "9.0"}
    return all(v in VALID_VALUES for v in value.split())


def check_versions():
    """
    check installed packages version
    """
    logger.info("Double check installed packages")
    patterns = ["torch", "torchvision", "torchaudio", "vllm"]
    for pkg in patterns:
        pkg_exists(pkg)
    logger.info("Done. checked installed packages")
