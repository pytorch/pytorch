import logging
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.docker_helper import local_image_exists
from cli.lib.common.envs_helper import env_bool_field, env_path_field, env_str_field
from cli.lib.common.git_helper import clone_external_repo
from cli.lib.common.path_helper import (
    copy,
    ensure_dir_exists,
    force_create_dir,
    get_path,
    is_path_exist,
    remove_dir,
)
from cli.lib.common.pip_helper import (
    pip_install,
    pip_install_first_match,
    pip_install_packages,
    run_python,
)
from cli.lib.common.utils import run_cmd, working_directory


logger = logging.getLogger(__name__)


# Default path for docker build artifacts
_DEFAULT_RESULT_PATH = "./shared"

# Temp folder in vllm work place to cp torch whls in vllm work directory for docker build
_VLLM_TEMP_FOLDER = "tmp"


@dataclass
class VllmBuildParameters:
    """
    Parameters defining the vllm external input configurations.
    Combine with VllmDockerBuildArgs to define the vllm build environment
    """

    # USE_TORCH_WHEEL: when true, use local Torch wheels; requires TORCH_WHEELS_PATH.
    #  Otherwise docker build pull torch nightly during build
    use_torch_whl: bool = env_bool_field("USE_TORCH_WHEEL", True)

    # USE_LOCAL_BASE_IMAGE: when true, use an existing local Docker base image; requires BASE_IMAGE
    # Otherwise, pull dockerfile's default image remotely

    use_local_base_image: bool = env_bool_field("USE_LOCAL_BASE_IMAGE", True)
    # USE_LOCAL_DOCKERFILE: when true("1"), use a local Dockerfile; requires DOCKERFILE_PATH.
    # otherwise, use vllm's default dockerfile.torch_nightly for build
    use_local_dockerfile: bool = env_bool_field("USE_LOCAL_DOCKERFILE", True)

    # --- Pre-build condition inputs ------------------------------------------
    # BASE_IMAGE: name:tag (only needed when use_local_base_image is True)
    base_image: str = env_str_field("BASE_IMAGE")

    # DOCKERFILE_PATH: path to Dockerfile used when use_local_dockerfile is True"
    dockerfile_path: Optional[Path] = env_path_field(
        "DOCKERFILE_PATH", ".github/ci_configs/vllm/Dockerfile.tmp_vllm"
    )

    # TORCH_WHEELS_PATH: directory containing local torch wheels when use_torch_whl is True
    torch_whls_path: Optional[Path] = env_path_field("TORCH_WHEELS_PATH", "./dist")

    # --- Build output ---------------------------------------------------------
    # output_dir: where docker buildx (local exporter) will write artifacts
    output_dir: Optional[Path] = env_path_field("OUTPUT_DIR", "shared")

    def __post_init__(self):
        checks = [
            (
                self.use_torch_whl,  # flag
                True,  # trigger_value
                "torch_whls_path",  # resource
                is_path_exist,  # check_func
                "torch_whls_path is not provided, but use_torch_whl is set to 1",
            ),
            (
                self.use_local_base_image,
                True,
                "base_image",
                local_image_exists,
                f"base_image {self.base_image} does not found, but use_local_base_image is set to 1",
            ),
            (
                self.use_local_dockerfile,
                True,
                "dockerfile_path",
                is_path_exist,
                "dockerfile path does not found, but use_local_dockerfile is set to 1",
            ),
        ]
        for flag, trigger_value, attr_name, check_func, error_msg in checks:
            value = getattr(self, attr_name)
            if flag == trigger_value:
                if not value or not check_func(value):
                    raise ValueError(error_msg)
            else:
                logger.info("flag  %s is not set", flag)
        if not self.output_dir:
            raise ValueError("missing required output_dir")


@dataclass
class VllmDockerBuildArgs:
    """
    Parameters defining the vllm main build arguments.
    Combine with VllmBuildParameters to define the vllm build environment
    """

    target: str = env_str_field("TARGET", "export-wheels")
    tag_name: str = env_str_field("TAG", "vllm-wheels")
    cuda: str = env_str_field("CUDA_VERSION", "12.8.1")
    py: str = env_str_field("PYTHON_VERSION", "3.12")
    max_jobs: str = env_str_field("MAX_JOBS", "64")
    sccache_bucket: str = env_str_field("SCCACHE_BUCKET")
    sccache_region: str = env_str_field("SCCACHE_REGION")
    torch_cuda_arch_list: str = env_str_field("TORCH_CUDA_ARCH_LIST", "8.9")


class VllmBuildRunner(BaseRunner):
    """
    Build vLLM using docker buildx.

    Environment variable options:
        "USE_TORCH_WHEEL":      "1: use local wheels; 0: pull nightly from pypi",
        "TORCH_WHEELS_PATH":    "Path to local wheels (when USE_TORCH_WHEEL=1)",

        "USE_LOCAL_BASE_IMAGE": "1: use local base image; 0: default image",
         "BASE_IMAGE":           "name:tag to indicate base image the dockerfile depends on (when USE_LOCAL_BASE_IMAGE=1)",

        "USE_LOCAL_DOCKERFILE": "1: use local Dockerfile; 0: vllm repo default dockerfile.torch_nightly",
        "DOCKERFILE_PATH":      "Path to Dockerfile (when USE_LOCAL_DOCKERFILE=1)",

        "OUTPUT_DIR":           "e.g. './shared'",

        "TORCH_CUDA_ARCH_LIST": "e.g. '8.0' or '8.0;9.0'",
        "CUDA_VERSION":         "e.g. '12.8.1'",
        "PYTHON_VERSION":       "e.g. '3.12'",
        "MAX_JOBS":             "e.g. '64'",
        "SCCACHE_BUCKET":       "e.g. 'my-bucket'",
        "SCCACHE_REGION":       "e.g. 'us-west-2'",
    """

    def __init__(self, args=None):
        self.work_directory = "vllm"

    def run(self):
        """
        main function to run vllm build
        1. prepare vllm build environment
        2. prepare the docker build command args
        3. run docker build
        """
        inputs = VllmBuildParameters()
        clone_vllm()

        self.cp_dockerfile_if_exist(inputs)

        # cp torch wheels from root direct to vllm workspace if exist
        self.cp_torch_whls_if_exist(inputs)

        ensure_dir_exists(inputs.output_dir)

        cmd = self._generate_docker_build_cmd(inputs)
        logger.info("Running docker build: \n %s", cmd)
        run_cmd(cmd, cwd="vllm", env=os.environ.copy())

    def cp_torch_whls_if_exist(self, inputs: VllmBuildParameters) -> str:
        if not inputs.use_torch_whl:
            return ""

        if not inputs.torch_whls_path or not inputs.torch_whls_path.exists():
            raise FileNotFoundError(
                "torch whl path is not provided, but use_torch_whl is set to 1"
            )

        tmp_dir = f"./{self.work_directory}/{_VLLM_TEMP_FOLDER}"
        tmp_path = Path(tmp_dir)
        force_create_dir(tmp_path)
        copy(inputs.torch_whls_path, tmp_dir)
        return tmp_dir

    def cp_dockerfile_if_exist(self, inputs: VllmBuildParameters):
        if not inputs.use_local_dockerfile:
            logger.info("using vllm default dockerfile.torch_nightly for build")
            return

        if not inputs.dockerfile_path or not inputs.dockerfile_path.exists():
            raise FileNotFoundError(
                "dockerfile is not found, but USE_LOCAL_DOCKERFILE env var is set to `true`"
            )
        dockerfile_path = get_path(inputs.dockerfile_path, full_path=True)
        vllm_torch_dockerfile = Path(
            f"./{self.work_directory}/docker/Dockerfile.nightly_torch"
        )
        copy(dockerfile_path, vllm_torch_dockerfile)

    def get_result_path(self, path):
        """
        Get the absolute path of the result path
        """
        if not path:
            path = _DEFAULT_RESULT_PATH
        abs_path = get_path(path, full_path=True)
        return abs_path

    def _get_torch_wheel_path_arg(self, torch_whl_dir: Optional[Path]) -> str:
        if not torch_whl_dir:
            return ""
        return f"--build-arg TORCH_WHEELS_PATH={_VLLM_TEMP_FOLDER}"

    def _get_base_image_args(self, inputs: VllmBuildParameters) -> tuple[str, str, str]:
        """
        Returns:
            - base_image_arg: docker buildx arg string for base image
            - final_base_image_arg:  docker buildx arg string for vllm-base stage
            - pull_flag: --pull=true or --pull=false depending on whether the image exists locally
        """
        if not inputs.use_local_base_image:
            return "", "", ""

        base_image = inputs.base_image

        # set both base image and final base image to the same local image
        base_image_arg = f"--build-arg BUILD_BASE_IMAGE={base_image}"
        final_base_image_arg = f"--build-arg FINAL_BASE_IMAGE={base_image}"

        if local_image_exists(base_image):
            pull_flag = "--pull=false"
            return base_image_arg, final_base_image_arg, pull_flag
        logger.info(
            "[INFO] Local image not found:%s will try to pull from remote", {base_image}
        )
        return base_image_arg, final_base_image_arg, ""

    def _generate_docker_build_cmd(
        self,
        inputs: VllmBuildParameters,
    ) -> str:
        cfg = VllmDockerBuildArgs()
        base_image_arg, final_base_image_arg, pull_flag = self._get_base_image_args(
            inputs
        )
        torch_arg = self._get_torch_wheel_path_arg(inputs.torch_whls_path)

        return textwrap.dedent(
            f"""
            docker buildx build \
                --output type=local,dest={inputs.output_dir} \
                -f docker/Dockerfile.nightly_torch \
                {pull_flag} \
                {torch_arg} \
                {base_image_arg} \
                {final_base_image_arg} \
                --build-arg max_jobs={cfg.max_jobs} \
                --build-arg CUDA_VERSION={cfg.cuda} \
                --build-arg PYTHON_VERSION={cfg.py} \
                --build-arg USE_SCCACHE={int(bool(cfg.sccache_bucket and cfg.sccache_region))} \
                --build-arg SCCACHE_BUCKET_NAME={cfg.sccache_bucket} \
                --build-arg SCCACHE_REGION_NAME={cfg.sccache_region} \
                --build-arg torch_cuda_arch_list='{cfg.torch_cuda_arch_list}' \
                --target {cfg.target} \
                -t {cfg.tag_name} \
                --progress=plain .
        """
        ).strip()


@dataclass
class VllmTestParameters:
    """
    Parameters defining the vllm external test input configurations.
    Combine with VllmDockerBuildArgs to define the vllm build environment
    """

    # TORCH_WHEELS_PATH: directory containing local torch wheels when use_torch_whl is True
    _torch_whls_path: Optional[Path] = env_path_field("TORCH_WHEELS_PATH", "./dist")
    _vllm_whls_path: Optional[Path] = env_path_field("VLLM_WHEELS_PATH", "./shared")

    def __post_init__(self):
        if not self.torch_whls_path or not self.torch_whls_path.exists():
            raise ValueError("missing torch_whls_path")
        if not self.vllm_whls_path or not self.vllm_whls_path.exists():
            raise ValueError("missing vllm_whls_path")
    @property
    def torch_whls_path(self) -> Path:
        return self._torch_whls_path  # type: ignore

    @property
    def vllm_whls_path(self) -> Path:
        return self._vllm_whls_path  # type: ignore


class VllmTestRunner(BaseRunner):
    def __init__(self, args=None):
        self.work_directory = "vllm"
        self.test_name = args.test_name
        self.torch_whl_relatvie_path = [
            "vision/torchvision*.whl",
            "audio/torchaudio*.whl",
        ]
        self.vllm_whl_relatvie_path = [
            "wheels/xformers/xformers*.whl",
            "wheels/vllm/vllm*.whl",
            "wheels/flashinfer-python/flashinfer*.whl",
        ]

    def run(self):
        """
        main function to run vllm test
        """
        inputs = VllmTestParameters()
        clone_vllm()
        with working_directory(self.work_directory):
            remove_dir(Path("vllm"))
            torch_whls_path = [
                f"{str(inputs.torch_whls_path)}/{whl_path}"
                for whl_path in self.torch_whl_relatvie_path
            ]

            vllm_whls_path = [
                f"{str(inputs.vllm_whls_path)}/{whl_path}"
                for whl_path in self.vllm_whl_relatvie_path
            ]
            for torch_whl in torch_whls_path:
                pip_install_first_match(torch_whl)
            for vllm_whl in vllm_whls_path:
                pip_install_first_match(vllm_whl)
            run_python("use_existing_torch.py")

            for requirements in ["requirements/common.txt", "requirements/build.txt"]:
                pip_install_packages(
                    requirements=requirements,
                    prefer_uv=True,
                )
            preprocess_test_in()

            copy("requirements/test.in", "snapshot_constraint.txt", full_path=False)
            run_cmd(
                f"{sys.executable} -m uv pip compile requirements/test.in "
                "-o test.txt "
                "--index-strategy unsafe-best-match "
                "--constraint snapshot_constraint.txt "
                "--torch-backend cu128"
            )
            pip_install_packages(requirements="test.txt", prefer_uv=True)
            pip_install_packages(
                "--no-build-isolation", "git+https://github.com/state-spaces/mamba@v2.2.4"
                prefer_uv=True,
            )
            patterns = ["torch", "xformers", "torchvision", "torchaudio"]
            for pkg in patterns:
                try:
                    module = __import__(pkg)
                    version = getattr(module, "__version__", None)
                    print(f"{pkg}: {version or 'Unknown version'}")
                except ImportError:
                    print(f"{pkg}: Not installed")

def clone_vllm():
    clone_external_repo(
        target="vllm",
        repo="https://github.com/vllm-project/vllm.git",
        dst="vllm",
        update_submodules=True,
    )


def preprocess_test_in(
    target_file: str = "requirements/test.in",
    pkgs_to_remove=["torch", "torchvision", "torchaudio", "xformers", "mamba_ssm"],
):
    """
    remove packges in target_file and replace with local torch whls
    """
    # Read current requirements
    target_path = Path(target_file)
    lines = target_path.read_text().splitlines()

    # Remove lines starting with the package names (==, @, >=) — case-insensitive
    pattern = re.compile(rf"^({'|'.join(pkgs_to_remove)})\s*(==|@|>=)", re.IGNORECASE)
    kept_lines = [line for line in lines if not pattern.match(line)]

    # Get local torch/vision/audio installs from pip freeze
    # this is hacky, but it works
    pip_freeze = subprocess.check_output(["pip", "freeze"], text=True)
    header_lines = [
        line
        for line in pip_freeze.splitlines()
        if re.match(
            r"^(torch|torchvision|torchaudio)\s*@\s*file://", line, re.IGNORECASE
        )
    ]

    # Write back: header_lines + blank + kept_lines
    out = "\n".join(header_lines + [""] + kept_lines) + "\n"
    target_path.write_text(out)

    print(f"[INFO] Updated {target_file}")
