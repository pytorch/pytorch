import logging
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.docker_helper import local_image_exists
from cli.lib.common.envs_helper import (
    env_bool_field,
    env_path_field,
    env_str_field,
    with_params_help,
)
from cli.lib.common.gh_summary import (
    gh_summary_path,
    summarize_content_from_file,
    summarize_wheels,
)
from cli.lib.common.path_helper import (
    copy,
    ensure_dir_exists,
    force_create_dir,
    get_path,
    is_path_exist,
)
from cli.lib.common.utils import run_command
from cli.lib.core.vllm.lib import clone_vllm, summarize_build_info


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
    # Otherwise docker build pull torch nightly during build
    # TORCH_WHEELS_PATH: directory containing local torch wheels when use_torch_whl is True
    use_torch_whl: bool = env_bool_field("USE_TORCH_WHEEL", True)
    torch_whls_path: Path = env_path_field("TORCH_WHEELS_PATH", "./dist")

    # USE_LOCAL_BASE_IMAGE: when true, use an existing local Docker base image; requires BASE_IMAGE
    # Otherwise, pull dockerfile's default image remotely
    # BASE_IMAGE: name:tag (only needed when use_local_base_image is True)
    use_local_base_image: bool = env_bool_field("USE_LOCAL_BASE_IMAGE", True)
    base_image: str = env_str_field("BASE_IMAGE")

    # USE_LOCAL_DOCKERFILE: when true("1"), use a local Dockerfile; requires DOCKERFILE_PATH.
    # otherwise, use vllm's default dockerfile.torch_nightly for build
    # DOCKERFILE_PATH: path to Dockerfile used when use_local_dockerfile is True"
    use_local_dockerfile: bool = env_bool_field("USE_LOCAL_DOCKERFILE", True)
    dockerfile_path: Path = env_path_field(
        "DOCKERFILE_PATH", ".github/ci_configs/vllm/Dockerfile.tmp_vllm"
    )

    # the cleaning script to remove torch dependencies from pip
    cleaning_script: Path = env_path_field(
        "cleaning_script", ".github/ci_configs/vllm/use_existing_torch.py"
    )

    # OUTPUT_DIR: where docker buildx (local exporter) will write artifacts
    output_dir: Path = env_path_field("OUTPUT_DIR", "external/vllm")

    # --- Build args ----------------------------------------------------------
    target_stage: str = env_str_field("TARGET_STAGE", "export-wheels")

    tag_name: str = env_str_field("TAG", "vllm-wheels")

    cuda_version: str = env_str_field("CUDA_VERSION", "12.8.1")

    python_version: str = env_str_field("PYTHON_VERSION", "3.12")

    max_jobs: str = env_str_field("MAX_JOBS", "64")

    sccache_bucket: str = env_str_field("SCCACHE_BUCKET")

    sccache_region: str = env_str_field("SCCACHE_REGION")

    torch_cuda_arch_list: str = env_str_field("TORCH_CUDA_ARCH_LIST", "8.9")

    def __post_init__(self):
        checks = [
            (
                self.use_torch_whl,  # flag
                True,  # trigger_value
                "torch_whls_path",  # resource
                is_path_exist,  # check_func
                "TORCH_WHEELS_PATH is not provided, but USE_TORCH_WHEEL is set to 1",
            ),
            (
                self.use_local_base_image,
                True,
                "base_image",
                local_image_exists,
                f"BASE_IMAGE {self.base_image} does not found, but USE_LOCAL_BASE_IMAGE is set to 1",
            ),
            (
                self.use_local_dockerfile,
                True,
                "dockerfile_path",
                is_path_exist,
                " DOCKERFILE_PATH path does not found, but USE_LOCAL_DOCKERFILE is set to 1",
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


@with_params_help(VllmBuildParameters)
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
        logger.info("Running vllm build with inputs: %s", inputs)
        vllm_commit = clone_vllm()

        self.cp_torch_cleaning_script(inputs)
        self.cp_dockerfile_if_exist(inputs)
        # cp torch wheels from root direct to vllm workspace if exist
        self.cp_torch_whls_if_exist(inputs)

        # make sure the output dir to store the build artifacts exist
        ensure_dir_exists(Path(inputs.output_dir))

        cmd = self._generate_docker_build_cmd(inputs)
        logger.info("Running docker build: \n %s", cmd)

        try:
            run_command(cmd, cwd="vllm", env=os.environ.copy())
        finally:
            self.genearte_vllm_build_summary(vllm_commit, inputs)

    def genearte_vllm_build_summary(
        self, vllm_commit: str, inputs: VllmBuildParameters
    ):
        if not gh_summary_path():
            return logger.info("Skipping, not detect GH Summary env var....")
        logger.info("Generate GH Summary ...")
        # summarize vllm build info
        summarize_build_info(vllm_commit)

        # summarize vllm build artifacts
        vllm_artifact_dir = inputs.output_dir / "wheels"
        summarize_content_from_file(
            vllm_artifact_dir,
            "build_summary.txt",
            title="Vllm build env pip package summary",
        )
        summarize_wheels(
            inputs.torch_whls_path, max_depth=3, title="Torch Wheels Artifacts"
        )
        summarize_wheels(vllm_artifact_dir, max_depth=3, title="Vllm Wheels Artifacts")

    def cp_torch_whls_if_exist(self, inputs: VllmBuildParameters) -> str:
        if not inputs.use_torch_whl:
            return ""
        tmp_dir = f"./{self.work_directory}/{_VLLM_TEMP_FOLDER}"
        tmp_path = Path(tmp_dir)
        force_create_dir(tmp_path)
        copy(inputs.torch_whls_path, tmp_dir)
        return tmp_dir

    def cp_torch_cleaning_script(self, inputs: VllmBuildParameters):
        script = get_path(inputs.cleaning_script, resolve=True)
        vllm_script = Path(f"./{self.work_directory}/use_existing_torch.py")
        copy(script, vllm_script)

    def cp_dockerfile_if_exist(self, inputs: VllmBuildParameters):
        if not inputs.use_local_dockerfile:
            logger.info("using vllm default dockerfile.torch_nightly for build")
            return
        dockerfile_path = get_path(inputs.dockerfile_path, resolve=True)
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
        abs_path = get_path(path, resolve=True)
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
                --build-arg max_jobs={inputs.max_jobs} \
                --build-arg CUDA_VERSION={inputs.cuda_version} \
                --build-arg PYTHON_VERSION={inputs.python_version} \
                --build-arg USE_SCCACHE={int(bool(inputs.sccache_bucket and inputs.sccache_region))} \
                --build-arg SCCACHE_BUCKET_NAME={inputs.sccache_bucket} \
                --build-arg SCCACHE_REGION_NAME={inputs.sccache_region} \
                --build-arg torch_cuda_arch_list='{inputs.torch_cuda_arch_list}' \
                --target {inputs.target_stage} \
                -t {inputs.tag_name} \
                --progress=plain .
        """
        ).strip()
