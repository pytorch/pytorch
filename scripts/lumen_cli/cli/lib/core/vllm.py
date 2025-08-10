import logging
import os
import textwrap
from dataclasses import dataclass, field

from cli.lib.common.git_utils import clone_external_repo
from cli.lib.common.type import BaseRunner

from scripts.lumen_cli.cli.lib.common.docker import local_image_exists


logger = logging.getLogger(__name__)

from cli.lib.common.file_utils import (
    ensure_dir_exists,
    force_create_dir,
    get_abs_path,
    is_path_exist,
)
from cli.lib.common.utils import get_env, run_cmd


# default path for docker build artifacts
_DEFAULT_RESULT_PATH = "./shared"

# temp folder in vllm to cp torch whls in vllm work directory for docker build
_VLLM_TEMP_FOLDER = "tmp"


@dataclass
class VllmBuildParameters:
    """
    Parameters controlling a vLLM build.

    Inputs come primarily from environment variables (see each field's comment).
    Flags are strings "0"/"1" (not bool). In __post_init__, when a flag is "1",
    the corresponding resource is validated and (if it’s a path) normalized to
    an absolute path via get_abs_path(). On failure, FileNotFoundError is raised.
    """

    # --- Pre-build feature flags (string "0"/"1") -----------------------------
    # USE_TORCH_WHEEL: when "1", use local Torch wheels; requires TORCH_WHEELS_PATH.
    #  Otherwise docker build pull torch nightly during build
    use_torch_whl: str = field(default_factory=lambda: get_env("USE_TORCH_WHEEL", "1"))

    # USE_LOCAL_BASE_IMAGE: when "1", use an existing local Docker base image; requires BASE_IMAGE
    # Otherwise, pull dockerfile's default image remotely
    use_local_base_image: str = field(
        default_factory=lambda: get_env("USE_LOCAL_BASE_IMAGE", "1")
    )

    # USE_LOCAL_DOCKERFILE: when "1", use a local Dockerfile; requires DOCKERFILE_PATH.
    # otherwise, use vllm's default dockerfile.torch_nightly for build
    use_local_dockerfile: str = field(
        default_factory=lambda: get_env("USE_LOCAL_DOCKERFILE", "1")
    )

    # --- Pre-build condition inputs ------------------------------------------
    # BASE_IMAGE: name:tag (only needed when use_local_base_image == "1")
    base_image: str = field(default_factory=lambda: get_env("BASE_IMAGE"))

    # DOCKERFILE_PATH: path to Dockerfile used when use_local_dockerfile == "1"
    dockerfile_path: str = field(
        default_factory=lambda: get_env(
            "DOCKERFILE_PATH", ".github/ci_configs/vllm/Dockerfile.tmp_vllm"
        )
    )

    # TORCH_WHEELS_PATH: directory containing local torch wheels when use_torch_whl == "1"
    torch_whls_path: str = field(
        default_factory=lambda: get_env("TORCH_WHEELS_PATH", "dist")
    )

    # --- Build output ---------------------------------------------------------
    # output_dir: where docker buildx (local exporter) will write artifacts
    output_dir: str = field(default_factory=lambda: get_env("output_dir", "shared"))

    def __post_init__(self):
        checks = [
            (
                self.use_torch_whl,  # flag
                "1",  # trigger_value
                "torch_whls_path",  # resource
                is_path_exist,  # check_func
                "torch whl path is not provided, but use_torch_whl is set to 1",
                get_abs_path,
            ),
            (
                self.use_local_base_image,
                "1",
                "base_image",
                local_image_exists,
                f"base_image {self.base_image} does not found, but use_local_base_image is set to 1",
                None,
            ),
            (
                self.use_local_dockerfile,
                "1",
                "dockerfile_path",
                is_path_exist,
                "dockerfile path does not found, but use_local_dockerfile is set to 1",
                get_abs_path,
            ),
        ]
        for flag, trigger_value, attr_name, check_func, error_msg, handler in checks:
            value = getattr(self, attr_name)
            if flag == trigger_value:
                if not value or not check_func(value):
                    raise FileNotFoundError(error_msg)
                if handler:
                    setattr(self, attr_name, handler(value))
                logger.info(f"found flag {flag} -> field  {attr_name}")
            else:
                logger.info(f"flag {flag} is not set")


@dataclass
class VllmDockerBuildArgs:
    output_dir: str = field(default_factory=lambda: get_env("output_dir", "shared"))
    target: str = field(default_factory=lambda: get_env("TARGET", "export-wheels"))
    tag_name: str = field(default_factory=lambda: get_env("TAG", "vllm-wheels"))
    cuda: str = field(default_factory=lambda: get_env("CUDA_VERSION", "12.8.1"))
    py: str = field(default_factory=lambda: get_env("PYTHON_VERSION", "3.12"))
    max_jobs: str = field(default_factory=lambda: get_env("MAX_JOBS", "64"))
    target: str = field(default_factory=lambda: get_env("TARGET", "export-wheels"))
    sccache_bucket: str = field(default_factory=lambda: get_env("SCCACHE_BUCKET"))
    sccache_region: str = field(default_factory=lambda: get_env("SCCACHE_REGION"))
    torch_cuda_arch_list: str = field(
        default_factory=lambda: get_env("TORCH_CUDA_ARCH_LIST", "8.0")
    )


class VllmBuildRunner(BaseRunner):
    """
    Build vLLM using docker buildx.

    Environment variable options:
        "USE_TORCH_WHEEL":      "1: use local wheels; 0: pull nightly from pypi",
        "USE_LOCAL_BASE_IMAGE": "1: use local base image; 0: default image",
        "USE_LOCAL_DOCKERFILE": "1: use local Dockerfile; 0: vllm repo default dockerfile.torch_nightly",
        "TORCH_WHEELS_PATH":    "Path to local wheels (when USE_TORCH_WHEEL=1)",
        "DOCKERFILE_PATH":      "Path to Dockerfile (when USE_LOCAL_DOCKERFILE=1)",
        "BASE_IMAGE":           "name:tag for local base image (when USE_LOCAL_BASE_IMAGE=1)",
        "TORCH_CUDA_ARCH_LIST": "e.g. '8.0' or '8.0;9.0'",
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
        torch_whl_path = self.cp_torch_whls_if_exist(inputs)

        ensure_dir_exists(inputs.output_dir)
        output_path = get_abs_path(inputs.output_dir)

        cmd = self._generate_docker_build_cmd(inputs, output_path, torch_whl_path)
        logger.info(f"Running docker build: \n{cmd}")
        run_cmd(cmd, cwd="vllm", env=os.environ.copy())

    def cp_torch_whls_if_exist(self, inputs) -> str:
        if inputs.use_torch_whl != "1":
            return ""

        if not inputs.torch_whls_path or not is_path_exist(inputs.torch_whls_path):
            raise FileNotFoundError(
                "torch whl path is not provided, but use_torch_whl is set to 1"
            )
        torch_whl_path = get_abs_path(inputs.torch_whls_path)

        tmp_dir = f"./{self.work_directory}/{_VLLM_TEMP_FOLDER}"
        force_create_dir(tmp_dir)
        run_cmd(f"cp -a {torch_whl_path}/. {tmp_dir}", log_cmd=True)
        return tmp_dir

    def cp_dockerfile_if_exist(self, inputs):
        if inputs.use_local_dockerfile == "0":
            logger.info("using vllm default dockerfile.torch_nightly for build")
            return

        if not inputs.dockerfile_path or not is_path_exist(inputs.dockerfile_path):
            raise FileNotFoundError(
                "dockerfile is not found, but use_local_dockerfile is set to 1"
            )
        dockerfile_path = get_abs_path(inputs.dockerfile_path)
        run_cmd(
            f"cp {dockerfile_path} ./vllm/docker/Dockerfile.nightly_torch",
        )

    def get_result_path(self, path):
        """
        Get the absolute path of the result path
        """
        if not path:
            path = _DEFAULT_RESULT_PATH
        abs_path = get_abs_path(path)
        return abs_path

    def _get_torch_wheel_path_arg(self, torch_whl_dir: str) -> str:
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
        if inputs.use_local_base_image == "0":
            return "", "", ""

        base_image = inputs.base_image

        # set both base image and final base image to the same local image
        base_image_arg = f"--build-arg BUILD_BASE_IMAGE={base_image}"
        final_base_image_arg = f"--build-arg FINAL_BASE_IMAGE={base_image}"

        if local_image_exists(base_image):
            logger.info(f"[INFO] Found local image: {base_image}")
            pull_flag = "--pull=false"
            return base_image_arg, final_base_image_arg, pull_flag
        logger.info(
            f"[INFO] Local image not found: {base_image}, will try to pull from remote"
        )
        return base_image_arg, final_base_image_arg, ""

    def _generate_docker_build_cmd(
        self,
        inputs: VllmBuildParameters,
        output_dir: str,
        torch_whl_path: str,
    ) -> str:
        cfg = VllmDockerBuildArgs()
        base_image_arg, final_base_image_arg, pull_flag = self._get_base_image_args(
            inputs
        )
        torch_arg = self._get_torch_wheel_path_arg(torch_whl_path)

        return textwrap.dedent(
            f"""
            docker buildx build \
                --output type=local,dest={output_dir} \
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


def clone_vllm():
    clone_external_repo(
        target="vllm",
        repo="https://github.com/vllm-project/vllm.git",
        dst="vllm",
        update_submodules=True,
    )
