import logging
import os
import textwrap
from dataclasses import dataclass, field

from cli.lib.type.build import BuildRunner, LinuxExternalBuildBaseConfig


logger = logging.getLogger(__name__)

from cli.lib.common.file_utils import (
    clone_external_repo,
    ensure_dir_exists,
    force_create_dir,
    get_abs_path,
    is_path_exist,
    local_image_exists,
)
from cli.lib.common.utils import get_env, run_cmd


# default path for docker build artifacts
_DEFAULT_RESULT_PATH = "./shared"

# temp folder in vllm to cp torch whls in vllm work directory for docker build
_VLLM_TEMP_FOLDER = "tmp"


@dataclass
class VllmBuildConfig(LinuxExternalBuildBaseConfig):
    """
    Configuration specific to vLLM build jobs.
    """

    artifact_dir: str = ""
    torch_whl_dir: str = ""
    base_image: str = ""
    dockerfile_path: str = ""
    target: str = field(default_factory=lambda: get_env("TARGET", "export-wheels"))
    tag_name: str = field(default_factory=lambda: get_env("TAG", "vllm-wheels"))


class VllmBuildRunner(BuildRunner):
    def __init__(self, config_path: str = ""):
        super().__init__(config_path)
        self.cfg = VllmBuildConfig()
        self.work_directory = "vllm"

    def prepare(self):
        """
        Prepare the vllm build environment:
        - clone vllm repo with  pinned commit
        - create result dir if it does not exist
        - copy torch whls to vllm work directory if provided
        - copy user provided dockerfile to vllm work directory if provided
        """
        clone_vllm()
        cfg = self._to_vllm_build_config()
        self.cfg = cfg
        logger.info(f"setup vllm build config: {self.cfg}")

        ensure_dir_exists(self.cfg.artifact_dir)
        self.cp_dockerfile_if_exist()
        self.cp_torch_whls_if_exist()

    def run(self):
        """
        main function to run vllm build
        1. prepare vllm build environment
        2. prepare the docker build command args
        3. run docker build
        """
        self.prepare()
        logger.info(f"Running vllm build: {self.cfg}")
        torch_arg = _get_torch_wheel_path_arg(self.cfg.torch_whl_dir)
        base_arg, final_base_img, pull_flag = _get_base_image_args(self.cfg.base_image)
        cmd = _generate_docker_build_cmd(
            self.cfg, torch_arg, base_arg, final_base_img, pull_flag
        )
        logger.info(f"Running docker build: \n{cmd}")
        run_cmd(cmd, cwd="vllm", env=os.environ.copy())

    def _to_vllm_build_config(self):
        external_build_config = self.get_external_build_config()
        base_image = external_build_config.get("base_image", "")
        artifact_dir = self.get_result_path(
            external_build_config.get("artifact_dir", "")
        )
        abs_whl_dir = get_abs_path(external_build_config.get("torch_whl_dir", ""))
        dockerfile_path = get_abs_path(external_build_config.get("dockerfile_path", ""))
        config = VllmBuildConfig(
            artifact_dir=artifact_dir,
            torch_whl_dir=abs_whl_dir,
            base_image=base_image,
            dockerfile_path=dockerfile_path,
        )
        return config

    def cp_torch_whls_if_exist(self):
        if not self.cfg.torch_whl_dir:
            logger.info(
                "torch whl dir not provided, using default setting when build vllm with torch nightly"
            )
            return
        if not is_path_exist(self.cfg.torch_whl_dir):
            raise ValueError(
                f"torch whl dir is provided: {self.cfg.torch_whl_dir}, but it does not exist"
            )
        tmp_dir = f"./{self.work_directory}/{_VLLM_TEMP_FOLDER}"
        force_create_dir(tmp_dir)
        run_cmd(f"cp -a {self.cfg.torch_whl_dir}/. {tmp_dir}", log_cmd=True)

    def cp_dockerfile_if_exist(self):
        if self.cfg.dockerfile_path:
            logger.info(f"use user provided dockerfile {self.cfg.dockerfile_path}")
            run_cmd(
                f"cp {self.cfg.dockerfile_path} ./vllm/docker/Dockerfile.nightly_torch",
            )
        else:
            logger.info("using vllm default dockerfile.torch_nightly for build")

    def get_result_path(self, path):
        """
        Get the absolute path of the result path
        """
        if not path:
            path = _DEFAULT_RESULT_PATH
        abs_path = get_abs_path(path)
        return abs_path


def _get_torch_wheel_path_arg(torch_whl_dir: str) -> str:
    if not torch_whl_dir:
        return ""
    return f"--build-arg TORCH_WHEELS_PATH={_VLLM_TEMP_FOLDER}"


def _get_base_image_args(base_image: str) -> tuple[str, str, str]:
    """
    Returns:
        - base_image_arg: docker buildx arg string for base image
        - pull_flag: --pull=true or --pull=false depending on whether the image exists locally
    """
    pull_flag = ""
    if not base_image:
        return "", "", ""

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
    cfg: VllmBuildConfig,
    torch_arg: str,
    base_image_arg: str,
    final_base_image_arg: str,
    pull_flag: str,
) -> str:
    return textwrap.dedent(
        f"""
        docker buildx build \
            --output type=local,dest={cfg.artifact_dir} \
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
        target="vllm", repo="https://github.com/vllm-project/vllm.git", cwd="vllm"
    )
