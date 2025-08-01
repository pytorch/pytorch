import os
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any, final

from lib.utils import (
    clone_vllm,
    ensure_dir_exists,
    force_create_dir,
    get_abs_path,
    get_env,
    get_existing_abs_path,
    get_post_build_pinned_commit,
    run,
)

@dataclass
class VllmBuildConfig:
    tag_name: str = get_env("TAG", "vllm-wheels")
    cuda: str = get_env("CUDA_VERSION", "12.8.1")
    py: str = get_env("PYTHON_VERSION", "3.12")
    max_jobs: str = get_env("MAX_JOBS", "64")
    target: str = get_env("TARGET", "export-wheels")
    sccache_bucket: str = get_env("SCCACHE_BUCKET", "")
    sccache_region: str = get_env("SCCACHE_REGION", "")
    torch_cuda_arch_list: str = get_env("TORCH_CUDA_ARCH_LIST", "8.0")
    vllm_fa_cmake_gpu_arches = get_env("VLLM_FA_CMAKE_GPU_ARCHES", "80-real")
    dev = get_env("DEV")


_DEFAULT_RESULT_PATH = "./results"
_VLLM_TEMP_FOLDER = "tmp"


def local_image_exists(image: str):
    try:
        subprocess.check_output(
            ["docker", "image", "inspect", image], stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def prepare_artifact_dir(path: str):
    if not path:
        path = _DEFAULT_RESULT_PATH
    abs_path = get_abs_path(path)
    ensure_dir_exists(abs_path)
    return abs_path


def build_vllm(artifact_dir: str, torch_whl_dir: str, base_image: str):
    cfg = VllmBuildConfig()

    result_path = prepare_artifact_dir(artifact_dir)
    print(f"Target artifact dir path is {result_path}", flush=True)

    if cfg.dev:
        vllm_commit = get_post_build_pinned_commit("vllm",".")
    else:
        vllm_commit = get_post_build_pinned_commit("vllm")
    clone_vllm(vllm_commit)

    # replace dockerfile
    # todo: remove this once the dockerfile is updated in vllm
    dockerfile_path = f".github/docker/external/vllm/Dockerfile.base"
    if cfg.dev:
        dockerfile_path = "Dockerfile.base"

    run(
        f"cp {dockerfile_path} ./vllm/docker/Dockerfile.nightly_torch",
        logging=True,
    )

    torch_arg, _ = _prepare_torch_wheels(torch_whl_dir)
    base_arg, final_base_img,pull_flag = _get_base_image_args(base_image)
    cmd = _generate_docker_build_cmd(cfg, result_path, torch_arg, base_arg,final_base_img, pull_flag)
    print("Running docker build", flush=True)
    print(cmd, flush=True)
    run(cmd, cwd="vllm", logging=True, env=os.environ.copy())


def _prepare_torch_wheels(torch_whl_dir: str) -> tuple[str, str]:
    if not torch_whl_dir:
        return "", ""
    abs_whl_dir = get_existing_abs_path(torch_whl_dir)
    tmp_dir = f"./vllm/{_VLLM_TEMP_FOLDER}"
    force_create_dir(tmp_dir)
    run(f"cp -a {abs_whl_dir}/. {tmp_dir}", logging=True)
    return f"--build-arg TORCH_WHEELS_PATH={_VLLM_TEMP_FOLDER}", tmp_dir


def _get_base_image_args(base_image: str) -> tuple[str, str, str]:
    """
    Returns:
        - base_image_arg: docker buildx arg string for base image
        - pull_flag: --pull=true or --pull=false depending on whether the image exists locally
    """
    pull_flag = ""
    if not base_image:
        return "","",""

    base_image_arg = f"--build-arg BUILD_BASE_IMAGE={base_image}"
    final_base_image_arg = f"--build-arg FINAL_BASE_IMAGE={base_image}"
    if local_image_exists(base_image):
        print(f"[INFO] Found local image: {base_image}", flush=True)
        pull_flag = "--pull=false"
        return base_image_arg,final_base_image_arg, pull_flag
    print(
        f"[INFO] Local image not found: {base_image}, will try to pull from remote",
        flush=True,
    )
    return base_image_arg,final_base_image_arg, ""


def _generate_docker_build_cmd(
    cfg: VllmBuildConfig,
    result_path: str,
    torch_arg: str,
    base_image_arg: str,
    final_base_image_arg: str,
    pull_flag: str,
) -> str:
    return textwrap.dedent(f"""
        docker buildx build \
            --output type=local,dest={result_path} \
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
            --build-arg torch_cuda_arch_list={cfg.torch_cuda_arch_list} \
            --build-arg vllm_fa_cmake_gpu_arches={cfg.vllm_fa_cmake_gpu_arches}\
            --target {cfg.target} \
            -t {cfg.tag_name} \
            --progress=plain .
    """).strip()
