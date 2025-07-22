
from typing import Dict
from utils import run, get_pinned_commit, get_env
import os
from pathlib import Path




def export_wheels_and_files(
    image_tag: str = "vllm-test",
    container_name: str = "vllm_tmp",
    export_dir: str = "./exported",
    extra_paths: dict[str, str] = {},
):
    os.makedirs(export_dir, exist_ok=True)

    # Create container from image
    run(f"docker create --name {container_name} {image_tag}")

    # Copy wheels
    wheels = {
        "vllm": "/vllm-workspace/vllm-dist",
        "xformers": "/vllm-workspace/xformers-dist",
        "flashinfer": "/vllm-workspace/flashinfer-dist",
    }
    for name, container_path in wheels.items():
        local_path = Path(export_dir) / "wheels" / name
        local_path.mkdir(parents=True, exist_ok=True)
        run(f"docker cp {container_name}:{container_path} {local_path}")

    # Copy extra files
    if extra_paths:
        for name, container_path in extra_paths.items():
            local_path = Path(export_dir) / "extras" / name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            run(f"docker cp {container_name}:{container_path} {local_path}")

    # Clean up container
    run(f"docker rm {container_name}")


def build_vllm() -> None:
    tag = get_env("TAG", "latest")
    cuda = get_env("CUDA_VERSION", "12.8.0")
    py = get_env("PYTHON_VERSION", "3.12")
    max_jobs = get_env("MAX_JOBS", "16")
    use_sccache = get_env("USE_SCCACHE", "")
    target = get_env("TARGET", "test")
    sccache_bucket_name=get_env("SCCACHE_BUCKET_NAME", "")
    sccache_region_name = get_env("SCCACHE_REGION_NAME", "")

    commit = get_pinned_commit("vllm")
    clone_vllm(commit)

    cmd = f"""
        docker build \
            -f docker/Dockerfile.nightly_torch \
            --build-arg max_jobs={max_jobs}
            --build-arg CUDA_VERSION={cuda} \
            --build-arg PYTHON_VERSION={py} \
            --build-arg USE_SCCACHE={use_sccache} \
            --build-arg SCCACHE_BUCKET_NAME={sccache_bucket_name} \
            --build-arg SCCACHE_REGION_NAME={sccache_region_name} \
            --target {target} \
            -t vllm:{tag} .
        """
    run(cmd, cwd="vllm")

    # run the the container 
    export_wheels_and_files()






def clone_vllm(commit:str):
    cwd = "vllm"
    # Clone the repo
    run("git clone https://github.com/vllm-project/vllm.git")
    run(f"git checkout {commit}", cwd)
    run("git submodule update --init --recursive",cwd)
