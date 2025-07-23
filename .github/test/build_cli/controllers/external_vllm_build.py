
from utils import run, get_post_build_pinned_commit, get_env, Timer, create_directory, delete_directory
import os

def build_vllm() -> None:
    print("begin")
    shared_folder_name="shared"

    tag_name = get_env("TAG", "vllm-wheels")
    cuda = get_env("CUDA_VERSION", "12.8.0")
    py = get_env("PYTHON_VERSION", "3.12")
    max_jobs = get_env("MAX_JOBS", "32")
    target = get_env("TARGET", "export-wheels")
    sccache_bucket_name=get_env("SCCACHE_BUCKET_NAME", "")
    sccache_region_name = get_env("SCCACHE_REGION_NAME", "")
    torch_cuda_arch_list = get_env("TORCH_CUDA_ARCH_LIST", "8.6;8.9")

    use_sccache = "0"
    if sccache_bucket_name and sccache_region_name:
        use_sccache = "1"

    # tracking the time of build
    with Timer():
        commit = get_post_build_pinned_commit("vllm")
        clone_vllm(commit)
        run("cp .github/script-v/Dockerfile.nightly_torch  vllm/docker/Dockerfile.nightly_torch", logging=True)

        create_directory(shared_folder_name)

        env = os.environ.copy()

        # run docker build for target stage `export-wheels` with output
        # this mount the root directory of the stage to targeted shared folder
        # in the host machine.
        cmd = f"""
        docker buildx build \
        --output type=local,dest=../{shared_folder_name} \
        -f docker/Dockerfile.nightly_torch \
        --build-arg max_jobs={max_jobs} \
        --build-arg CUDA_VERSION={cuda} \
        --build-arg PYTHON_VERSION={py} \
        --build-arg USE_SCCACHE={use_sccache} \
        --build-arg SCCACHE_BUCKET_NAME={sccache_bucket_name} \
        --build-arg SCCACHE_REGION_NAME={sccache_region_name} \
        --build-arg torch_cuda_arch_list={torch_cuda_arch_list} \
        --target {target} \
        -t {tag_name} \
        --progress=plain .
        """
        run(cmd, cwd="vllm", logging=True, env=env)


def clone_vllm(commit:str):
    cwd = "vllm"

    # delete the directory if it exists
    delete_directory(cwd)

    # Clone the repo & checkout commit
    run("git clone https://github.com/vllm-project/vllm.git")
    run(f"git checkout {commit}", cwd)
    run("git submodule update --init --recursive",cwd)
