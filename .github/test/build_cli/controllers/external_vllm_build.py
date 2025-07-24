from utils import (
    ensure_dir_exists,
    force_create_dir,
    remove_dir,
    run,
    get_post_build_pinned_commit,
    get_env,
    Timer,
    get_abs_path,
    get_existing_abs_path,
)
import os

_DEFAULT_RESULT_PATH = "./results"
_VLLM_TEMP_FOLDER = "tmp"

def prepare_artifact_dir(path: str):
    if not path:
        path = _DEFAULT_RESULT_PATH
    abs_path = get_abs_path(path)
    ensure_dir_exists(abs_path)
    return abs_path

def get_torch_whl_path(path:str):
    torch_whl_abs_path = ""
    if path:
        torch_whl_abs_path = get_existing_abs_path(path)
        print(f"torch wheel path is {torch_whl_abs_path}")
    else:
        print("no torch whl path detected, vllm will be built using torch nightly")
    return torch_whl_abs_path

def build_vllm(artifact_dir: str = _DEFAULT_RESULT_PATH, torch_whl_dir="") -> None:
    result_path = prepare_artifact_dir(artifact_dir)
    print(f"Target artifact dir path is {result_path}")

    tag_name = get_env("TAG", "vllm-wheels")
    cuda = get_env("CUDA_VERSION", "12.8.0")
    py = get_env("PYTHON_VERSION", "3.12")
    max_jobs = get_env("MAX_JOBS", "32")
    target = get_env("TARGET", "export-wheels")
    sccache_bucket_name = get_env("SCCACHE_BUCKET_NAME", "")
    sccache_region_name = get_env("SCCACHE_REGION_NAME", "")
    torch_cuda_arch_list = get_env("TORCH_CUDA_ARCH_LIST", "8.6;8.9")

    use_sccache = "0"
    if sccache_bucket_name and sccache_region_name:
        use_sccache = "1"

    # tracking the time of build
    with Timer():
        commit = get_post_build_pinned_commit("vllm")
        clone_vllm(commit)

        run(
            "cp .github/script-v/Dockerfile.nightly_torch  vllm/docker/Dockerfile.nightly_torch",
            logging=True,
        )
        docker_torch_arg = ""
        if torch_whl_dir:
            torch_whl_abs_path = get_torch_whl_path(torch_whl_dir)
            # copy the torch wheel in tmp folder into the vllm's build context directory
            tmp_file = get_abs_path(f"./vllm/{_VLLM_TEMP_FOLDER}")
            force_create_dir(tmp_file)
            run(f"cp -a {torch_whl_abs_path}/. {tmp_file}",logging=True)
            print(f"constructing TORCH_WHEELS_PATH {_VLLM_TEMP_FOLDER}")
            docker_torch_arg = f"--build-arg TORCH_WHEELS_PATH={_VLLM_TEMP_FOLDER}"

        env = os.environ.copy()

        # run docker build for target stage `export-wheels` with output
        # this mount the root directory of the stage to targeted shared folder
        # in the host machine.
        cmd = f"""
        docker buildx build --no-cache \
        --output type=local,dest={result_path} \
        -f docker/Dockerfile.nightly_torch \
        {docker_torch_arg} \
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


def clone_vllm(commit: str):
    cwd = "vllm"

    # delete the directory if it exists
    remove_dir(cwd)

    # Clone the repo & checkout commit
    run("git clone https://github.com/vllm-project/vllm.git")
    run(f"git checkout {commit}", cwd)
    run("git submodule update --init --recursive", cwd)
