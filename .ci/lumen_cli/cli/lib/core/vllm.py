import logging
import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.docker_helper import local_image_exists
from cli.lib.common.envs_helper import (
    env_bool_field,
    env_path_field,
    env_str_field,
    get_env,
    with_params_help,
)
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
    pip_install_first_match,
    pip_install_packages,
    run_python,
)
from cli.lib.common.utils import run_cmd, run_shell, temp_environ, working_directory


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

    # OUTPUT_DIR: where docker buildx (local exporter) will write artifacts
    output_dir: Path = env_path_field("OUTPUT_DIR", "shared")

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
        tmp_dir = f"./{self.work_directory}/{_VLLM_TEMP_FOLDER}"
        tmp_path = Path(tmp_dir)
        force_create_dir(tmp_path)
        copy(inputs.torch_whls_path, tmp_dir)
        return tmp_dir

    def cp_dockerfile_if_exist(self, inputs: VllmBuildParameters):
        if not inputs.use_local_dockerfile:
            logger.info("using vllm default dockerfile.torch_nightly for build")
            return
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


@dataclass
class VllmTestParameters:
    """
    Parameters defining the vllm external test input

    !!!DO NOT ADD SECRETS IN THIS CLASS!!!
    you can put environment variable name in VllmTestParameters if it's not the same as the secret one
    fetch secrests directly from env variables during runtime
    """

    # TORCH_WHEELS_PATH: directory containing local torch wheels when use_torch_whl is True
    torch_whls_path: Path = env_path_field("TORCH_WHEELS_PATH", "./dist")
    vllm_whls_path: Path = env_path_field("VLLM_WHEELS_PATH", "./shared")
    torch_cuda_arch_list: str = env_str_field("TORCH_CUDA_ARCH_LIST", "8.9")

    def __post_init__(self):
        if not self.torch_whls_path.exists():
            raise ValueError("missing torch_whls_path")
        if not self.vllm_whls_path.exists():
            raise ValueError("missing vllm_whls_path")


class VllmTestRunner(BaseRunner):
    def __init__(self, args=None):
        self.work_directory = "vllm"
        self.test_name = args.test_name
        self.torch_whl_path = "torch*.whl"
        self.torch_whl_extra = "opt-einsum"
        self.torch_whl_relatvie_path = [
            "vision/torchvision*.whl",
            "audio/torchaudio*.whl",
        ]
        self.vllm_whl_relatvie_path = [
            "wheels/xformers/xformers*.whl",
            "wheels/vllm/vllm*.whl",
            "wheels/flashinfer-python/flashinfer*.whl",
        ]

    def _install_wheels(self, inputs: VllmTestParameters):
        logger.info("Running vllm test with inputs: %s", inputs)
        logger.info("Installing torch wheel")
        torch_p = f"{str(inputs.torch_whls_path)}/{self.torch_whl_path}"
        pip_install_first_match(torch_p, self.torch_whl_extra)

        logger.info("Installing other torch-related wheels")
        torch_whls_path = [
            f"{str(inputs.torch_whls_path)}/{whl_path}"
            for whl_path in self.torch_whl_relatvie_path
        ]
        for torch_whl in torch_whls_path:
            pip_install_first_match(torch_whl)
        logger.info("Done. Installed torch and other torch-related wheels ")

        logger.info("Installing vllm wheels")
        vllm_whls_path = [
            f"{str(inputs.vllm_whls_path)}/{whl_path}"
            for whl_path in self.vllm_whl_relatvie_path
        ]
        for vllm_whl in vllm_whls_path:
            pip_install_first_match(vllm_whl)
        logger.info("Done. Installed vllm wheels")

    def _install_test_dependencies(self):
        """
        Install test dependencies for vllm test
        This method replaces default torch dependencies with local whls in
        requirements/test.in file from vllm repo.

        Then generate the test.txt file using uv pip compile, along with requirements/test.txt as constrain to
        match workable packages' version. Notice during the uv pip compile. --constarint is a soft constraint.

        """
        # TODO(elainewy): move this as part of vllm build, to generate the test.txt file
        logger.info("generate test.txt from requirements/test.in with local torch whls")
        preprocess_test_in()
        copy(
            Path("requirements/test.in"),
            Path("snapshot_constraint.txt"),
            full_path=False,
        )
        run_cmd(
            f"{sys.executable} -m uv pip compile requirements/test.in "
            "-o test.txt "
            "--index-strategy unsafe-best-match "
            "--constraint snapshot_constraint.txt "
            "--torch-backend cu128"
        )
        logger.info("install requirements from test.txt")
        pip_install_packages(requirements="test.txt", prefer_uv=True)
        logger.info("Done. install requirements from test.txt")

        # install mambda from source since it does not work now with pip
        # TODO(elainewy): move this as part of vllm build
        pip_install_packages(
            packages=[
                "--no-build-isolation",
                "git+https://github.com/state-spaces/mamba@v2.2.4",
            ],
            prefer_uv=True,
        )
        logger.info("Done. installed requirements from test.txt")

    def _install_dependencies(self):
        logger.info("install vllm_test_util ...")
        pip_install_packages(packages=["-e", "tests/vllm_test_utils"], prefer_uv=True)
        logger.info("Done. installed vllm_test_utils")

        pip_install_packages(packages=["hf_transfer"], prefer_uv=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        logger.info("remove all torch packages from requirements txt")
        run_python("use_existing_torch.py")

        logger.info("install requirements from vllm work directory ...")
        for requirements in ["requirements/common.txt", "requirements/build.txt"]:
            pip_install_packages(
                requirements=requirements,
                prefer_uv=True,
            )
        logger.info("Done. installed requirements from vllm work directory")

    def check_versions(self):
        """
        check installed packages version
        """
        logger.info("double check installed packages")
        patterns = ["torch", "xformers", "torchvision", "torchaudio", "vllm"]
        for pkg in patterns:
            try:
                module = __import__(pkg)
                version = getattr(module, "__version__", None)
                version = version if version else "Unknown version"
                logger.info("%s: %s", pkg, version)
            except ImportError:
                logger.info(" %s: Not installed", pkg)
        logger.info("Done. checked installed packages")

    def _set_envs(self, inputs: VllmTestParameters):
        os.environ["TORCH_CUDA_ARCH_LIST"] = inputs.torch_cuda_arch_list
        os.environ["HF_TOKEN"] = os.getenv("VLLM_TEST_HUGGING_FACE_TOKEN", "")
        if not get_env("HF_TOKEN"):
            raise ValueError(
                "missing required HF_TOKEN, please set VLLM_TEST_HUGGING_FACE_TOKEN env var"
            )
        if not get_env("TORCH_CUDA_ARCH_LIST"):
            raise ValueError(
                "missing required TORCH_CUDA_ARCH_LIST, please set TORCH_CUDA_ARCH_LIST env var"
            )

    def run(self):
        """
        main function to run vllm test
        """
        inputs = VllmTestParameters()
        self._set_envs(inputs)
        clone_vllm()

        with working_directory(self.work_directory):
            remove_dir(Path("vllm"))
            self._install_wheels(inputs)
            self._install_dependencies()
            self.check_versions()

            # Must set env variables before running tests
            self.run_test(self.test_name)

    def run_test(self, test_name: str):
        logger.info("run tests.....")
        tests_map = sample_tests()
        if test_name not in tests_map:
            raise RuntimeError(
                f"test {test_name} not found, please add it to test pool"
            )
        tests = tests_map[test_name]
        logger.info("Running tests: %s", tests["title"])

        with temp_environ(tests.get("env_var", {})):
            failures = []
            for step in tests["steps"]:
                with (
                    temp_environ(step.get("env_var", {})),
                    working_directory(step.get("working_directory", "tests")),
                ):
                    code = run_shell(cmd=step["command"], check=False)
                    if code != 0:
                        failures.append(step)
            if failures:
                logger.error("Failed tests: %s", failures)
                raise RuntimeError(f"{len(failures)} pytest runs failed: {failures}")
            logger.info("Done. All tests passed")


def clone_vllm():
    clone_external_repo(
        target="vllm",
        repo="https://github.com/vllm-project/vllm.git",
        dst="vllm",
        update_submodules=True,
    )


def preprocess_test_in(
    target_file: str = "requirements/test.in", additional_packages: Iterable[str] = ()
):
    """
    remove torch packages in target_file and replace with local torch whls
    """
    additional_package_to_move = list(additional_packages or ())
    pkgs_to_remove = [
        "torch",
        "torchvision",
        "torchaudio",
        "xformers",
        "mamba_ssm",
    ] + additional_package_to_move
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
    logger.info("[INFO] Updated %s", target_file)


def sample_tests():
    # TODO(elainewy): add test.yaml to handle the env and tests
    return {
        "basic_correctness_test": {
            "title": "Basic Correctness Test",
            "id": "basic_correctness_test",
            "env_var": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "steps": [
                {
                    "command": "pytest -v -s basic_correctness/test_cumem.py",
                },
                {
                    "command": "pytest -v -s basic_correctness/test_basic_correctness.py",
                },
                {
                    "command": "pytest -v -s basic_correctness/test_cpu_offload.py",
                },
                {
                    "command": "pytest -v -s basic_correct",
                    "env_var": {
                        "VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT": "1",
                    },
                },
            ],
        }
    }
