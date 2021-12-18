#!/usr/bin/env python3

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Any

import jinja2
from typing_extensions import Literal, TypedDict

Arch = Literal["windows", "linux", "macos"]


class Config(TypedDict):
    num_shards: int
    runner: str


DOCKER_REGISTRY = "308535385114.dkr.ecr.us-east-1.amazonaws.com"
GITHUB_DIR = Path(__file__).resolve().parent.parent

WINDOWS_CPU_TEST_RUNNER = "windows.4xlarge"
# contains 1 gpu
WINDOWS_CUDA_TEST_RUNNER = "windows.8xlarge.nvidia.gpu"
WINDOWS_RUNNERS = {
    WINDOWS_CPU_TEST_RUNNER,
    WINDOWS_CUDA_TEST_RUNNER,
}

LINUX_CPU_TEST_RUNNER = "linux.2xlarge"
# contains 1 gpu
LINUX_CUDA_TEST_RUNNER = "linux.4xlarge.nvidia.gpu"
LINUX_RUNNERS = {
    LINUX_CPU_TEST_RUNNER,
    LINUX_CUDA_TEST_RUNNER,
}

MACOS_TEST_RUNNER_10_15 = "macos-10.15"
MACOS_TEST_RUNNER_11 = "macos-11"


@dataclass
class CIWorkflow:
    # Required fields
    arch: Arch
    build_environment: str
    template: str

    # Optional fields
    test_runner_type: str = ""
    cuda_version: str = ""
    docker_image_base: str = ""
    enable_doc_jobs: bool = False
    exclude_test: bool = False
    build_generates_artifacts: bool = True
    build_with_debug: bool = False
    is_scheduled: str = ""
    num_test_shards: int = 1
    only_run_smoke_tests_on_pull_request: bool = False
    num_test_shards_on_pull_request: int = -1
    timeout_after: int = 240
    xcode_version: str = ""

    # The following variables will be set as environment variables,
    # so it's easier for both shell and Python scripts to consume it if false is represented as the empty string.
    enable_jit_legacy_test: bool = False
    enable_distributed_test: bool = True
    enable_multigpu_test: bool = False
    enable_nogpu_no_avx_test: bool = False
    enable_nogpu_no_avx2_test: bool = False
    enable_slow_test: bool = False
    enable_docs_test: bool = False
    enable_backwards_compat_test: bool = False
    enable_xla_test: bool = False
    enable_noarch_test: bool = False
    enable_force_on_cpu_test: bool = False

    def __post_init__(self) -> None:
        if not self.build_generates_artifacts:
            self.exclude_test = True

        # If num_test_shards_on_pull_request is not user-defined, default to num_test_shards unless we are
        # only running smoke tests on the pull request.
        if self.num_test_shards_on_pull_request == -1:
            # Don't run the default if we are only running smoke tests
            if self.only_run_smoke_tests_on_pull_request:
                self.num_test_shards_on_pull_request = 0
            else:
                self.num_test_shards_on_pull_request = self.num_test_shards
        self.assert_valid()

    def assert_valid(self) -> None:
        err_message = (
            f"invalid test_runner_type for {self.arch}: {self.test_runner_type}"
        )
        if self.arch == "linux":
            assert self.test_runner_type in LINUX_RUNNERS, err_message
        if self.arch == "windows":
            assert self.test_runner_type in WINDOWS_RUNNERS, err_message

        if self.build_with_debug:
            assert self.build_environment.endswith("-debug")

    def normalized_build_environment(self, suffix: str) -> str:
        return self.build_environment.replace(".", "_") + suffix

    def build_name(self) -> str:
        return self.normalized_build_environment("-build")

    def test_name(self) -> str:
        return self.normalized_build_environment("-test")

    def docs_name(self) -> str:
        return self.normalized_build_environment("-docs")

    def test_jobs(self) -> Any:
        if self.arch == "linux":
            MULTIGPU_RUNNER_TYPE = "linux.16xlarge.nvidia.gpu"
            DISTRIBUTED_GPU_RUNNER_TYPE = "linux.8xlarge.nvidia.gpu"
            NOGPU_RUNNER_TYPE = "linux.2xlarge"
        elif self.arch == "windows":
            DISTRIBUTED_GPU_RUNNER_TYPE = self.test_runner_type
            NOGPU_RUNNER_TYPE = "windows.4xlarge"

        test_jobs = []

        # todo: no run_as_if_on_trunk implemented
        num_test_shards = self.num_test_shards_on_pull_request

        configs: Dict[str, Config] = {}
        if self.enable_jit_legacy_test:
            configs["jit_legacy"] = {"num_shards": 1, "runner": self.test_runner_type}
        if self.enable_multigpu_test:
            configs["multigpu"] = {"num_shards": 1, "runner": MULTIGPU_RUNNER_TYPE}

        if self.enable_nogpu_no_avx_test:
            configs["nogpu_NO_AVX"] = {"num_shards": 1, "runner": NOGPU_RUNNER_TYPE}
        if self.enable_nogpu_no_avx2_test:
            configs["nogpu_NO_AVX2"] = {"num_shards": 1, "runner": NOGPU_RUNNER_TYPE}
        if self.enable_force_on_cpu_test:
            configs["force_on_cpu"] = {"num_shards": 1, "runner": NOGPU_RUNNER_TYPE}
        if self.enable_distributed_test:
            configs["distributed"] = {
                "num_shards": 1,
                "runner": DISTRIBUTED_GPU_RUNNER_TYPE
                if "cuda" in str(self.build_environment)
                else self.test_runner_type,
            }
        if self.enable_slow_test:
            configs["slow"] = {"num_shards": 1, "runner": self.test_runner_type}
        if self.enable_docs_test:
            configs["docs_test"] = {"num_shards": 1, "runner": self.test_runner_type}
        if self.enable_backwards_compat_test:
            configs["backwards_compat"] = {
                "num_shards": 1,
                "runner": self.test_runner_type,
            }
        if self.enable_xla_test:
            configs["xla"] = {"num_shards": 1, "runner": self.test_runner_type}
        if self.enable_noarch_test:
            configs["noarch"] = {"num_shards": 1, "runner": self.test_runner_type}

        run_smoke_tests = self.only_run_smoke_tests_on_pull_request
        if run_smoke_tests:
            configs["smoke_tests"] = {"num_shards": 1, "runner": self.test_runner_type}

        for name, config in configs.items():
            for shard in range(1, config["num_shards"] + 1):
                test_jobs.append(
                    {
                        "name": self.test_name()
                        + f"_{name}_{shard}_{config['num_shards']}",
                        "config": name,
                        "shard": shard,
                        "num_shards": config["num_shards"],
                        "runner": config["runner"],
                    }
                )

        for shard in range(1, num_test_shards + 1):
            test_jobs.append(
                {
                    "name": self.test_name() + f"_default_{shard}_{num_test_shards}",
                    "config": "default",
                    "shard": shard,
                    "num_shards": num_test_shards,
                    "runner": self.test_runner_type,
                }
            )
        return test_jobs


@dataclass
class DockerWorkflow:
    build_environment: str
    docker_images: List[str]

    # Optional fields
    cuda_version: str = ""
    is_scheduled: str = ""


PULL_JOBS = [
    CIWorkflow(
        arch="windows",
        build_environment="win-vs2019-cpu-py3",
        template="job/windows_job.yml.j2",
        cuda_version="cpu",
        test_runner_type=WINDOWS_CPU_TEST_RUNNER,
        num_test_shards=2,
    ),
    CIWorkflow(
        arch="windows",
        build_environment="win-vs2019-cuda11.3-py3",
        template="job/windows_job.yml.j2",
        cuda_version="11.3",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        only_run_smoke_tests_on_pull_request=True,
        enable_force_on_cpu_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-gcc5.4",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        enable_jit_legacy_test=True,
        enable_backwards_compat_test=True,
        enable_docs_test=True,
        num_test_shards=2,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-docs",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        enable_doc_jobs=True,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc7",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=2,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3-clang5-mobile-build",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-asan",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3-clang5-mobile-custom-build-static",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-clang7-asan",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang7-asan",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=3,
        enable_distributed_test=False,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-clang7-onnx",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang7-onnx",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=2,
        enable_distributed_test=False,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-cuda11.3-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.3-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-bionic-py3.6-clang9",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=2,
        enable_distributed_test=False,
        enable_noarch_test=True,
        enable_xla_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-vulkan-bionic-py3.6-clang9",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=1,
        enable_distributed_test=False,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-linux-xenial-cuda10.2-py3-gcc7-slow-gradcheck",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        template="job/linux_job.yml.j2",
        num_test_shards=2,
        enable_distributed_test=False,
        timeout_after=360,
        # Only run this on master 4 times per day since it does take a while
        is_scheduled="0 */4 * * *",
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-cuda11.3-py3.6-gcc7-bazel-test",
        template="job/bazel_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.3-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-custom-build-single",
        template="job/android_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-custom-build-single-full-jit",
        template="job/android_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        exclude_test=True,
    ),
]

TRUNK_JOBS = [
    CIWorkflow(
        arch="linux",
        build_environment="parallelnative-linux-xenial-py3.6-gcc5.4",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
    ),
    # Build PyTorch with BUILD_CAFFE2=ON
    CIWorkflow(
        arch="linux",
        build_environment="caffe2-linux-xenial-py3.6-gcc5.4",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-bionic-cuda10.2-py3.9-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda10.2-cudnn7-py3.9-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        enable_jit_legacy_test=True,
        enable_multigpu_test=True,
        enable_nogpu_no_avx_test=True,
        enable_nogpu_no_avx2_test=True,
        enable_slow_test=True,
        num_test_shards=2,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="libtorch-linux-xenial-cuda10.2-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64-coreml",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64-full-jit",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64-custom-ops",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64-metal",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-x86-64",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-x86-64-coreml",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-x86-64-full-jit",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    # Distributed tests are still run on MacOS, but part of regular shards
    CIWorkflow(
        arch="macos",
        build_environment="macos-11-py3-x86-64",
        template="job/macos_job.yml.j2",
        xcode_version="12.4",
        test_runner_type=MACOS_TEST_RUNNER_11,
        num_test_shards=2,
        enable_distributed_test=False,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="macos-10-15-py3-lite-interpreter-x86-64",
        template="job/macos_job.yml.j2",
        xcode_version="12",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        build_generates_artifacts=False,
    ),
    CIWorkflow(
        arch="macos",
        build_environment="macos-10-15-py3-arm64",
        template="job/macos_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="libtorch-linux-xenial-cuda11.3-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.3-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="pytorch-linux-xenial-py3-clang5-android-ndk-r19c-build",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
        template="job/android_full_job.yml.j2",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        exclude_test=True,
    ),
]

PERIODIC_JOBS = [
    CIWorkflow(
        arch="windows",
        build_environment="periodic-win-vs2019-cuda11.5-py3",
        template="job/windows_job.yml.j2",
        cuda_version="11.5",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        enable_force_on_cpu_test=True,
        is_scheduled="45 4,10,16,22 * * *",
    ),
    CIWorkflow(
        arch="windows",
        build_environment="periodic-win-vs2019-cuda11.1-py3",
        template="job/windows_job.yml.j2",
        cuda_version="11.1",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        is_scheduled="45 0,4,8,12,16,20 * * *",
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-linux-bionic-cuda11.5-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda11.5-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        is_scheduled="45 4,10,16,22 * * *",
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-libtorch-linux-bionic-cuda11.5-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda11.5-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        build_generates_artifacts=False,
        is_scheduled="45 4,10,16,22 * * *",
        exclude_test=True,
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-linux-xenial-cuda11.1-py3.6-gcc7-debug",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        build_with_debug=True,
        is_scheduled="45 0,4,8,12,16,20 * * *",
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-libtorch-linux-xenial-cuda11.1-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
        is_scheduled="45 0,4,8,12,16,20 * * *",
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-linux-xenial-cuda10.2-py3-gcc7-slow-gradcheck",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        enable_distributed_test=False,
        timeout_after=360,
        # Only run this on master 4 times per day since it does take a while
        is_scheduled="0 */4 * * *",
    ),
]

NIGHTLY_JOBS = [
    CIWorkflow(
        arch="linux",
        build_environment="linux-docs-push",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        enable_doc_jobs=True,
        exclude_test=True,
        is_scheduled="0 0 * * *",  # run pushes only on a nightly schedule
    ),
]

DOCKER_IMAGES = {
    f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda10.2-cudnn7-py3.6-clang9",  # for pytorch/xla
    f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm4.1-py3.6",  # for rocm
    f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm4.2-py3.6",  # for rocm
    f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm4.3.1-py3.6",  # for rocm
}

DOCKER_IMAGES.update(
    {
        workflow.docker_image_base
        for workflow in [*PULL_JOBS, *PERIODIC_JOBS, *TRUNK_JOBS]
        if workflow.docker_image_base
    }
)

DOCKER_WORKFLOW = DockerWorkflow(
    build_environment="docker-builds",
    docker_images=sorted(DOCKER_IMAGES),
    # Run weekly to ensure they can build
    is_scheduled="1 * */7 * *",
)


def main() -> None:
    jinja_env = jinja2.Environment(
        variable_start_string="!{{",
        loader=jinja2.FileSystemLoader(str(GITHUB_DIR.joinpath("templates"))),
        undefined=jinja2.StrictUndefined,
    )

    def generate_workflow(
        jobs: List[CIWorkflow], template_path: str, output_path: str
    ) -> None:
        p: Path = GITHUB_DIR / "workflows" / output_path
        template = jinja_env.get_template(template_path)
        content = template.render(jobs=jobs)

        with open(p, "w") as output_file:
            output_file.write(content)

    generate_workflow(PULL_JOBS, "pull.yml.j2", "pull.yml")
    generate_workflow(PERIODIC_JOBS, "periodic.yml.j2", "periodic.yml")
    generate_workflow(TRUNK_JOBS, "trunk.yml.j2", "trunk.yml")

    # Write docker workflow:
    template = jinja_env.get_template("docker_builds.yml.j2")
    with open(GITHUB_DIR / "workflows" / "docker_builds.yml", "w") as f:
        f.write(template.render(**asdict(DOCKER_WORKFLOW)))


if __name__ == "__main__":
    main()
