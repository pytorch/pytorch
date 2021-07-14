#!/usr/bin/env python3

from pathlib import Path
from typing import Any, Dict, Optional

import jinja2
from typing_extensions import Literal

DOCKER_REGISTRY = "308535385114.dkr.ecr.us-east-1.amazonaws.com"

GITHUB_DIR = Path(__file__).resolve().parent.parent


# it would be nice to statically specify that build_environment must be
# present, but currently Python has no easy way to do that
# https://github.com/python/mypy/issues/4617
PyTorchWorkflow = Dict[str, Any]

YamlShellBool = Literal["''", 1]

WINDOWS_CPU_TEST_RUNNER = "windows.4xlarge"
WINDOWS_CUDA_TEST_RUNNER = "windows.8xlarge.nvidia.gpu"


def PyTorchWindowsWorkflow(
    *,
    build_environment: str,
    test_runner_type: str,
    cuda_version: str,
    on_pull_request: bool = False,
    only_build_on_pull_request: bool = False,
    num_test_shards: int = 1,
    is_scheduled: Optional[str] = None,
) -> PyTorchWorkflow:
    return {
        "build_environment": build_environment,
        "test_runner_type": test_runner_type,
        "cuda_version": cuda_version,
        "on_pull_request": on_pull_request,
        "only_build_on_pull_request": only_build_on_pull_request and on_pull_request,
        "is_scheduled": is_scheduled,
        "num_test_shards": num_test_shards,
    }


LINUX_CPU_TEST_RUNNER = "linux.2xlarge"
LINUX_CUDA_TEST_RUNNER = "linux.8xlarge.nvidia.gpu"


def PyTorchLinuxWorkflow(
    *,
    build_environment: str,
    docker_image_base: str,
    test_runner_type: str,
    on_pull_request: bool = False,
    enable_doc_jobs: bool = False,
    enable_jit_legacy_test: YamlShellBool = "''",
    enable_multigpu_test: YamlShellBool = "''",
    enable_nogpu_no_avx_test: YamlShellBool = "''",
    enable_nogpu_no_avx2_test: YamlShellBool = "''",
    enable_slow_test: YamlShellBool = "''",
    num_test_shards: int = 1,
    is_scheduled: Optional[str] = None,
) -> PyTorchWorkflow:
    return {
        "build_environment": build_environment,
        "docker_image_base": docker_image_base,
        "test_runner_type": test_runner_type,
        "on_pull_request": on_pull_request,
        "is_scheduled": is_scheduled,
        "enable_doc_jobs": enable_doc_jobs,
        "enable_jit_legacy_test": enable_jit_legacy_test,
        "enable_multigpu_test": enable_multigpu_test,
        "enable_nogpu_no_avx_test": enable_nogpu_no_avx_test,
        "enable_nogpu_no_avx2_test": enable_nogpu_no_avx2_test,
        "enable_slow_test": enable_slow_test,
        "num_test_shards": num_test_shards,
    }


def generate_workflow_file(
    *,
    workflow: PyTorchWorkflow,
    workflow_template: jinja2.Template,
) -> Path:
    output_file_path = GITHUB_DIR / f"workflows/{workflow['build_environment']}.yml"
    with open(output_file_path, "w") as output_file:
        GENERATED = "generated"
        output_file.writelines([f"# @{GENERATED} DO NOT EDIT MANUALLY\n"])
        output_file.write(workflow_template.render(**workflow))
        output_file.write("\n")
    return output_file_path


WINDOWS_WORKFLOWS = [
    PyTorchWindowsWorkflow(
        build_environment="pytorch-win-vs2019-cpu-py3",
        cuda_version="cpu",
        test_runner_type=WINDOWS_CPU_TEST_RUNNER,
        on_pull_request=True,
        num_test_shards=2,
    ),
    PyTorchWindowsWorkflow(
        build_environment="pytorch-win-vs2019-cuda10-cudnn7-py3",
        cuda_version="10.1",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        on_pull_request=True,
        num_test_shards=2,
    ),
    PyTorchWindowsWorkflow(
        build_environment="pytorch-win-vs2019-cuda11-cudnn8-py3",
        cuda_version="11.1",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
    ),
    PyTorchWindowsWorkflow(
        build_environment="periodic-pytorch-win-vs2019-cuda11-cudnn8-py3",
        cuda_version="11.3",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        is_scheduled="45 0,4,8,12,16,20 * * *",
    ),
]

LINUX_WORKFLOWS = [
    PyTorchLinuxWorkflow(
        build_environment="pytorch-linux-xenial-py3.6-gcc5.4",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        on_pull_request=True,
        enable_doc_jobs=True,
        num_test_shards=2,
    ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-paralleltbb-linux-xenial-py3.6-gcc5.4",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-parallelnative-linux-xenial-py3.6-gcc5.4",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-pure_torch-linux-xenial-py3.6-gcc5.4",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-gcc7",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc7",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-asan",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-asan",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang7-onnx",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang7-onnx",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    PyTorchLinuxWorkflow(
        build_environment="pytorch-linux-bionic-cuda10.2-cudnn7-py3.9-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda10.2-cudnn7-py3.9-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
    ),
    PyTorchLinuxWorkflow(
        build_environment="pytorch-linux-xenial-cuda10.2-cudnn7-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        enable_jit_legacy_test=1,
        enable_multigpu_test=1,
        enable_nogpu_no_avx_test=1,
        enable_nogpu_no_avx2_test=1,
        enable_slow_test=1,
        num_test_shards=2,
    ),
    PyTorchLinuxWorkflow(
        build_environment="pytorch-linux-xenial-cuda11.1-cudnn8-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
    ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-libtorch-linux-xenial-cuda11.1-cudnn8-py3.6-gcc7",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
    #     test_runner_type=LINUX_CUDA_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-bionic-py3.6-clang9-noarch",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-xla-linux-bionic-py3.6-clang9",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-vulkan-linux-bionic-py3.6-clang9",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    PyTorchLinuxWorkflow(
        build_environment="pytorch-linux-bionic-py3.8-gcc9-coverage",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.8-gcc9",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        on_pull_request=True,
        num_test_shards=2,
    ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-bionic-rocm3.9-py3.6",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm3.9-py3.6",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-android-ndk-r19c-x86_32",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-android-ndk-r19c-x86_64",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-android-ndk-r19c-arm-v7a",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-android-ndk-r19c-arm-v8a",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-mobile",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-asan",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-mobile-custom-dynamic",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-mobile-custom-static",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-mobile-code-analysis",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
]


BAZEL_WORKFLOWS = [
    PyTorchLinuxWorkflow(
        build_environment="pytorch-linux-xenial-py3.6-gcc7-bazel-test",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc7",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
    ),
]

if __name__ == "__main__":
    jinja_env = jinja2.Environment(
        variable_start_string="!{{",
        loader=jinja2.FileSystemLoader(str(GITHUB_DIR.joinpath("templates"))),
    )
    template_and_workflows = [
        (jinja_env.get_template("linux_ci_workflow.yml.j2"), LINUX_WORKFLOWS),
        (jinja_env.get_template("windows_ci_workflow.yml.j2"), WINDOWS_WORKFLOWS),
        (jinja_env.get_template("bazel_ci_workflow.yml.j2"), BAZEL_WORKFLOWS),
    ]
    for template, workflows in template_and_workflows:
        for workflow in workflows:
            print(generate_workflow_file(workflow=workflow, workflow_template=template))
