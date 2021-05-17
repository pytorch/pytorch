#!/usr/bin/env python3

from pathlib import Path

import jinja2

DOCKER_REGISTRY = "308535385114.dkr.ecr.us-east-1.amazonaws.com"

GITHUB_DIR = Path(__file__).parent.parent

CPU_TEST_RUNNER = "linux.2xlarge"
CUDA_TEST_RUNNER = "linux.8xlarge.nvidia.gpu"


class PyTorchLinuxWorkflow:
    def __init__(
            self,
            build_environment: str,
            docker_image_base: str,
            on_pull_request: bool = False,
            enable_doc_jobs: bool = False,
    ):
        self.build_environment = build_environment
        self.docker_image_base = docker_image_base
        self.test_runner_type = CPU_TEST_RUNNER
        self.on_pull_request = on_pull_request
        self.enable_doc_jobs = enable_doc_jobs
        if "cuda" in build_environment:
            self.test_runner_type = CUDA_TEST_RUNNER

    def generate_workflow_file(
        self, workflow_template: jinja2.Template, jinja_env: jinja2.Environment
    ) -> Path:
        output_file_path = GITHUB_DIR.joinpath(
            f"workflows/{self.build_environment}.yml"
        )
        with open(output_file_path, "w") as output_file:
            output_file.writelines(["# @generated DO NOT EDIT MANUALLY\n"])
            output_file.write(
                workflow_template.render(
                    build_environment=self.build_environment,
                    docker_image_base=self.docker_image_base,
                    test_runner_type=self.test_runner_type,
                    enable_doc_jobs=self.enable_doc_jobs,
                    on_pull_request=self.on_pull_request,
                )
            )
            output_file.write('\n')
        return output_file_path


WORKFLOWS = [
    PyTorchLinuxWorkflow(
        build_environment="pytorch-linux-xenial-py3.6-gcc5.4",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        on_pull_request=True,
        enable_doc_jobs=True,
    ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-paralleltbb-linux-xenial-py3.6-gcc5.4",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-parallelnative-linux-xenial-py3.6-gcc5.4",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-pure_torch-linux-xenial-py3.6-gcc5.4",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-gcc7",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc7",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-asan",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-asan",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang7-onnx",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang7-onnx",
    # ),
    PyTorchLinuxWorkflow(
        build_environment="pytorch-linux-xenial-cuda10.2-cudnn7-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
    ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-cuda11.1-cudnn8-py3.6-gcc7",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-libtorch-linux-xenial-cuda11.1-cudnn8-py3.6-gcc7",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-bionic-py3.6-clang9-noarch",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-xla-linux-bionic-py3.6-clang9",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-vulkan-linux-bionic-py3.6-clang9",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-bionic-py3.8-gcc9-coverage",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.8-gcc9",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-bionic-rocm3.9-py3.6",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm3.9-py3.6",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-android-ndk-r19c-x86_32",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-android-ndk-r19c-x86_64",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-android-ndk-r19c-arm-v7a",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-android-ndk-r19c-arm-v8a",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-mobile",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-asan",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-mobile-custom-dynamic",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-mobile-custom-static",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    # ),
    # PyTorchLinuxWorkflow(
    #     build_environment="pytorch-linux-xenial-py3.6-clang5-mobile-code-analysis",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    # ),
]

if __name__ == "__main__":
    jinja_env = jinja2.Environment(
        variable_start_string="!{{",
        loader=jinja2.FileSystemLoader(str(GITHUB_DIR.joinpath("templates"))),
    )
    workflow_template = jinja_env.get_template("linux_ci_workflow.yml.in")
    for workflow in WORKFLOWS:
        print(
            workflow.generate_workflow_file(
                workflow_template=workflow_template,
                jinja_env=jinja_env
            )
        )
