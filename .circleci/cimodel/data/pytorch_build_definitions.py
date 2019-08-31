#!/usr/bin/env python3

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import cimodel.lib.miniutils as miniutils


DOCKER_IMAGE_PATH_BASE = "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/"

DOCKER_IMAGE_VERSION = 339

@dataclass
class Spec:
    base_name: str
    phase: str
    resource_class: str = "large"
    extra_params: Optional[str] = None
    dependency: "Optional[Spec]" = None
    run_only_on_master: bool = False

    @property
    def job_name(self):
        """
        This is used to uniquely specify the job in a workflow in CircleCI
        """
        full_name = self.base_name
        if self.extra_params:
            full_name += "_" + self.extra_params

        return full_name.replace(".", "_").replace("-", "_") + "_" + self.phase

    @property
    def docker_image(self):
        return DOCKER_IMAGE_PATH_BASE + self.base_name + ":" + str(DOCKER_IMAGE_VERSION)

    @property
    def build_environment(self):
        if self.extra_params is not None:
            extra_params = "-" + self.extra_params
        else:
            extra_params = ""

        return self.base_name + extra_params + "-" + self.phase

    def to_job_dict(self):
        """
        Returns a dictionary suitable for placing into a workflow's `jobs` list.
        """
        use_cuda_docker_runtime = "1" if "gpu" in self.resource_class else ""
        job_params = OrderedDict(
            {
                "name": self.job_name,
                "build_environment": miniutils.quote(self.build_environment),
                "docker_image": miniutils.quote(self.docker_image),
                "resource_class": self.resource_class,
                "use_cuda_docker_runtime": miniutils.quote(use_cuda_docker_runtime),
            }
        )

        requires = ["setup"]
        if self.dependency:
            requires.append(self.dependency.job_name)
        job_params["requires"] = requires

        if self.run_only_on_master:
            job_params["filters"] = {"branches": {"only": ["master", "/ci-all\/.*/"]}}

        base_job_name = "pytorch_linux_" + self.phase
        return {base_job_name: job_params}


@dataclass
class NamedTensorSpec(Spec):
    @property
    def docker_image(self):
        # NamedTensor jobs re-use the regular docker images
        docker_image = super().docker_image
        return docker_image.replace("namedtensor-", "")


@dataclass
class XLASpec(Spec):
    @property
    def docker_image(self):
        # XLA jobs re-use the regular docker images
        docker_image = super().docker_image
        return docker_image.replace("xla-", "")


specs = []

specs += Spec(base_name="pytorch-linux-xenial-py2.7.9", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py2.7.9", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py2.7", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py2.7", phase="test", dependency=specs[-1], run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3.5", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3.5", phase="test", dependency=specs[-1], run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-pynightly", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-pynightly", phase="test", dependency=specs[-1], run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc4.8", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc4.8", phase="test", dependency=specs[-1], run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc5.4", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc5.4", phase="test", dependency=specs[-1]),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-py3.6-gcc5.4", phase="build"),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-py3.6-gcc5.4", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc7", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3.6-gcc7", phase="test", dependency=specs[-1], run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-asan", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-asan", phase="test", dependency=specs[-1]),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-py3-clang5-asan", phase="build"),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-py3-clang5-asan", phase="test", dependency=specs[-1]),
specs += XLASpec(base_name="pytorch-xla-linux-xenial-py3.6-clang7", phase="build"),
specs += XLASpec(base_name="pytorch-xla-linux-xenial-py3.6-clang7", phase="test", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py2", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py2", phase="test", resource_class="gpu.medium", dependency=specs[-1], run_only_on_master=True),

gpu_py3_spec = Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", phase="build")
specs += gpu_py3_spec,
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", phase="test", resource_class="gpu.medium", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="multigpu", resource_class="gpu.large", phase="test", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="NO_AVX2", phase="test", resource_class="gpu.medium", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="NO_AVX-NO_AVX2", phase="test", resource_class="gpu.medium", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="slow", phase="test", resource_class="gpu.medium", dependency=gpu_py3_spec),
specs += Spec(base_name="pytorch-linux-xenial-cuda9-cudnn7-py3", extra_params="nogpu", phase="test", dependency=gpu_py3_spec),

specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-cuda9-cudnn7-py2", phase="build"),
specs += NamedTensorSpec(base_name="pytorch-namedtensor-linux-xenial-cuda9-cudnn7-py2", phase="test", resource_class="gpu.medium", dependency=specs[-1]),
specs += Spec(base_name="pytorch-linux-xenial-cuda9.2-cudnn7-py3-gcc7", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-cuda9.2-cudnn7-py3-gcc7", phase="test", resource_class="gpu.medium", dependency=specs[-1], run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-cuda10-cudnn7-py3-gcc7", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-cuda10.1-cudnn7-py3-gcc7", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-cuda10.1-cudnn7-py3-gcc7", phase="test", resource_class="gpu.medium", dependency=specs[-1], run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-android-ndk-r19c", extra_params="x86_32", phase="build"),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-android-ndk-r19c", extra_params="x86_64", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-android-ndk-r19c", extra_params="arm-v7a", phase="build", run_only_on_master=True),
specs += Spec(base_name="pytorch-linux-xenial-py3-clang5-android-ndk-r19c", extra_params="arm-v8a", phase="build", run_only_on_master=True),


def get_workflow_jobs():
    jobs = ["setup"]
    for spec in specs:
        jobs.append(spec.to_job_dict())

    # special jobs that we hard code
    for job in [
        "pytorch_short_perf_test_gpu",
        "pytorch_python_doc_push",
        "pytorch_cpp_doc_push",
    ]:
        jobs.append(
            {job: {"requires": ["pytorch_linux_xenial_cuda9_cudnn7_py3_build"]}}
        )

    return jobs
