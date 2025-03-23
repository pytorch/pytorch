#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
    * Latest XPU
"""

import argparse
import copy
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Optional


# Include function name, line number, and timestamp
formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(funcName)s:%(lineno)d:%(message)s"
)

# Create and configure handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Configure logger
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

ROOT_DIR = Path(__file__).parent.parent.parent


def run_command(command: list[str], cwd: Optional[str] = None) -> str:
    logger.debug("Running command: %s in %s", " ".join(command), cwd)
    return subprocess.run(
        command, check=True, cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


class CpuArch:
    X86_64: str = "x86_64"
    AARCH64: str = "aarch64"
    S390X: str = "s390x"


class OperatingSystem:
    LINUX: str = "linux"
    WINDOWS: str = "windows"
    MACOS: str = "macos"


@cache
def _generate_docker_tag() -> str:
    """
    Returns the version for the container image by checking the git hash of the .ci/docker directory.

    If we're not in a git repo, we'll just use the default tag.
    """
    try:
        return run_command(["git", "rev-parse", "HEAD:.ci/docker"], cwd=ROOT_DIR)
    except subprocess.CalledProcessError:
        # If the git command fails, we're probably not in a git repo, so we'll just use the default tag
        return "main"


@dataclass
class BinaryBuild:
    """
    Base class representing a PyTorch binary build configuration.

    This class contains common attributes and methods used across different types of builds
    (CPU, CUDA, ROCm). It defines the build environment, testing infrastructure, and container
    image specifications.

    Attributes:
        operating_systems (list[OperatingSystem]): List of supported operating systems for this build
        cpu_arch (CpuArch): CPU architecture (x86_64, aarch64, or s390x)
        accelerator_version (str): Version of the accelerator (e.g., CUDA/ROCm version)
        accelerator_type (str): Type of accelerator (cpu, cuda, or rocm)
        builds_on (str): AWS EC2 instance type for building
        tests_on (str): AWS EC2 instance type for testing
        arch_list (list[str]): List of specific architectures to build for
        extra_install_requirements (list[str]): Additional packages needed for the build
    """

    operating_systems: list[OperatingSystem]
    cpu_arch: CpuArch
    accelerator_version: str
    accelerator_type: str
    builds_on: str = field(default="linux.12xlarge.memory.ephemeral")
    tests_on: str = field(default="linux.4xlarge")
    arch_list: list[str] = field(default_factory=list)
    extra_install_requirements: list[str] = field(default_factory=list)

    def container_image(self) -> str:
        """
        Returns the container image for the build.

        Format: pytorch/wheel-build:{cpu_arch}-{accelerator_type}{accelerator_version}-{DEFAULT_TAG}
        Produces images that look like:
        - pytorch/wheel-build:x86_64-cpu-e8d4567
        - pytorch/wheel-build:x86_64-cuda12.8-e8d4567
        - pytorch/wheel-build:aarch64-cuda12.8-e8d4567
        - pytorch/wheel-build:x86_64-rocm6.2.4-e8d4567
        """
        tag = _generate_docker_tag()
        return f"pytorch/wheel-build:{str(self.cpu_arch)}-{self.accelerator_type}{self.accelerator_version}-{tag}"


@dataclass
class CpuBuild(BinaryBuild):
    accelerator_type: str = "cpu"
    accelerator_version: str = ""


@dataclass
class CudaBuild(BinaryBuild):
    cudnn_version: str = "9"
    accelerator_type: str = "cuda"
    tests_on: str = "linux.4xlarge.nvidia.gpu"


@dataclass
class StableCudaBuild(CudaBuild):
    is_stable: bool = True


@dataclass
class RocmBuild(BinaryBuild):
    accelerator_type: str = "rocm"
    tests_on: str = "linux.rocm.gpu"


@dataclass
class XpuBuild(BinaryBuild):
    accelerator_type: str = "xpu"
    accelerator_version: str = ""
    builds_on: str = "linux.s390x"
    tests_on: str = "linux.s390x"


MANYWHEEL_BUILDS: list[BinaryBuild] = [
    # CPU x86_64
    CpuBuild(
        operating_systems=[OperatingSystem.LINUX, OperatingSystem.WINDOWS],
        cpu_arch=CpuArch.X86_64,
    ),
    # CPU ARM64
    CpuBuild(
        cpu_arch=CpuArch.AARCH64,
        operating_systems=[
            OperatingSystem.LINUX,
            OperatingSystem.WINDOWS,
            OperatingSystem.MACOS,
        ],
        builds_on="linux.arm64.m7g.4xlarge.ephemeral",
        tests_on="linux.arm64.2xlarge",
    ),
    # CPU S390X
    CpuBuild(
        cpu_arch=CpuArch.S390X,
        operating_systems=[OperatingSystem.LINUX],
    ),
    # NOTE: Also update the CUDA sources in tools/nightly.py when changing this list
    # CUDA x86_64
    CudaBuild(
        accelerator_version="11.8",
        cpu_arch=CpuArch.X86_64,
        operating_systems=[OperatingSystem.LINUX, OperatingSystem.WINDOWS],
        extra_install_requirements=[
            "nvidia-cuda-nvrtc-cu11==11.8.89; platform_system == 'Linux' and platform_machine == 'x86_64' | "  # noqa: B950
            "nvidia-cuda-runtime-cu11==11.8.89; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cuda-cupti-cu11==11.8.87; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cudnn-cu11==9.1.0.70; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cublas-cu11==11.11.3.6; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cufft-cu11==10.9.0.58; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-curand-cu11==10.3.0.86; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusolver-cu11==11.4.1.48; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusparse-cu11==11.7.5.86; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nccl-cu11==2.21.5; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nvtx-cu11==11.8.86; platform_system == 'Linux' and platform_machine == 'x86_64'"
        ],
    ),
    CudaBuild(
        accelerator_version="12.6",
        cpu_arch=CpuArch.X86_64,
        operating_systems=[OperatingSystem.LINUX, OperatingSystem.WINDOWS],
        extra_install_requirements=[
            "nvidia-cuda-nvrtc-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cuda-runtime-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cuda-cupti-cu12==12.6.80; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cudnn-cu12==9.5.1.17; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cublas-cu12==12.6.4.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cufft-cu12==11.3.0.4; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-curand-cu12==10.3.7.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusolver-cu12==11.7.1.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusparse-cu12==12.5.4.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusparselt-cu12==0.6.3; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nccl-cu12==2.26.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nvtx-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nvjitlink-cu12==12.6.85; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cufile-cu12==1.11.1.6; platform_system == 'Linux' and platform_machine == 'x86_64'"
        ],
    ),
    CudaBuild(
        accelerator_version="12.8",
        cpu_arch=CpuArch.X86_64,
        operating_systems=[OperatingSystem.LINUX, OperatingSystem.WINDOWS],
        extra_install_requirements=[
            "nvidia-cuda-nvrtc-cu12==12.8.61; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cuda-runtime-cu12==12.8.57; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cuda-cupti-cu12==12.8.57; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cudnn-cu12==9.8.0.87; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cublas-cu12==12.8.3.14; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cufft-cu12==11.3.3.41; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-curand-cu12==10.3.9.55; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusolver-cu12==11.7.2.55; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusparse-cu12==12.5.7.53; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusparselt-cu12==0.6.3; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nccl-cu12==2.26.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nvtx-cu12==12.8.55; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nvjitlink-cu12==12.8.61; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cufile-cu12==1.13.0.11; platform_system == 'Linux' and platform_machine == 'x86_64'"
        ],
    ),
    # CUDA ARM64
    CudaBuild(
        accelerator_version="12.8",
        cpu_arch=CpuArch.AARCH64,
        operating_systems=[OperatingSystem.LINUX],
        extra_install_requirements=[
            "nvidia-cuda-nvrtc-cu12==12.8.61; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cuda-runtime-cu12==12.8.57; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cuda-cupti-cu12==12.8.57; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cudnn-cu12==9.8.0.87; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cublas-cu12==12.8.3.14; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cufft-cu12==11.3.3.41; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-curand-cu12==10.3.9.55; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusolver-cu12==11.7.2.55; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusparse-cu12==12.5.7.53; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cusparselt-cu12==0.6.3; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nccl-cu12==2.26.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nvtx-cu12==12.8.55; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-nvjitlink-cu12==12.8.61; platform_system == 'Linux' and platform_machine == 'x86_64' | "
            "nvidia-cufile-cu12==1.13.0.11; platform_system == 'Linux' and platform_machine == 'x86_64'"
        ],
        builds_on="linux.arm64.m7g.4xlarge.ephemeral",
        # We skip tests for ARM64 builds
        tests_on="",
    ),
    # NOTE: Also update the ROCm sources in tools/nightly.py when changing this list
    # ROCm x86_64
    RocmBuild(
        accelerator_version="6.2.4",
        cpu_arch=CpuArch.X86_64,
        operating_systems=[OperatingSystem.LINUX],
    ),
    RocmBuild(
        accelerator_version="6.3",
        cpu_arch=CpuArch.X86_64,
        operating_systems=[OperatingSystem.LINUX],
    ),
    # XPU x86_64
    XpuBuild(
        cpu_arch=CpuArch.X86_64,
        operating_systems=[OperatingSystem.LINUX],
        extra_install_requirements=[
            "intel-cmplr-lib-rt==2025.0.4; platform_system == 'Linux' | "
            "intel-cmplr-lib-ur==2025.0.4; platform_system == 'Linux' | "
            "intel-cmplr-lic-rt==2025.0.4; platform_system == 'Linux' | "
            "intel-sycl-rt==2025.0.4; platform_system == 'Linux' | "
            "intel-cmplr-lib-rt==2025.0.5; platform_system == 'Windows' | "
            "intel-cmplr-lib-ur==2025.0.5; platform_system == 'Windows' | "
            "intel-cmplr-lic-rt==2025.0.5; platform_system == 'Windows' | "
            "intel-sycl-rt==2025.0.5; platform_system == 'Windows' | "
            "tcmlib==1.2.0 | "
            "umf==0.9.1 | "
            "intel-pti==0.10.1"
        ],
    ),
]


def read_nccl_pin(arch_version: str) -> str:
    nccl_pin_path = os.path.join(
        ROOT_DIR,
        ".ci",
        "docker",
        "ci_commit_pins",
        f"nccl-cu{arch_version[:2]}.txt",
    )
    with open(nccl_pin_path) as f:
        return f.read().strip()


def validate_nccl_dep_consistency() -> None:
    for build in MANYWHEEL_BUILDS:
        if isinstance(build, CudaBuild):
            nccl_release_tag = read_nccl_pin(build.accelerator_version)

            wheel_nccl_version = ""
            for requirement in build.extra_install_requirements:
                if requirement.startswith("nvidia-nccl-cu"):
                    wheel_nccl_version = requirement.split("==")[1]
                    break
            if not nccl_release_tag.startswith(f"v{wheel_nccl_version}"):
                raise RuntimeError(
                    f"{build.accelerator_version} NCCL release tag version "
                    f"{nccl_release_tag} does not correspond to wheel version "
                    f"{wheel_nccl_version}"
                )


FULL_PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13", "3.13t"]


def generate_wheels_matrix(
    os: OperatingSystem,
    accelerator_type: str,
    python_versions: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    if python_versions is None:
        python_versions = FULL_PYTHON_VERSIONS
    validate_nccl_dep_consistency()

    ret: list[dict[str, str]] = []
    for build in MANYWHEEL_BUILDS:
        if build.accelerator_type != accelerator_type:
            continue
        for python_version in python_versions:
            accelerator_info = f"{build.accelerator_type}{build.accelerator_version}"
            build_name = f"wheel-{build.cpu_arch}-py{python_version}-{accelerator_info}"
            matrix = {
                "python_version": python_version,
                "accelerator_type": build.accelerator_type,
                "accelerator_version": build.accelerator_version,
                "container_image": build.container_image(),
                "package_type": "wheel",
                "pytorch_extra_install_requirements": build.extra_install_requirements,
                "build_name": build_name,
                "builds_on": build.builds_on,
                "tests_on": build.tests_on,
            }
            if os in build.operating_systems:
                # We use deepcopy to avoid mutating the original matrix object for static builds down below
                ret.append(copy.deepcopy(matrix))
            # Special static build to use on Colab. x86_64 builds with Python 3.11 for 12.6 CUDA
            if all(
                [
                    build.cpu_arch == CpuArch.X86_64,
                    build.accelerator_type == "cuda",
                    build.accelerator_version == "12.6",
                    python_version == "3.11",
                ]
            ):
                matrix.update(
                    {
                        # This is a special build that doesn't need any extra install requirements
                        # passing this as empty signals to build_cuda.sh to build statically
                        "pytorch_extra_install_requirements": [],
                        "build_name": f"{build_name}-static",
                    }
                )

                ret.append(matrix)
    return {"include": ret}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate binary build matrix for PyTorch wheels"
    )
    parser.add_argument(
        "--os",
        choices=["linux", "windows", "macos"],
        required=True,
        help="Operating system for the build (linux, windows, or macos)",
    )
    parser.add_argument(
        "--accelerator-type",
        choices=["cpu", "cuda", "rocm", "xpu"],
        required=True,
        help="Accelerator type (cpu, cuda, rocm, or xpu)",
    )
    parser.add_argument(
        "--python-versions",
        nargs="+",
        help="Python versions to build for (e.g., 3.9 3.10 3.11). If not specified, all supported versions will be used.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output the matrix as JSON (default is to print as indented JSON)",
    )

    args = parser.parse_args()

    # Map string OS argument to OperatingSystem enum
    os_map = {
        "linux": OperatingSystem.LINUX,
        "windows": OperatingSystem.WINDOWS,
        "macos": OperatingSystem.MACOS,
    }

    # Generate the matrix
    matrix = generate_wheels_matrix(
        os=os_map[args.os],
        accelerator_type=args.accelerator_type,
        python_versions=args.python_versions,
    )

    # Output the matrix
    if args.output_json:
        # Simple JSON output format
        print(json.dumps(matrix))
    else:
        # Pretty printed JSON for human readability
        print(json.dumps(matrix, indent=2))
