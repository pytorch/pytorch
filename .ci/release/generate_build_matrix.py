#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
    * Latest XPU

Some basics about this script:
- It is designed to be run in a github action
- You should be able to run it locally on ANY machine with python 3.10+
- It SHOULD NOT depend on anything outside of the standard library
- It is written in a way that should make it easy to understand what builds
  are being generated
- We should prefer to write this in a way that avoids extra business logic,
  and instead relies on the inputs to drive the output
"""

import argparse
import copy
import json
import os
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent

STABLE_CUDA_VERSION = "12.6"


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
        return subprocess.run(
            ["git", "rev-parse", "HEAD:.ci/docker"],
            check=True,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        # If the git command fails, we're probably not in a git repo, so we'll just use the default tag
        return "main"


# Dictionary of extra install requirements keyed by (accelerator_type, accelerator_version)
EXTRA_INSTALL_REQUIREMENTS: dict[tuple[str, str], list[str]] = {
    # CUDA 11.8
    ("cuda", "11.8"): [
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
    # CUDA 12.6
    ("cuda", "12.6"): [
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
    # CUDA 12.8
    ("cuda", "12.8"): [
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
    # XPU
    ("xpu", ""): [
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
}


@dataclass
class BinaryBuild:
    """
    Base class representing a PyTorch binary build configuration.

    This class contains common attributes and methods used across different types of builds
    (CPU, CUDA, ROCm). It defines the build environment, testing infrastructure, and container
    image specifications.

    Attributes:
        operating_system (OperatingSystem): Operating system for this build
        cpu_arch (CpuArch): CPU architecture (x86_64, aarch64, or s390x)
        accelerator_version (str): Version of the accelerator (e.g., CUDA/ROCm version)
        accelerator_type (str): Type of accelerator (cpu, cuda, or rocm)
        builds_on (str): Github Actions runner for building
        tests_on (str): Github Actions runner for testing
        arch_list (list[str]): List of specific architectures to build for
    """

    operating_system: OperatingSystem
    cpu_arch: CpuArch
    accelerator_version: str
    accelerator_type: str
    builds_on: str = field(default="linux.12xlarge.memory.ephemeral")
    tests_on: str = field(default="linux.4xlarge")
    arch_list: list[str] = field(default_factory=list)

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

    def get_extra_install_requirements(self) -> list[str]:
        """Returns the extra install requirements for this build configuration

        We need to keep the install requirements for packages 'mostly' consistent across
        packages with like accelerator types + versions for compat with package managers like uv + poetry.
        This is because they only read METADATA from the first wheel found on the index, which can be inconsistent
        across different accelerator types + versions.

        TODO: We need to also ensure that packages uploaded to PyPI are consistent with their requirements as well,
              but to do that we will probably need to modify the wheel prior to uploading it to PyPI since doing
              these dependencies for wheels published on download.pytorch.org can create tricky situations. Like
              where cpu binary packages for linux depend on cuda dependencies.

        See https://github.com/pytorch/pytorch/issues/146679 for more details.
        """
        key = (self.accelerator_type, self.accelerator_version)
        requirements = EXTRA_INSTALL_REQUIREMENTS.get(key, [])
        return requirements


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


MANYWHEEL_CPU_BUILDS: list[CpuBuild] = [
    # CPU x86_64 - Linux
    CpuBuild(
        operating_system=OperatingSystem.LINUX,
        cpu_arch=CpuArch.X86_64,
        builds_on="linux.12xlarge.memory.ephemeral",
        tests_on="linux.4xlarge",
    ),
    # CPU ARM64 - Linux
    CpuBuild(
        operating_system=OperatingSystem.LINUX,
        cpu_arch=CpuArch.AARCH64,
        builds_on="linux.arm64.m7g.4xlarge.ephemeral",
        tests_on="linux.arm64.2xlarge",
    ),
    # CPU S390X - Linux
    CpuBuild(
        cpu_arch=CpuArch.S390X,
        operating_system=OperatingSystem.LINUX,
        builds_on="linux.s390x",
        tests_on="linux.s390x",
    ),
    # CPU x86_64 - Windows
    CpuBuild(
        operating_system=OperatingSystem.WINDOWS,
        cpu_arch=CpuArch.X86_64,
        builds_on="windows.4xlarge",
        tests_on="windows.4xlarge",
    ),
    # CPU ARM64 - Windows
    CpuBuild(
        cpu_arch=CpuArch.AARCH64,
        operating_system=OperatingSystem.WINDOWS,
        builds_on="windows-11-arm64",
        tests_on="windows-11-arm64",
    ),
    # CPU ARM64 - macOS
    CpuBuild(
        cpu_arch=CpuArch.AARCH64,
        operating_system=OperatingSystem.MACOS,
        builds_on="macos-14-xlarge",
        tests_on="macos-14-xlarge",
    ),
]

MANYWHEEL_CUDA_BUILDS: list[CudaBuild] = [
    # NOTE: Also update the CUDA sources in tools/nightly.py when changing this list
    # CUDA 11.8 x86_64 - Linux
    CudaBuild(
        accelerator_version="11.8",
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.LINUX,
    ),
    # CUDA 11.8 x86_64 - Windows
    CudaBuild(
        accelerator_version="11.8",
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.WINDOWS,
        builds_on="windows.4xlarge",
        tests_on="windows.g4dn.xlarge",
    ),
    # CUDA 12.6 x86_64 - Linux
    CudaBuild(
        accelerator_version="12.6",
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.LINUX,
    ),
    # CUDA 12.6 x86_64 - Windows
    CudaBuild(
        accelerator_version="12.6",
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.WINDOWS,
        builds_on="windows.4xlarge",
        tests_on="windows.g4dn.xlarge",
    ),
    # CUDA 12.8 x86_64 - Linux
    CudaBuild(
        accelerator_version="12.8",
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.LINUX,
    ),
    # CUDA 12.8 x86_64 - Windows
    CudaBuild(
        accelerator_version="12.8",
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.WINDOWS,
        builds_on="windows.4xlarge",
        tests_on="windows.g4dn.xlarge",
    ),
    # CUDA 12.8 AARCH64 - Linux
    CudaBuild(
        accelerator_version="12.8",
        cpu_arch=CpuArch.AARCH64,
        operating_system=OperatingSystem.LINUX,
        builds_on="linux.arm64.m7g.4xlarge.ephemeral",
        # We skip tests for ARM64 builds
        tests_on="",
    ),
]

MANYWHEEL_ROCM_BUILDS: list[RocmBuild] = [
    # ROCm 6.2.4 x86_64
    RocmBuild(
        accelerator_version="6.2.4",
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.LINUX,
    ),
    # ROCm 6.3 x86_64
    RocmBuild(
        accelerator_version="6.3",
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.LINUX,
    ),
]

MANYWHEEL_XPU_BUILDS: list[XpuBuild] = [
    # XPU x86_64
    XpuBuild(
        cpu_arch=CpuArch.X86_64,
        operating_system=OperatingSystem.LINUX,
    ),
]

MANYWHEEL_BUILDS: list[BinaryBuild] = [
    *MANYWHEEL_CPU_BUILDS,
    *MANYWHEEL_CUDA_BUILDS,
    *MANYWHEEL_ROCM_BUILDS,
    *MANYWHEEL_XPU_BUILDS,
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
            key = (build.accelerator_type, build.accelerator_version)

            wheel_nccl_version = ""
            requirements = EXTRA_INSTALL_REQUIREMENTS.get(key, [])
            for requirement in requirements:
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
    python_versions: list[str] | None = None,
) -> list[dict[str, str]]:
    if python_versions is None:
        python_versions = FULL_PYTHON_VERSIONS
    validate_nccl_dep_consistency()

    ret: list[dict[str, str]] = []
    for build in MANYWHEEL_BUILDS:
        if build.accelerator_type != accelerator_type:
            continue
        for python_version in python_versions:
            if build.operating_system != os:
                continue
            accelerator_info = f"{build.accelerator_type}{build.accelerator_version}"
            build_name = f"wheel-{build.cpu_arch}-py{python_version}-{accelerator_info}"
            matrix = {
                "python_version": python_version,
                "accelerator_type": build.accelerator_type,
                "accelerator_version": build.accelerator_version,
                "container_image": build.container_image(),
                "package_type": "wheel",
                "pytorch_extra_install_requirements": build.get_extra_install_requirements(),
                "build_name": build_name,
                "builds_on": build.builds_on,
                "tests_on": build.tests_on,
            }
            # We use deepcopy to avoid mutating the original matrix object for static builds down below
            ret.append(copy.deepcopy(matrix))
            # Special static build to use on Colab. x86_64 builds with Python 3.11 for 12.6 CUDA
            if all(
                [
                    build.cpu_arch == CpuArch.X86_64,
                    build.accelerator_type == "cuda",
                    build.accelerator_version == "12.6",
                    python_version == "3.11",
                    # Only add static build for Linux
                    build.operating_system == OperatingSystem.LINUX,
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
        "--package-type",
        choices=["wheel", "libtorch"],
        required=True,
        help="Type of binary package to build (wheel libtorch)",
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
        "--output",
        choices=["json", "github_output"],
        required=False,
        help="Output format (json, github_output)",
    )
    parser.add_argument(
        "--to-github-output",
        action="store_true",
        help="Output the matrix to the GitHub Actions output file",
    )

    args = parser.parse_args()

    # Map string OS argument to OperatingSystem enum
    os_map = {
        "linux": OperatingSystem.LINUX,
        "windows": OperatingSystem.WINDOWS,
        "macos": OperatingSystem.MACOS,
    }

    if args.package_type == "wheel":
        # Generate the matrix
        matrix = generate_wheels_matrix(
            os=os_map[args.os],
            accelerator_type=args.accelerator_type,
            python_versions=args.python_versions,
        )
    elif args.package_type == "libtorch":
        # TODO: Implement libtorch matrix generation
        pass
    else:
        raise ValueError(f"Invalid --package-type: {args.package_type}")

    # Pretty printed JSON for human readability
    print(json.dumps(matrix, indent=2))

    # Output the matrix
    if args.to_github_output:
        # Print the matrix in the format expected by GitHub Actions
        with open(os.environ["GITHUB_OUTPUT"], "w") as f:
            print(f"matrix={json.dumps(matrix)}", file=f)
