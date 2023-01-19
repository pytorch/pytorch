#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
"""


import argparse
import os
import sys
import json

from typing import Dict, List, Tuple, Optional

mod = sys.modules[__name__]

FULL_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]
ROCM_ARCHES = ["5.1.1", "5.2"]
CUDA_ACRHES_DICT = {
    "nightly": ["11.6", "11.7"],
    "test": ["11.6", "11.7"],
    "release": ["11.6", "11.7"],
}
PACKAGE_TYPES = ["wheel", "conda", "libtorch"]
PRE_CXX11_ABI = "pre-cxx11"
CXX11_ABI = "cxx11-abi"
RELEASE = "release"
DEBUG = "debug"

CURRENT_STABLE_VERSION = "1.13.0"

# By default use Nightly for CUDA arches
mod.CUDA_ARCHES = CUDA_ACRHES_DICT["nightly"]

LINUX_GPU_RUNNER = "ubuntu-20.04-m60"
LINUX_CPU_RUNNER = "ubuntu-20.04"
WIN_GPU_RUNNER = "windows.8xlarge.nvidia.gpu"
WIN_CPU_RUNNER = "windows.4xlarge"
MACOS_M1_RUNNER = "macos-m1-12"
MACOS_RUNNER = "macos-12"

PACKAGES_TO_INSTALL_WHL = "torch torchvision torchaudio"
PACKAGES_TO_INSTALL_WHL_TORCHONLY = "torch"

PACKAGES_TO_INSTALL_CONDA = "pytorch torchvision torchaudio"
CONDA_INSTALL_BASE = f"conda install {PACKAGES_TO_INSTALL_CONDA}"
WHL_INSTALL_BASE = "pip3 install"
DOWNLOAD_URL_BASE = "https://download.pytorch.org"

ENABLE = "enable"
DISABLE = "disable"

def arch_type(arch_version: str) -> str:
    if arch_version in mod.CUDA_ARCHES:
        return "cuda"
    elif arch_version in ROCM_ARCHES:
        return "rocm"
    else:  # arch_version should always be "cpu" in this case
        return "cpu"

def validation_runner(arch_type: str, os: str) -> str:
    if os == "linux":
        if arch_type == "cuda":
            return LINUX_GPU_RUNNER
        else:
            return LINUX_CPU_RUNNER
    elif os == "windows":
        if arch_type == "cuda":
            return WIN_GPU_RUNNER
        else:
            return WIN_CPU_RUNNER
    elif os == "macos-arm64":
        return MACOS_M1_RUNNER
    elif os == "macos":
        return MACOS_RUNNER
    else: # default to linux cpu runner
        return LINUX_CPU_RUNNER

def initialize_globals(channel: str):
    mod.CUDA_ARCHES = CUDA_ACRHES_DICT[channel]
    mod.WHEEL_CONTAINER_IMAGES = {
        **{
            gpu_arch: f"pytorch/manylinux-builder:cuda{gpu_arch}"
            for gpu_arch in mod.CUDA_ARCHES
        },
        **{
            gpu_arch: f"pytorch/manylinux-builder:rocm{gpu_arch}"
            for gpu_arch in ROCM_ARCHES
        },
        "cpu": "pytorch/manylinux-builder:cpu",
    }
    mod.CONDA_CONTAINER_IMAGES = {
        **{gpu_arch: f"pytorch/conda-builder:cuda{gpu_arch}" for gpu_arch in mod.CUDA_ARCHES},
        "cpu": "pytorch/conda-builder:cpu",
    }
    mod.LIBTORCH_CONTAINER_IMAGES: Dict[Tuple[str, str], str] = {
        **{
            (gpu_arch, PRE_CXX11_ABI): f"pytorch/manylinux-builder:cuda{gpu_arch}"
            for gpu_arch in mod.CUDA_ARCHES
        },
        **{
            (gpu_arch, CXX11_ABI): f"pytorch/libtorch-cxx11-builder:cuda{gpu_arch}"
            for gpu_arch in mod.CUDA_ARCHES
        },
        **{
            (gpu_arch, PRE_CXX11_ABI): f"pytorch/manylinux-builder:rocm{gpu_arch}"
            for gpu_arch in ROCM_ARCHES
        },
        **{
            (gpu_arch, CXX11_ABI): f"pytorch/libtorch-cxx11-builder:rocm{gpu_arch}"
            for gpu_arch in ROCM_ARCHES
        },
        ("cpu", PRE_CXX11_ABI): "pytorch/manylinux-builder:cpu",
        ("cpu", CXX11_ABI): "pytorch/libtorch-cxx11-builder:cpu",
    }


def translate_desired_cuda(gpu_arch_type: str, gpu_arch_version: str) -> str:
    return {
        "cpu": "cpu",
        "cuda": f"cu{gpu_arch_version.replace('.', '')}",
        "rocm": f"rocm{gpu_arch_version}",
    }.get(gpu_arch_type, gpu_arch_version)


def list_without(in_list: List[str], without: List[str]) -> List[str]:
    return [item for item in in_list if item not in without]

def get_conda_install_command(channel: str, gpu_arch_type: str, arch_version: str) -> str:
    conda_channels = "-c pytorch" if channel == "release" else f"-c pytorch-{channel}"

    if gpu_arch_type == "cuda":
        conda_package_type = f"pytorch-cuda={arch_version}"
        conda_channels = f"{conda_channels} -c nvidia"
    else:
        conda_package_type = "cpuonly"

    return f"{CONDA_INSTALL_BASE} {conda_package_type} {conda_channels}"

def get_base_download_url_for_repo(repo: str, channel: str, gpu_arch_type: str, desired_cuda: str) -> str:
    base_url_for_type = f"{DOWNLOAD_URL_BASE}/{repo}"
    base_url_for_type = base_url_for_type if channel == "release" else f"{base_url_for_type}/{channel}"

    if gpu_arch_type != "cpu":
        base_url_for_type= f"{base_url_for_type}/{desired_cuda}"
    else:
        base_url_for_type= f"{base_url_for_type}/{gpu_arch_type}"

    return base_url_for_type

def get_libtorch_install_command(os: str, channel: str, gpu_arch_type: str, libtorch_variant: str, devtoolset: str, desired_cuda: str) -> str:
    build_name = f"libtorch-{devtoolset}-{libtorch_variant}-latest.zip" if devtoolset ==  "cxx11-abi" else f"libtorch-{libtorch_variant}-latest.zip"

    if channel == 'release':
        prefix = "libtorch" if os != 'windows' else "libtorch-win"
        build_name = f"{prefix}-{devtoolset}-{libtorch_variant}-{CURRENT_STABLE_VERSION}%2B{desired_cuda}.zip" if devtoolset ==  "cxx11-abi" else f"{prefix}-{libtorch_variant}-{CURRENT_STABLE_VERSION}%2B{desired_cuda}.zip"

    return f"{get_base_download_url_for_repo('libtorch', channel, gpu_arch_type, desired_cuda)}/{build_name}"

def get_wheel_install_command(channel: str, gpu_arch_type: str, desired_cuda: str, python_version: str, with_pypi: bool = False) -> str:
    packages_to_install = PACKAGES_TO_INSTALL_WHL_TORCHONLY if python_version == "3.11" or with_pypi else PACKAGES_TO_INSTALL_WHL
    whl_install_command = f"{WHL_INSTALL_BASE} --pre {packages_to_install}" if channel == "nightly" else f"{WHL_INSTALL_BASE} {packages_to_install}"
    desired_cuda_pkg = f"{desired_cuda}_pypi_cudnn" if with_pypi else desired_cuda
    return f"{whl_install_command} --extra-index-url {get_base_download_url_for_repo('whl', channel, gpu_arch_type, desired_cuda_pkg)}"

def generate_conda_matrix(os: str, channel: str, with_cuda: str) -> List[Dict[str, str]]:
    ret: List[Dict[str, str]] = []
    arches = ["cpu"]
    python_versions = FULL_PYTHON_VERSIONS

    if with_cuda == ENABLE:
        if os == "linux":
            arches += mod.CUDA_ARCHES
        elif os == "windows":
            # We don't build CUDA 10.2 for window see https://github.com/pytorch/pytorch/issues/65648
            arches += list_without(mod.CUDA_ARCHES, ["10.2"])

    if os == "macos-arm64":
        python_versions = list_without(python_versions, ["3.7"])

    for python_version in python_versions:
        # We don't currently build conda packages for rocm
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version

            ret.append(
                {
                    "python_version": python_version,
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": translate_desired_cuda(
                        gpu_arch_type, gpu_arch_version
                    ),
                    "container_image": mod.CONDA_CONTAINER_IMAGES[arch_version],
                    "package_type": "conda",
                    "build_name": f"conda-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                        ".", "_"
                    ),
                    "validation_runner": validation_runner(gpu_arch_type, os),
                    "channel": channel,
                    "installation": get_conda_install_command(channel, gpu_arch_type, arch_version)
                }
            )
    return ret


def generate_libtorch_matrix(
    os: str,
    channel: str,
    with_cuda: str,
    abi_versions: Optional[List[str]] = None,
    arches: Optional[List[str]] = None,
    libtorch_variants: Optional[List[str]] = None,
) -> List[Dict[str, str]]:

    ret: List[Dict[str, str]] = []

    if arches is None:
        arches = ["cpu"]

        if with_cuda == ENABLE:
            if os == "linux":
                arches += mod.CUDA_ARCHES
                arches += ROCM_ARCHES
            elif os == "windows":
                # We don't build CUDA 10.2 for window see https://github.com/pytorch/pytorch/issues/65648
                arches += list_without(mod.CUDA_ARCHES, ["10.2"])

    if abi_versions is None:
        if os == "windows":
            abi_versions = [RELEASE, DEBUG]
        elif os == "linux":
            abi_versions = [PRE_CXX11_ABI, CXX11_ABI]
        elif os == "macos":
            abi_versions = [PRE_CXX11_ABI, CXX11_ABI]

    if libtorch_variants is None:
        libtorch_variants = [
            "shared-with-deps",
            "shared-without-deps",
            "static-with-deps",
            "static-without-deps",
        ]

    for abi_version in abi_versions:
        for arch_version in arches:
            for libtorch_variant in libtorch_variants:
                # one of the values in the following list must be exactly
                # CXX11_ABI, but the precise value of the other one doesn't
                # matter
                gpu_arch_type = arch_type(arch_version)
                gpu_arch_version = "" if arch_version == "cpu" else arch_version
                # ROCm builds without-deps failed even in ROCm runners; skip for now
                if gpu_arch_type == "rocm" and "without-deps" in libtorch_variant:
                    continue

                # For windows release we support only shared-with-deps variant
                # see: https://github.com/pytorch/pytorch/issues/87782
                if os == 'windows' and channel == 'release' and libtorch_variant != "shared-with-deps":
                    continue

                desired_cuda = translate_desired_cuda(gpu_arch_type, gpu_arch_version)
                devtoolset = abi_version if os != "windows" else ""
                ret.append(
                    {
                        "gpu_arch_type": gpu_arch_type,
                        "gpu_arch_version": gpu_arch_version,
                        "desired_cuda": desired_cuda,
                        "libtorch_variant": libtorch_variant,
                        "libtorch_config": abi_version if os == "windows" else "",
                        "devtoolset": devtoolset,
                        "container_image": mod.LIBTORCH_CONTAINER_IMAGES[
                            (arch_version, abi_version)
                        ]
                        if os != "windows"
                        else "",
                        "package_type": "libtorch",
                        "build_name": f"libtorch-{gpu_arch_type}{gpu_arch_version}-{libtorch_variant}-{abi_version}".replace(
                            ".", "_"
                        ),
                        "validation_runner": validation_runner(gpu_arch_type, os),
                        "installation": get_libtorch_install_command(os, channel, gpu_arch_type, libtorch_variant, devtoolset, desired_cuda),
                        "channel": channel
                    }
                )
    return ret


def generate_wheels_matrix(
    os: str,
    channel: str,
    with_cuda: str,
    with_py311: str,
    with_pypi_cudnn: str = DISABLE,
    arches: Optional[List[str]] = None,
    python_versions: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    package_type = "wheel"

    if python_versions is None:
        # Define default python version
        python_versions = list(FULL_PYTHON_VERSIONS)
        if os == "macos-arm64":
            python_versions = list_without(python_versions, ["3.7"])

    if os == "linux":
        # NOTE: We only build manywheel packages for linux
        package_type = "manywheel"
        if with_py311 == ENABLE and channel != "release":
            python_versions += ["3.11"]

    if arches is None:
        # Define default compute archivectures
        arches = ["cpu"]

        if with_cuda == ENABLE:
            if os == "linux":
                arches += mod.CUDA_ARCHES + ROCM_ARCHES
            elif os == "windows":
                # We don't build CUDA 10.2 for window see https://github.com/pytorch/pytorch/issues/65648
                arches += list_without(mod.CUDA_ARCHES, ["10.2"])

    ret: List[Dict[str, str]] = []
    for python_version in python_versions:
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version
            # Skip rocm 3.11 binaries for now as the docker image are not correct
            if python_version == "3.11" and gpu_arch_type == "rocm":
                continue
            desired_cuda = translate_desired_cuda(gpu_arch_type, gpu_arch_version)

            installation_pypi = ""
            # special 11.7 wheels package without dependencies
            # dependency downloaded via pip install
            if arch_version == "11.7" and os == "linux" and with_pypi_cudnn == ENABLE:
                installation_pypi = get_wheel_install_command(channel, gpu_arch_type, desired_cuda, python_version, True)
                ret.append(
                    {
                        "python_version": python_version,
                        "gpu_arch_type": gpu_arch_type,
                        "gpu_arch_version": gpu_arch_version,
                        "desired_cuda": translate_desired_cuda(
                            gpu_arch_type, gpu_arch_version
                        ),
                        "container_image": mod.WHEEL_CONTAINER_IMAGES[arch_version],
                        "package_type": package_type,
                        "pytorch_extra_install_requirements":
                        "nvidia-cuda-runtime-cu11;"
                        "nvidia-cudnn-cu11==8.5.0.96;"
                        "nvidia-cublas-cu11==11.10.3.66",
                        "installation_pypi": installation_pypi,
                        "build_name":
                        f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}-with-pypi-cudnn"
                        .replace(
                            ".", "_"
                        ),
                        "validation_runner": validation_runner(gpu_arch_type, os),
                        "channel": channel,
                    }
                )

            ret.append(
                {
                    "python_version": python_version,
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": desired_cuda,
                    "container_image": mod.WHEEL_CONTAINER_IMAGES[arch_version],
                    "package_type": package_type,
                    "build_name": f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                        ".", "_"
                    ),
                    "validation_runner": validation_runner(gpu_arch_type, os),
                    "installation": get_wheel_install_command(channel, gpu_arch_type, desired_cuda, python_version),
                    "channel": channel,
                }
            )
    return ret


GENERATING_FUNCTIONS_BY_PACKAGE_TYPE = {
    "wheel": generate_wheels_matrix,
    "conda": generate_conda_matrix,
    "libtorch": generate_libtorch_matrix,
}

def main(args) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-type",
        help="Package type to lookup for",
        type=str,
        choices=["wheel", "conda", "libtorch", "all"],
        default=os.getenv("PACKAGE_TYPE", "wheel"),
    )
    parser.add_argument(
        "--operating-system",
        help="Operating system to generate for",
        type=str,
        default=os.getenv("OS", "linux"),
    )
    parser.add_argument(
        "--channel",
        help="Channel to use, default nightly",
        type=str,
        choices=["nightly", "test", "release", "all"],
        default=os.getenv("CHANNEL", "nightly"),
    )
    parser.add_argument(
        "--with-cuda",
        help="Build with Cuda?",
        type=str,
        choices=[ENABLE, DISABLE],
        default=os.getenv("WITH_CUDA", ENABLE),
    )
    parser.add_argument(
        "--with-py311",
        help="Include Python 3.11 builds",
        type=str,
        choices=[ENABLE, DISABLE],
        default=os.getenv("WITH_PY311", DISABLE),
    )
    parser.add_argument(
        "--with-pypi-cudnn",
        help="Include PyPI cudnn builds",
        type=str,
        choices=[ENABLE, DISABLE],
        default=os.getenv("WITH_PYPI_CUDNN", DISABLE),
    )

    options = parser.parse_args(args)
    includes = []

    package_types = PACKAGE_TYPES if options.package_type == "all" else [options.package_type]
    channels = CUDA_ACRHES_DICT.keys() if options.channel == "all" else [options.channel]

    for channel in channels:
        for package in package_types:
            initialize_globals(channel)
            if package == "wheel":
                includes.extend(
                    GENERATING_FUNCTIONS_BY_PACKAGE_TYPE[package](options.operating_system,
                                                                channel,
                                                                options.with_cuda,
                                                                options.with_py311,
                                                                options.with_pypi_cudnn)
                    )
            else:
                includes.extend(
                    GENERATING_FUNCTIONS_BY_PACKAGE_TYPE[package](options.operating_system,
                                                                channel,
                                                                options.with_cuda)
                    )


    print(json.dumps({"include": includes}))

if __name__ == "__main__":
    main(sys.argv[1:])
