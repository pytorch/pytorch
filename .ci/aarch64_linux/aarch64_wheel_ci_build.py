#!/usr/bin/env python3
# encoding: UTF-8

import os
import shutil
from subprocess import check_call, check_output
from typing import List

from pygit2 import Repository


def list_dir(path: str) -> List[str]:
    """'
    Helper for getting paths for Python
    """
    return check_output(["ls", "-1", path]).decode().split("\n")


def build_ArmComputeLibrary() -> None:
    """
    Using ArmComputeLibrary for aarch64 PyTorch
    """
    print("Building Arm Compute Library")
    acl_build_flags = [
        "debug=0",
        "neon=1",
        "opencl=0",
        "os=linux",
        "openmp=1",
        "cppthreads=0",
        "arch=armv8a",
        "multi_isa=1",
        "fixed_format_kernels=1",
        "build=native",
    ]
    acl_install_dir = "/acl"
    acl_checkout_dir = "ComputeLibrary"
    os.makedirs(acl_install_dir)
    check_call(
        [
            "git",
            "clone",
            "https://github.com/ARM-software/ComputeLibrary.git",
            "-b",
            "v24.09",
            "--depth",
            "1",
            "--shallow-submodules",
        ]
    )

    check_call(
        ["scons", "Werror=1", "-j8", f"build_dir=/{acl_install_dir}/build"]
        + acl_build_flags,
        cwd=acl_checkout_dir,
    )
    for d in ["arm_compute", "include", "utils", "support", "src"]:
        shutil.copytree(f"{acl_checkout_dir}/{d}", f"{acl_install_dir}/{d}")


def update_wheel(wheel_path) -> None:
    """
    Update the cuda wheel libraries
    """
    folder = os.path.dirname(wheel_path)
    wheelname = os.path.basename(wheel_path)
    os.mkdir(f"{folder}/tmp")
    os.system(f"unzip {wheel_path} -d {folder}/tmp")
    libs_to_copy = [
        "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12",
        "/usr/local/cuda/lib64/libcudnn.so.9",
        "/usr/local/cuda/lib64/libcublas.so.12",
        "/usr/local/cuda/lib64/libcublasLt.so.12",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/local/cuda/lib64/libcufft.so.11",
        "/usr/local/cuda/lib64/libcusparse.so.12",
        "/usr/local/cuda/lib64/libcusparseLt.so.0",
        "/usr/local/cuda/lib64/libcusolver.so.11",
        "/usr/local/cuda/lib64/libcurand.so.10",
        "/usr/local/cuda/lib64/libnvToolsExt.so.1",
        "/usr/local/cuda/lib64/libnvJitLink.so.12",
        "/usr/local/cuda/lib64/libnvrtc.so.12",
        "/usr/local/cuda/lib64/libnvrtc-builtins.so.12.6",
        "/usr/local/cuda/lib64/libcudnn_adv.so.9",
        "/usr/local/cuda/lib64/libcudnn_cnn.so.9",
        "/usr/local/cuda/lib64/libcudnn_graph.so.9",
        "/usr/local/cuda/lib64/libcudnn_ops.so.9",
        "/usr/local/cuda/lib64/libcudnn_engines_runtime_compiled.so.9",
        "/usr/local/cuda/lib64/libcudnn_engines_precompiled.so.9",
        "/usr/local/cuda/lib64/libcudnn_heuristic.so.9",
        "/lib64/libgomp.so.1",
        "/usr/lib64/libgfortran.so.5",
        "/acl/build/libarm_compute.so",
        "/acl/build/libarm_compute_graph.so",
    ]
    if enable_cuda:
        libs_to_copy += [
            "/usr/local/lib/libnvpl_lapack_lp64_gomp.so.0",
            "/usr/local/lib/libnvpl_blas_lp64_gomp.so.0",
            "/usr/local/lib/libnvpl_lapack_core.so.0",
            "/usr/local/lib/libnvpl_blas_core.so.0",
        ]
    else:
        libs_to_copy += [
            "/opt/OpenBLAS/lib/libopenblas.so.0",
        ]
    # Copy libraries to unzipped_folder/a/lib
    for lib_path in libs_to_copy:
        lib_name = os.path.basename(lib_path)
        shutil.copy2(lib_path, f"{folder}/tmp/torch/lib/{lib_name}")
        os.system(
            f"cd {folder}/tmp/torch/lib/; "
            f"patchelf --set-rpath '$ORIGIN' --force-rpath {folder}/tmp/torch/lib/{lib_name}"
        )
    os.mkdir(f"{folder}/cuda_wheel")
    os.system(f"cd {folder}/tmp/; zip -r {folder}/cuda_wheel/{wheelname} *")
    shutil.move(
        f"{folder}/cuda_wheel/{wheelname}",
        f"{folder}/{wheelname}",
        copy_function=shutil.copy2,
    )
    os.system(f"rm -rf {folder}/tmp/ {folder}/cuda_wheel/")


def complete_wheel(folder: str) -> str:
    """
    Complete wheel build and put in artifact location
    """
    wheel_name = list_dir(f"/{folder}/dist")[0]

    if "pytorch" in folder and not enable_cuda:
        print("Repairing Wheel with AuditWheel")
        check_call(["auditwheel", "repair", f"dist/{wheel_name}"], cwd=folder)
        repaired_wheel_name = list_dir(f"/{folder}/wheelhouse")[0]

        print(f"Moving {repaired_wheel_name} wheel to /{folder}/dist")
        os.rename(
            f"/{folder}/wheelhouse/{repaired_wheel_name}",
            f"/{folder}/dist/{repaired_wheel_name}",
        )
    else:
        repaired_wheel_name = wheel_name

    print(f"Copying {repaired_wheel_name} to artifacts")
    shutil.copy2(
        f"/{folder}/dist/{repaired_wheel_name}", f"/artifacts/{repaired_wheel_name}"
    )

    return repaired_wheel_name


def parse_arguments():
    """
    Parse inline arguments
    """
    from argparse import ArgumentParser

    parser = ArgumentParser("AARCH64 wheels python CD")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--test-only", type=str)
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--enable-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Entry Point
    """
    args = parse_arguments()
    enable_mkldnn = args.enable_mkldnn
    enable_cuda = args.enable_cuda
    repo = Repository("/pytorch")
    branch = repo.head.name
    if branch == "HEAD":
        branch = "master"

    print("Building PyTorch wheel")
    build_vars = "MAX_JOBS=5 CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 TORCH_CUDA_ARCH_LIST='9.0'"
    os.system("cd /pytorch; python setup.py clean")

    override_package_version = os.getenv("OVERRIDE_PACKAGE_VERSION")
    if override_package_version is not None:
        version = override_package_version
        build_vars += (
            f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version} PYTORCH_BUILD_NUMBER=1 "
        )
    elif branch in ["nightly", "master"]:
        build_date = (
            check_output(["git", "log", "--pretty=format:%cs", "-1"], cwd="/pytorch")
            .decode()
            .replace("-", "")
        )
        version = (
            check_output(["cat", "version.txt"], cwd="/pytorch").decode().strip()[:-2]
        )
        if enable_cuda:
            desired_cuda = os.getenv("DESIRED_CUDA")
            build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version}.dev{build_date}+{desired_cuda} PYTORCH_BUILD_NUMBER=1 "
        else:
            build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version}.dev{build_date} PYTORCH_BUILD_NUMBER=1 "
    elif branch.startswith(("v1.", "v2.")):
        build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={branch[1:branch.find('-')]} PYTORCH_BUILD_NUMBER=1 "

    if enable_mkldnn:
        build_ArmComputeLibrary()
        print("build pytorch with mkldnn+acl backend")
        build_vars += (
            "USE_MKLDNN=ON USE_MKLDNN_ACL=ON "
            "ACL_ROOT_DIR=/acl "
            "LD_LIBRARY_PATH=/pytorch/build/lib:/acl/build:$LD_LIBRARY_PATH "
            "ACL_INCLUDE_DIR=/acl/build "
            "ACL_LIBRARY=/acl/build "
        )
        if enable_cuda:
            build_vars += "BLAS=NVPL "
        else:
            build_vars += "BLAS=OpenBLAS OpenBLAS_HOME=/OpenBLAS "
    else:
        print("build pytorch without mkldnn backend")

    os.system(f"cd /pytorch; {build_vars} python3 setup.py bdist_wheel")
    if enable_cuda:
        print("Updating Cuda Dependency")
        filename = os.listdir("/pytorch/dist/")
        wheel_path = f"/pytorch/dist/{filename[0]}"
        update_wheel(wheel_path)
    pytorch_wheel_name = complete_wheel("/pytorch/")
    print(f"Build Complete. Created {pytorch_wheel_name}..")
