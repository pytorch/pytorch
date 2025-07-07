#!/usr/bin/env python3
# encoding: UTF-8

import os
import shutil
from subprocess import check_call, check_output


def list_dir(path: str) -> list[str]:
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
    acl_checkout_dir = os.getenv("ACL_SOURCE_DIR", "ComputeLibrary")
    if os.path.isdir(acl_install_dir):
        shutil.rmtree(acl_install_dir)
    if not os.path.isdir(acl_checkout_dir) or not len(os.listdir(acl_checkout_dir)):
        check_call(
            [
                "git",
                "clone",
                "https://github.com/ARM-software/ComputeLibrary.git",
                "-b",
                "v25.02",
                "--depth",
                "1",
                "--shallow-submodules",
            ]
        )

    check_call(
        ["scons", "Werror=1", f"-j{os.cpu_count()}"] + acl_build_flags,
        cwd=acl_checkout_dir,
    )
    for d in ["arm_compute", "include", "utils", "support", "src", "build"]:
        shutil.copytree(f"{acl_checkout_dir}/{d}", f"{acl_install_dir}/{d}")


def replace_tag(filename) -> None:
    with open(filename) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("Tag:"):
            lines[i] = line.replace("-linux_", "-manylinux_2_28_")
            print(f"Updated tag from {line} to {lines[i]}")
            break

    with open(filename, "w") as f:
        f.writelines(lines)


def package_cuda_wheel(wheel_path, desired_cuda) -> None:
    """
    Package the cuda wheel libraries
    """
    folder = os.path.dirname(wheel_path)
    wheelname = os.path.basename(wheel_path)
    os.mkdir(f"{folder}/tmp")
    os.system(f"unzip {wheel_path} -d {folder}/tmp")
    libs_to_copy = [
        "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12",
        "/usr/local/cuda/extras/CUPTI/lib64/libnvperf_host.so",
        "/usr/local/cuda/lib64/libcudnn.so.9",
        "/usr/local/cuda/lib64/libcublas.so.12",
        "/usr/local/cuda/lib64/libcublasLt.so.12",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/local/cuda/lib64/libcufft.so.11",
        "/usr/local/cuda/lib64/libcusparse.so.12",
        "/usr/local/cuda/lib64/libcusparseLt.so.0",
        "/usr/local/cuda/lib64/libcusolver.so.11",
        "/usr/local/cuda/lib64/libcurand.so.10",
        "/usr/local/cuda/lib64/libnccl.so.2",
        "/usr/local/cuda/lib64/libnvJitLink.so.12",
        "/usr/local/cuda/lib64/libnvrtc.so.12",
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
        "/usr/local/lib/libnvpl_lapack_lp64_gomp.so.0",
        "/usr/local/lib/libnvpl_blas_lp64_gomp.so.0",
        "/usr/local/lib/libnvpl_lapack_core.so.0",
        "/usr/local/lib/libnvpl_blas_core.so.0",
    ]

    if "129" in desired_cuda:
        libs_to_copy += [
            "/usr/local/cuda/lib64/libnvrtc-builtins.so.12.9",
            "/usr/local/cuda/lib64/libcufile.so.0",
            "/usr/local/cuda/lib64/libcufile_rdma.so.1",
        ]

    # Copy libraries to unzipped_folder/a/lib
    for lib_path in libs_to_copy:
        lib_name = os.path.basename(lib_path)
        shutil.copy2(lib_path, f"{folder}/tmp/torch/lib/{lib_name}")
        os.system(
            f"cd {folder}/tmp/torch/lib/; "
            f"patchelf --set-rpath '$ORIGIN' --force-rpath {folder}/tmp/torch/lib/{lib_name}"
        )

    # Make sure the wheel is tagged with manylinux_2_28
    for f in os.scandir(f"{folder}/tmp/"):
        if f.is_dir() and f.name.endswith(".dist-info"):
            replace_tag(f"{f.path}/WHEEL")
            break

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

    # Please note for cuda we don't run auditwheel since we use custom script to package
    # the cuda dependencies to the wheel file using update_wheel() method.
    # However we need to make sure filename reflects the correct Manylinux platform.
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
        repaired_wheel_name = wheel_name.replace(
            "linux_aarch64", "manylinux_2_28_aarch64"
        )
        print(f"Renaming {wheel_name} wheel to {repaired_wheel_name}")
        os.rename(
            f"/{folder}/dist/{wheel_name}",
            f"/{folder}/dist/{repaired_wheel_name}",
        )

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
    branch = check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd="/pytorch"
    ).decode()

    print("Building PyTorch wheel")
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    # MAX_JOB=5 is not required for CPU backend (see commit 465d98b)
    if enable_cuda:
        build_vars = "MAX_JOBS=5 " + build_vars

    override_package_version = os.getenv("OVERRIDE_PACKAGE_VERSION")
    desired_cuda = os.getenv("DESIRED_CUDA")
    if override_package_version is not None:
        version = override_package_version
        build_vars += (
            f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version} PYTORCH_BUILD_NUMBER=1 "
        )
    elif branch in ["nightly", "main"]:
        build_date = (
            check_output(["git", "log", "--pretty=format:%cs", "-1"], cwd="/pytorch")
            .decode()
            .replace("-", "")
        )
        version = (
            check_output(["cat", "version.txt"], cwd="/pytorch").decode().strip()[:-2]
        )
        if enable_cuda:
            build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version}.dev{build_date}+{desired_cuda} PYTORCH_BUILD_NUMBER=1 "
        else:
            build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version}.dev{build_date} PYTORCH_BUILD_NUMBER=1 "
    elif branch.startswith(("v1.", "v2.")):
        build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={branch[1 : branch.find('-')]} PYTORCH_BUILD_NUMBER=1 "

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
        package_cuda_wheel(wheel_path, desired_cuda)
    pytorch_wheel_name = complete_wheel("/pytorch/")
    print(f"Build Complete. Created {pytorch_wheel_name}..")
