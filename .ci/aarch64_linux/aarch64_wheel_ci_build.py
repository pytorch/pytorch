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
                "v52.1.0",
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


def patch_library_rpath(
    folder: str,
    lib_name: str,
    use_nvidia_pypi_libs: bool = False,
    desired_cuda: str = "",
) -> None:
    """Apply patchelf to set RPATH for a library in torch/lib"""
    lib_path = f"{folder}/tmp/torch/lib/{lib_name}"

    if use_nvidia_pypi_libs:
        # For PyPI NVIDIA libraries, construct CUDA RPATH
        cuda_rpaths = [
            "$ORIGIN/../../nvidia/cudnn/lib",
            "$ORIGIN/../../nvidia/nvshmem/lib",
            "$ORIGIN/../../nvidia/nccl/lib",
            "$ORIGIN/../../nvidia/cusparselt/lib",
        ]

        if "130" in desired_cuda:
            cuda_rpaths.append("$ORIGIN/../../nvidia/cu13/lib")
        else:
            cuda_rpaths.extend(
                [
                    "$ORIGIN/../../nvidia/cublas/lib",
                    "$ORIGIN/../../nvidia/cuda_cupti/lib",
                    "$ORIGIN/../../nvidia/cuda_nvrtc/lib",
                    "$ORIGIN/../../nvidia/cuda_runtime/lib",
                    "$ORIGIN/../../nvidia/cufft/lib",
                    "$ORIGIN/../../nvidia/curand/lib",
                    "$ORIGIN/../../nvidia/cusolver/lib",
                    "$ORIGIN/../../nvidia/cusparse/lib",
                    "$ORIGIN/../../nvidia/nvtx/lib",
                    "$ORIGIN/../../nvidia/cufile/lib",
                ]
            )

        # Add $ORIGIN for local torch libs
        rpath = ":".join(cuda_rpaths) + ":$ORIGIN"
    else:
        # For bundled libraries, just use $ORIGIN
        rpath = "$ORIGIN"

    if os.path.exists(lib_path):
        os.system(
            f"cd {folder}/tmp/torch/lib/; "
            f"patchelf --set-rpath '{rpath}' --force-rpath {lib_name}"
        )


def copy_and_patch_library(
    src_path: str,
    folder: str,
    use_nvidia_pypi_libs: bool = False,
    desired_cuda: str = "",
) -> None:
    """Copy a library to torch/lib and patch its RPATH"""
    if os.path.exists(src_path):
        lib_name = os.path.basename(src_path)
        shutil.copy2(src_path, f"{folder}/tmp/torch/lib/{lib_name}")
        patch_library_rpath(folder, lib_name, use_nvidia_pypi_libs, desired_cuda)


def package_cuda_wheel(wheel_path, desired_cuda) -> None:
    """
    Package the cuda wheel libraries
    """
    folder = os.path.dirname(wheel_path)
    os.mkdir(f"{folder}/tmp")
    os.system(f"unzip {wheel_path} -d {folder}/tmp")
    # Delete original wheel since it will be repackaged
    os.system(f"rm {wheel_path}")

    # Check if we should use PyPI NVIDIA libraries or bundle system libraries
    use_nvidia_pypi_libs = os.getenv("USE_NVIDIA_PYPI_LIBS", "0") == "1"

    if use_nvidia_pypi_libs:
        print("Using nvidia libs from pypi - skipping CUDA library bundling")
        # For PyPI approach, we don't bundle CUDA libraries - they come from PyPI packages
        # We only need to bundle non-NVIDIA libraries
        minimal_libs_to_copy = [
            "/lib64/libgomp.so.1",
            "/usr/lib64/libgfortran.so.5",
            "/acl/build/libarm_compute.so",
            "/acl/build/libarm_compute_graph.so",
            "/usr/local/lib/libnvpl_lapack_lp64_gomp.so.0",
            "/usr/local/lib/libnvpl_blas_lp64_gomp.so.0",
            "/usr/local/lib/libnvpl_lapack_core.so.0",
            "/usr/local/lib/libnvpl_blas_core.so.0",
        ]

        # Copy minimal libraries to unzipped_folder/torch/lib
        for lib_path in minimal_libs_to_copy:
            copy_and_patch_library(lib_path, folder, use_nvidia_pypi_libs, desired_cuda)

        # Patch torch libraries used for searching libraries
        torch_libs_to_patch = [
            "libtorch.so",
            "libtorch_cpu.so",
            "libtorch_cuda.so",
            "libtorch_cuda_linalg.so",
            "libtorch_global_deps.so",
            "libtorch_python.so",
            "libtorch_nvshmem.so",
            "libc10.so",
            "libc10_cuda.so",
            "libcaffe2_nvrtc.so",
            "libshm.so",
        ]
        for lib_name in torch_libs_to_patch:
            patch_library_rpath(folder, lib_name, use_nvidia_pypi_libs, desired_cuda)
    else:
        print("Bundling CUDA libraries with wheel")
        # Original logic for bundling system CUDA libraries
        # Common libraries for all CUDA versions
        common_libs = [
            # Non-NVIDIA system libraries
            "/lib64/libgomp.so.1",
            "/usr/lib64/libgfortran.so.5",
            "/acl/build/libarm_compute.so",
            "/acl/build/libarm_compute_graph.so",
            # Common CUDA libraries (same for all versions)
            "/usr/local/lib/libnvpl_lapack_lp64_gomp.so.0",
            "/usr/local/lib/libnvpl_blas_lp64_gomp.so.0",
            "/usr/local/lib/libnvpl_lapack_core.so.0",
            "/usr/local/lib/libnvpl_blas_core.so.0",
            "/usr/local/cuda/extras/CUPTI/lib64/libnvperf_host.so",
            "/usr/local/cuda/lib64/libcudnn.so.9",
            "/usr/local/cuda/lib64/libcusparseLt.so.0",
            "/usr/local/cuda/lib64/libcurand.so.10",
            "/usr/local/cuda/lib64/libnccl.so.2",
            "/usr/local/cuda/lib64/libnvshmem_host.so.3",
            "/usr/local/cuda/lib64/libcudnn_adv.so.9",
            "/usr/local/cuda/lib64/libcudnn_cnn.so.9",
            "/usr/local/cuda/lib64/libcudnn_graph.so.9",
            "/usr/local/cuda/lib64/libcudnn_ops.so.9",
            "/usr/local/cuda/lib64/libcudnn_engines_runtime_compiled.so.9",
            "/usr/local/cuda/lib64/libcudnn_engines_precompiled.so.9",
            "/usr/local/cuda/lib64/libcudnn_heuristic.so.9",
            "/usr/local/cuda/lib64/libcufile.so.0",
            "/usr/local/cuda/lib64/libcufile_rdma.so.1",
            "/usr/local/cuda/lib64/libcusparse.so.12",
        ]

        # CUDA version-specific libraries
        if "13" in desired_cuda:
            minor_version = desired_cuda[-1]
            version_specific_libs = [
                "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.13",
                "/usr/local/cuda/lib64/libcublas.so.13",
                "/usr/local/cuda/lib64/libcublasLt.so.13",
                "/usr/local/cuda/lib64/libcudart.so.13",
                "/usr/local/cuda/lib64/libcufft.so.12",
                "/usr/local/cuda/lib64/libcusolver.so.12",
                "/usr/local/cuda/lib64/libnvJitLink.so.13",
                "/usr/local/cuda/lib64/libnvrtc.so.13",
                f"/usr/local/cuda/lib64/libnvrtc-builtins.so.13.{minor_version}",
            ]
        elif "12" in desired_cuda:
            # Get the last character for libnvrtc-builtins version (e.g., "129" -> "9")
            minor_version = desired_cuda[-1]
            version_specific_libs = [
                "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12",
                "/usr/local/cuda/lib64/libcublas.so.12",
                "/usr/local/cuda/lib64/libcublasLt.so.12",
                "/usr/local/cuda/lib64/libcudart.so.12",
                "/usr/local/cuda/lib64/libcufft.so.11",
                "/usr/local/cuda/lib64/libcusolver.so.11",
                "/usr/local/cuda/lib64/libnvJitLink.so.12",
                "/usr/local/cuda/lib64/libnvrtc.so.12",
                f"/usr/local/cuda/lib64/libnvrtc-builtins.so.12.{minor_version}",
            ]
        else:
            raise ValueError(f"Unsupported CUDA version: {desired_cuda}.")

        # Combine all libraries
        libs_to_copy = common_libs + version_specific_libs

        # Copy libraries to unzipped_folder/torch/lib
        for lib_path in libs_to_copy:
            copy_and_patch_library(lib_path, folder, use_nvidia_pypi_libs, desired_cuda)

    # Make sure the wheel is tagged with manylinux_2_28
    for f in os.scandir(f"{folder}/tmp/"):
        if f.is_dir() and f.name.endswith(".dist-info"):
            replace_tag(f"{f.path}/WHEEL")
            break

    os.system(f"wheel pack {folder}/tmp/ -d {folder}")
    os.system(f"rm -rf {folder}/tmp/")


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
        repaired_wheel_name = list_dir(f"/{folder}/dist")[0]

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
        build_vars += "MAX_JOBS=5 "

        # Handle PyPI NVIDIA libraries vs bundled libraries
        use_nvidia_pypi_libs = os.getenv("USE_NVIDIA_PYPI_LIBS", "0") == "1"
        if use_nvidia_pypi_libs:
            print("Configuring build for PyPI NVIDIA libraries")
            # Configure for dynamic linking (matching x86 logic)
            build_vars += "ATEN_STATIC_CUDA=0 USE_CUDA_STATIC_LINK=0 USE_CUPTI_SO=1 "
        else:
            print("Configuring build for bundled NVIDIA libraries")
            # Keep existing static linking approach - already configured above

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
