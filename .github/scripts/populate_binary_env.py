import os

# Environment variables that form our input, we expect these all to be present.
GPU_ARCH_VERSION = os.environ["GPU_ARCH_VERSION"]
DESIRED_PYTHON = os.environ["DESIRED_PYTHON"]
PACKAGE_TYPE = os.environ["PACKAGE_TYPE"]
GITHUB_ENV = os.environ["GITHUB_ENV"]


CUDA_ARCHES = ["10.2", "11.3", "11.6"]
ROCM_ARCHES = ["5.0", "5.1.1"]
WHEEL_CONTAINER_IMAGES = {
    **{
        gpu_arch: f"pytorch/manylinux-builder:cuda{gpu_arch}"
        for gpu_arch in CUDA_ARCHES
    },
    **{
        gpu_arch: f"pytorch/manylinux-builder:rocm{gpu_arch}"
        for gpu_arch in ROCM_ARCHES
    },
    "cpu": "pytorch/manylinux-builder:cpu",
}


def arch_type(arch_version: str) -> str:
    if arch_version in CUDA_ARCHES:
        return "cuda"
    elif arch_version in ROCM_ARCHES:
        return "rocm"
    else:  # arch_version should always be "cpu" in this case
        return "cpu"


def translate_desired_cuda(gpu_arch_type: str, gpu_arch_version: str) -> str:
    return {
        "cpu": "cpu",
        "cuda": f"cu{gpu_arch_version.replace('.', '')}",
        "rocm": f"rocm{gpu_arch_version}",
    }.get(gpu_arch_type, gpu_arch_version)


GPU_ARCH_TYPE = arch_type(GPU_ARCH_VERSION)
DESIRED_CUDA = translate_desired_cuda(GPU_ARCH_TYPE, GPU_ARCH_VERSION)
CONTAINER_IMAGE = WHEEL_CONTAINER_IMAGES[GPU_ARCH_VERSION]
BUILD_NAME = (
    f"{PACKAGE_TYPE}-py{DESIRED_PYTHON}-{GPU_ARCH_TYPE}{GPU_ARCH_VERSION}".replace(
        ".", "_"  # DERIVED
    ),
)

with open(GITHUB_ENV, "a") as f:
    f.write(f"GPU_ARCH_TYPE={GPU_ARCH_TYPE}\n")
    f.write(f"DESIRED_CUDA={DESIRED_CUDA}\n")
    f.write(f"CONTAINER_IMAGE={CONTAINER_IMAGE}\n")
    f.write(f"BUILD_NAME={BUILD_NAME}\n")
