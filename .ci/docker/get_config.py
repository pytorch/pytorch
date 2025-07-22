import argparse
import sys
from enum import Enum
import shlex

class HardwareType(Enum):
    DEFAULT = "default"
    ROCM = "rocm"

    @staticmethod
    def from_image_name(image_name: str) -> "HardwareType":
        if "rocm" in image_name:
            return HardwareType.ROCM
        return HardwareType.DEFAULT

class HardcodedBaseConfig:
    _UCX_UCC_CONFIGS: dict[HardwareType, dict[str, str]] = {
        HardwareType.DEFAULT: {
            "UCX_COMMIT": "7bb2722ff2187a0cad557ae4a6afa090569f83fb",
            "UCC_COMMIT": "20eae37090a4ce1b32bcce6144ccad0b49943e0b",
        },
        HardwareType.ROCM: {
            "UCX_COMMIT": "cc312eaa4655c0cc5c2bcd796db938f90563bcf6",
            "UCC_COMMIT": "0c0fc21559835044ab107199e334f7157d6a0d3d",
        },
    }

    def __init__(self, hardwareType: HardwareType) -> None:
        commits = self.get_ucx_ucc_commits(hardwareType)
        self.ucx_commit = commits["UCX_COMMIT"]
        self.ucc_commit = commits["UCC_COMMIT"]

    def _get_tag(self, image: str):
        if ":" not in image:
            print(f"echo 'Invalid image format (missing :): {image}'", file=sys.stderr)
            return
        tag = image.split(":")[1]
        return tag

    def get_all_configs(self):
        _TAG_CONFIGS = {
            "pytorch-linux-jammy-cuda12.4-cudnn9-py3-gcc11": {
                "CUDA_VERSION": "12.4",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11": {
                "CUDA_VERSION": "12.8.1",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc9-inductor-benchmarks": {
                "CUDA_VERSION": "12.8.1",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-jammy-cuda12.8-cudnn9-py3.12-gcc9-inductor-benchmarks": {
                "CUDA_VERSION": "12.8.1",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.12",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-jammy-cuda12.8-cudnn9-py3.13-gcc9-inductor-benchmarks": {
                "CUDA_VERSION": "12.8.1",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.13",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-jammy-cuda12.6-cudnn9-py3-gcc9": {
                "CUDA_VERSION": "12.6.3",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-cuda12.6-cudnn9-py3-gcc9-inductor-benchmarks": {
                "CUDA_VERSION": "12.6",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-jammy-cuda12.6-cudnn9-py3.12-gcc9-inductor-benchmarks": {
                "CUDA_VERSION": "12.6",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.12",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-jammy-cuda12.6-cudnn9-py3.13-gcc9-inductor-benchmarks": {
                "CUDA_VERSION": "12.6",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.13",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc9": {
                "CUDA_VERSION": "12.8.1",
                "CUDNN_VERSION": "9",
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-py3-clang12-onnx": {
                "ANACONDA_PYTHON_VERSION": "3.9",
                "CLANG_VERSION": "12",
                "VISION": "yes",
                "ONNX": "yes",
            },
            "pytorch-linux-jammy-py3.9-clang12": {
                "ANACONDA_PYTHON_VERSION": "3.9",
                "CLANG_VERSION": "12",
                "VISION": "yes",
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-py3.11-clang12": {
                "ANACONDA_PYTHON_VERSION": "3.11",
                "CLANG_VERSION": "12",
                "VISION": "yes",
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-py3.9-gcc9": {
                "ANACONDA_PYTHON_VERSION": "3.9",
                "GCC_VERSION": "9",
                "VISION": "yes",
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-rocm-n-py3": {
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "ROCM_VERSION": "6.4",
                "NINJA_VERSION": "1.9.0",
                "TRITON": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-noble-rocm-n-py3": {
                "ANACONDA_PYTHON_VERSION": "3.12",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "ROCM_VERSION": "6.4",
                "NINJA_VERSION": "1.9.0",
                "TRITON": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-noble-rocm-alpha-py3": {
                "ANACONDA_PYTHON_VERSION": "3.12",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "ROCM_VERSION": "7.0",
                "NINJA_VERSION": "1.9.0",
                "TRITON": "yes",
                "KATEX": "yes",
                "UCX_COMMIT": self.ucx_commit,
                "UCC_COMMIT": self.ucc_commit,
                "INDUCTOR_BENCHMARKS": "yes",
                "PYTORCH_ROCM_ARCH": "gfx90a;gfx942;gfx950",
            },
            "pytorch-linux-jammy-xpu-2025.0-py3": {
                "ANACONDA_PYTHON_VERSION": "3.9",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "XPU_VERSION": "2025.0",
                "NINJA_VERSION": "1.9.0",
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-xpu-2025.1-py3": {
                "ANACONDA_PYTHON_VERSION": "3.9",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "XPU_VERSION": "2025.1",
                "NINJA_VERSION": "1.9.0",
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-py3.9-gcc11-inductor-benchmarks": {
                "ANACONDA_PYTHON_VERSION": "3.9",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "KATEX": "yes",
                "TRITON": "yes",
                "DOCS": "yes",
                "INDUCTOR_BENCHMARKS": "yes",
            },
            "pytorch-linux-jammy-cuda12.8-cudnn9-py3.9-clang12": {
                "ANACONDA_PYTHON_VERSION": "3.9",
                "CUDA_VERSION": "12.8.1",
                "CUDNN_VERSION": "9",
                "CLANG_VERSION": "12",
                "VISION": "yes",
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-py3-clang18-asan": {
                "ANACONDA_PYTHON_VERSION": "3.10",
                "CLANG_VERSION": "18",
                "VISION": "yes",
            },
            "pytorch-linux-jammy-py3.9-gcc11": {
                "ANACONDA_PYTHON_VERSION": "3.9",
                "GCC_VERSION": "11",
                "VISION": "yes",
                "KATEX": "yes",
                "TRITON": "yes",
                "DOCS": "yes",
                "UNINSTALL_DILL": "yes",
            },
            "pytorch-linux-jammy-py3-clang12-executorch": {
                "ANACONDA_PYTHON_VERSION": "3.10",
                "CLANG_VERSION": "12",
                "EXECUTORCH": "yes",
            },
            "pytorch-linux-jammy-py3.12-halide": {
                "CUDA_VERSION": "12.6",
                "ANACONDA_PYTHON_VERSION": "3.12",
                "GCC_VERSION": "11",
                "HALIDE": "yes",
                "TRITON": "yes",
            },
            "pytorch-linux-jammy-py3.12-triton-cpu": {
                "CUDA_VERSION": "12.6",
                "ANACONDA_PYTHON_VERSION": "3.12",
                "GCC_VERSION": "11",
                "TRITON_CPU": "yes",
            },
            "pytorch-linux-jammy-linter": {
                "PYTHON_VERSION": "3.9",
            },
            "pytorch-linux-jammy-cuda12.8-cudnn9-py3.9-linter": {
                "PYTHON_VERSION": "3.9",
                "CUDA_VERSION": "12.8.1",
            },
            "pytorch-linux-jammy-aarch64-py3.10-gcc11": {
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "11",
                "ACL": "yes",
                "VISION": "yes",
                "CONDA_CMAKE": "yes",
                "OPENBLAS": "yes",
                "SKIP_LLVM_SRC_BUILD_INSTALL": "yes",
            },
            "pytorch-linux-jammy-aarch64-py3.10-gcc11-inductor-benchmarks": {
                "ANACONDA_PYTHON_VERSION": "3.10",
                "GCC_VERSION": "11",
                "ACL": "yes",
                "VISION": "yes",
                "CONDA_CMAKE": "yes",
                "OPENBLAS": "yes",
                "SKIP_LLVM_SRC_BUILD_INSTALL": "yes",
                "INDUCTOR_BENCHMARKS": "yes",
            },
        }
        return _TAG_CONFIGS
    def get_config(self, image_name:str) -> dict:
        tag = self._get_tag(image_name)

        config_dict = self.get_all_configs()
        if tag not in config_dict:
            raise ValueError(f"Unknown tag: {tag}")
        return config_dict[tag]

    def get_ucx_ucc_commits(self, hw_type: HardwareType) -> dict[str, str]:
        if hw_type not in self._UCX_UCC_CONFIGS:
            raise ValueError(f"Unsupported hardware type: {hw_type}")
        return self._UCX_UCC_CONFIGS[hw_type]

def main():
    parser = argparse.ArgumentParser(
        description="Return  for a given image tag."
    )
    parser.add_argument(
        "--image", required=True, help="Full image string (e.g., repo/name:tag)"
    )
    args = parser.parse_args()

    try:

        image_name = args.image
        hw_type = HardwareType.from_image_name(image_name)

        config_runner = HardcodedBaseConfig(hw_type)
        config = config_runner.get_config(args.image)
        for key, val in config.items():
            print(f'export {key}={shlex.quote(val)}')
    except Exception as e:
        # Any error will signal fallback
        print(f"# Fallback due to error: {e}", file=sys.stderr)
        sys.exit(42)


if __name__ == "__main__":
    main()
