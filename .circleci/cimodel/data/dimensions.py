PHASES = ["build", "test"]

CUDA_VERSIONS = [
    "92",
    "101",
    "102",
    "110",
]

ROCM_VERSIONS = [
    "3.7",
]

GPU_VERSIONS = [None] + ["cuda" + v for v in CUDA_VERSIONS] + ["rocm" + v for v in ROCM_VERSIONS]

STANDARD_PYTHON_VERSIONS = [
    "3.6",
    "3.7",
    "3.8",
]
