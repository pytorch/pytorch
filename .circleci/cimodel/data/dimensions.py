PHASES = ["build", "test"]

CUDA_VERSIONS = [
    "102",
    "113",
    "116",
    "117",
]

ROCM_VERSIONS = [
    "4.3.1",
    "4.5.2",
]

ROCM_VERSION_LABELS = ["rocm" + v for v in ROCM_VERSIONS]

GPU_VERSIONS = [None] + ["cuda" + v for v in CUDA_VERSIONS] + ROCM_VERSION_LABELS

STANDARD_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]
