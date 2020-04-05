PHASES = ["build", "test"]

CUDA_VERSIONS = [
    None,  # cpu build
    "92",
    "101",
    "102",
]

STANDARD_PYTHON_VERSIONS = [
    "3.5",
    "3.6",
    "3.7",
    "3.8"
]

STANDARD_PYTHON_VERSIONS_WINDOWS = STANDARD_PYTHON_VERSIONS.copy()
STANDARD_PYTHON_VERSIONS_WINDOWS.remove("3.5")
