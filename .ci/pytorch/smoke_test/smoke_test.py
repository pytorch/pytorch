import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F


if "MATRIX_GPU_ARCH_VERSION" in os.environ:
    gpu_arch_ver = os.getenv("MATRIX_GPU_ARCH_VERSION")
else:
    gpu_arch_ver = os.getenv("GPU_ARCH_VERSION")  # Use fallback if available
gpu_arch_type = os.getenv("MATRIX_GPU_ARCH_TYPE")
channel = os.getenv("MATRIX_CHANNEL")
package_type = os.getenv("MATRIX_PACKAGE_TYPE")
target_os = os.getenv("TARGET_OS", sys.platform)
BASE_DIR = Path(__file__).parent.parent.parent

is_cuda_system = gpu_arch_type == "cuda"
NIGHTLY_ALLOWED_DELTA = 3

MODULES = [
    {
        "name": "torchvision",
        "repo": "https://github.com/pytorch/vision.git",
        "smoke_test": "./vision/test/smoke_test.py",
        "extension": "extension",
        "repo_name": "vision",
    },
    {
        "name": "torchaudio",
        "repo": "https://github.com/pytorch/audio.git",
        "smoke_test": "./audio/test/smoke_test/smoke_test.py --no-ffmpeg",
        "extension": "_extension",
        "repo_name": "audio",
    },
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        return output


def load_json_from_basedir(filename: str):
    try:
        with open(BASE_DIR / filename) as fptr:
            return json.load(fptr)
    except FileNotFoundError as exc:
        raise ImportError(f"File {filename} not found error: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise ImportError(f"Invalid JSON {filename}") from exc


def read_release_matrix():
    return load_json_from_basedir("release_matrix.json")


def test_numpy():
    import numpy as np

    x = np.arange(5)
    torch.tensor(x)


def check_version(package: str) -> None:
    release_version = os.getenv("RELEASE_VERSION")
    # if release_version is specified, use it to validate the packages
    if release_version:
        release_matrix = read_release_matrix()
        stable_version = release_matrix["torch"]
    else:
        stable_version = os.getenv("MATRIX_STABLE_VERSION")

    # only makes sense to check nightly package where dates are known
    if channel == "nightly":
        check_nightly_binaries_date(package)
    elif stable_version is not None:
        if not torch.__version__.startswith(stable_version):
            raise RuntimeError(
                f"Torch version mismatch, expected {stable_version} for channel {channel}. But its {torch.__version__}"
            )

        if release_version and package == "all":
            for module in MODULES:
                imported_module = importlib.import_module(module["name"])
                module_version = imported_module.__version__
                if not module_version.startswith(release_matrix[module["name"]]):
                    raise RuntimeError(
                        f"{module['name']} version mismatch, expected: \
                            {release_matrix[module['name']]} for channel {channel}. But its {module_version}"
                    )
                else:
                    print(f"{module['name']} version actual: {module_version} expected: \
                        {release_matrix[module['name']]} for channel {channel}.")

    else:
        print(f"Skip version check for channel {channel} as stable version is None")


def check_nightly_binaries_date(package: str) -> None:
    from datetime import datetime

    format_dt = "%Y%m%d"

    date_t_str = re.findall("dev\\d+", torch.__version__)
    date_t_delta = datetime.now() - datetime.strptime(date_t_str[0][3:], format_dt)
    if date_t_delta.days >= NIGHTLY_ALLOWED_DELTA:
        raise RuntimeError(
            f"the binaries are from {date_t_str} and are more than {NIGHTLY_ALLOWED_DELTA} days old!"
        )

    if package == "all":
        for module in MODULES:
            imported_module = importlib.import_module(module["name"])
            module_version = imported_module.__version__
            date_m_str = re.findall("dev\\d+", module_version)
            date_m_delta = datetime.now() - datetime.strptime(
                date_m_str[0][3:], format_dt
            )
            print(f"Nightly date check for {module['name']} version {module_version}")
            if date_m_delta.days > NIGHTLY_ALLOWED_DELTA:
                raise RuntimeError(
                    f"Expected {module['name']} to be less then {NIGHTLY_ALLOWED_DELTA} days. But its {date_m_delta}"
                )


def test_cuda_runtime_errors_captured() -> None:
    cuda_exception_missed = True
    try:
        print("Testing test_cuda_runtime_errors_captured")
        torch._assert_async(torch.tensor(0, device="cuda"))
        torch._assert_async(torch.tensor(0 + 0j, device="cuda"))
    except RuntimeError as e:
        if re.search("CUDA", f"{e}"):
            print(f"Caught CUDA exception with success: {e}")
            cuda_exception_missed = False
        else:
            raise e
    if cuda_exception_missed:
        raise RuntimeError("Expected CUDA RuntimeError but have not received!")


def smoke_test_cuda(
    package: str, runtime_error_check: str, torch_compile_check: str
) -> None:
    if not torch.cuda.is_available() and is_cuda_system:
        raise RuntimeError(f"Expected CUDA {gpu_arch_ver}. However CUDA is not loaded.")

    if package == "all" and is_cuda_system:
        for module in MODULES:
            imported_module = importlib.import_module(module["name"])
            # TBD for vision move extension module to private so it will
            # be _extention.
            version = "N/A"
            if module["extension"] == "extension":
                version = imported_module.extension._check_cuda_version()
            else:
                version = imported_module._extension._check_cuda_version()
            print(f"{module['name']} CUDA: {version}")

    # torch.compile is available on macos-arm64 and Linux for python 3.8-3.13
    if (
        torch_compile_check == "enabled"
        and sys.version_info < (3, 13, 0)
        and target_os in ["linux", "linux-aarch64", "macos-arm64", "darwin"]
    ):
        smoke_test_compile("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        if torch.version.cuda != gpu_arch_ver:
            raise RuntimeError(
                f"Wrong CUDA version. Loaded: {torch.version.cuda} Expected: {gpu_arch_ver}"
            )
        print(f"torch cuda: {torch.version.cuda}")
        # todo add cudnn version validation
        print(f"torch cudnn: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled? {torch.backends.cudnn.enabled}")

        torch.cuda.init()
        print("CUDA initialized successfully")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

        # nccl is availbale only on Linux
        if sys.platform in ["linux", "linux2"]:
            print(f"torch nccl version: {torch.cuda.nccl.version()}")

        if runtime_error_check == "enabled":
            test_cuda_runtime_errors_captured()


def smoke_test_conv2d() -> None:
    import torch.nn as nn

    print("Testing smoke_test_conv2d")
    # With square kernels and equal stride
    m = nn.Conv2d(16, 33, 3, stride=2)
    # non-square kernels and unequal stride and with padding
    m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    assert m is not None
    # non-square kernels and unequal stride and with padding and dilation
    basic_conv = nn.Conv2d(
        16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
    )
    input = torch.randn(20, 16, 50, 100)
    output = basic_conv(input)

    if is_cuda_system:
        print("Testing smoke_test_conv2d with cuda")
        conv = nn.Conv2d(3, 3, 3).cuda()
        x = torch.randn(1, 3, 24, 24, device="cuda")
        with torch.cuda.amp.autocast():
            out = conv(x)
        assert out is not None

        supported_dtypes = [torch.float16, torch.float32, torch.float64]
        for dtype in supported_dtypes:
            print(f"Testing smoke_test_conv2d with cuda for {dtype}")
            conv = basic_conv.to(dtype).cuda()
            input = torch.randn(20, 16, 50, 100, device="cuda").type(dtype)
            output = conv(input)
            assert output is not None


def test_linalg(device="cpu") -> None:
    print(f"Testing smoke_test_linalg on {device}")
    A = torch.randn(5, 3, device=device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    assert (
        U.shape == A.shape
        and S.shape == torch.Size([3])
        and Vh.shape == torch.Size([3, 3])
    )
    torch.dist(A, U @ torch.diag(S) @ Vh)

    U, S, Vh = torch.linalg.svd(A)
    assert (
        U.shape == torch.Size([5, 5])
        and S.shape == torch.Size([3])
        and Vh.shape == torch.Size([3, 3])
    )
    torch.dist(A, U[:, :3] @ torch.diag(S) @ Vh)

    A = torch.randn(7, 5, 3, device=device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    torch.dist(A, U @ torch.diag_embed(S) @ Vh)

    if device == "cuda":
        supported_dtypes = [torch.float32, torch.float64]
        for dtype in supported_dtypes:
            print(f"Testing smoke_test_linalg with cuda for {dtype}")
            A = torch.randn(20, 16, 50, 100, device=device, dtype=dtype)
            torch.linalg.svd(A)


def smoke_test_compile(device: str = "cpu") -> None:
    supported_dtypes = [torch.float16, torch.float32, torch.float64]

    def foo(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + torch.cos(x)

    for dtype in supported_dtypes:
        print(f"Testing smoke_test_compile for {device} and {dtype}")
        x = torch.rand(3, 3, device=device).type(dtype)
        x_eager = foo(x)
        x_pt2 = torch.compile(foo)(x)
        torch.testing.assert_close(x_eager, x_pt2)

    # Check that SIMD were detected for the architecture
    if device == "cpu":
        from torch._inductor.codecache import pick_vec_isa

        isa = pick_vec_isa()
        if not isa:
            raise RuntimeError("Can't detect vectorized ISA for CPU")
        print(f"Picked CPU ISA {type(isa).__name__} bit width {isa.bit_width()}")

    # Reset torch dynamo since we are changing mode
    torch._dynamo.reset()
    dtype = torch.float32
    torch.set_float32_matmul_precision("high")
    print(f"Testing smoke_test_compile with mode 'max-autotune' for {dtype}")
    x = torch.rand(64, 1, 28, 28, device=device).type(torch.float32)
    model = Net().to(device=device)
    x_pt2 = torch.compile(model, mode="max-autotune")(x)


def smoke_test_modules():
    cwd = os.getcwd()
    for module in MODULES:
        if module["repo"]:
            if not os.path.exists(f"{cwd}/{module['repo_name']}"):
                print(f"Path does not exist: {cwd}/{module['repo_name']}")
                try:
                    subprocess.check_output(
                        f"git clone --depth 1 {module['repo']}",
                        stderr=subprocess.STDOUT,
                        shell=True,
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"Cloning {module['repo']} FAIL: {exc.returncode} Output: {exc.output}"
                    ) from exc
            try:
                smoke_test_command = f"python3 {module['smoke_test']}"
                if target_os == "windows":
                    smoke_test_command = f"python {module['smoke_test']}"
                output = subprocess.check_output(
                    smoke_test_command,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    universal_newlines=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Module {module['name']} FAIL: {exc.returncode} Output: {exc.output}"
                ) from exc
            else:
                print(f"Output: \n{output}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        help="Package to include in smoke testing",
        type=str,
        choices=["all", "torchonly"],
        default="all",
    )
    parser.add_argument(
        "--runtime-error-check",
        help="No Runtime Error check",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
    )
    parser.add_argument(
        "--torch-compile-check",
        help="Check torch compile",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
    )
    options = parser.parse_args()
    print(f"torch: {torch.__version__}")
    print(torch.__config__.parallel_info())

    check_version(options.package)
    smoke_test_conv2d()
    test_linalg()
    test_numpy()
    if is_cuda_system:
        test_linalg("cuda")

    if options.package == "all":
        smoke_test_modules()

    smoke_test_cuda(
        options.package, options.runtime_error_check, options.torch_compile_check
    )


if __name__ == "__main__":
    main()
