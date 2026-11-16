import os
import re
import subprocess
import sys
import traceback
import warnings


MIN_CUDA_VERSION = "12.1"
MIN_ROCM_VERSION = "5.4"
MIN_PYTHON_VERSION = (3, 10)


class VerifyDynamoError(BaseException):
    pass


def check_python():
    if sys.version_info < MIN_PYTHON_VERSION:
        raise VerifyDynamoError(
            f"Python version not supported: {sys.version_info} "
            f"- minimum requirement: {MIN_PYTHON_VERSION}"
        )
    return sys.version_info


def check_torch():
    import torch

    return torch.__version__


# based on torch/utils/cpp_extension.py
def get_cuda_version():
    from torch.torch_version import TorchVersion
    from torch.utils import cpp_extension

    CUDA_HOME = cpp_extension._find_cuda_home()
    if not CUDA_HOME:
        raise VerifyDynamoError(cpp_extension.CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    cuda_version_str = (
        subprocess.check_output([nvcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )
    cuda_version = re.search(r"release (\d+[.]\d+)", cuda_version_str)
    if cuda_version is None:
        raise VerifyDynamoError("CUDA version not found in `nvcc --version` output")

    cuda_str_version = cuda_version.group(1)
    return TorchVersion(cuda_str_version)


def _find_rocm_home():
    from torch.utils import cpp_extension

    rocm_home = cpp_extension._find_rocm_home()
    if not rocm_home:
        raise VerifyDynamoError(
            "ROCM was not found on the system, please set ROCM_HOME environment variable"
        )
    return rocm_home


def _parse_version(version_str, components=None):
    from torch.torch_version import TorchVersion

    version = version_str.split("-", maxsplit=1)[0]
    parts = version.split(".")
    if components is not None:
        parts = parts[:components]
    return TorchVersion(".".join(parts))


def get_rocm_sdk_version():
    """System ROCm SDK version from rocm-core/rocm_version.h, or None if absent.

    LoadHIP.cmake parses the same header to produce torch.version.rocm. Old
    consumer HIP SDKs do not ship it.
    """
    from torch.torch_version import TorchVersion

    rocm_home = _find_rocm_home()
    header = os.path.join(rocm_home, "include", "rocm-core", "rocm_version.h")
    if not os.path.isfile(header):
        return None

    with open(header) as f:
        content = f.read()
    major = re.search(r"ROCM_VERSION_MAJOR\s+(\d+)", content)
    minor = re.search(r"ROCM_VERSION_MINOR\s+(\d+)", content)
    patch = re.search(r"ROCM_VERSION_PATCH\s+(\d+)", content)
    if major is None or minor is None or patch is None:
        return None
    return TorchVersion(f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}")


def get_hip_version():
    """System HIP version from `hipcc --version`."""
    from torch.torch_version import TorchVersion
    from torch.utils import cpp_extension

    rocm_home = _find_rocm_home()
    hipcc = os.path.join(rocm_home, "bin", "hipcc")
    hip_version_str = (
        subprocess.check_output([hipcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )
    hip_version = re.search(r"HIP version: (\d+[.]\d+)", hip_version_str)

    if hip_version is None:
        raise VerifyDynamoError("HIP version not found in `hipcc --version` output")

    return TorchVersion(hip_version.group(1))


def check_cuda():
    import torch
    from torch.torch_version import TorchVersion

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return None

    torch_cuda_ver = TorchVersion(torch.version.cuda)

    # check if torch cuda version matches system cuda version
    cuda_ver = get_cuda_version()
    if cuda_ver != torch_cuda_ver:
        # raise VerifyDynamoError(
        warnings.warn(
            f"CUDA version mismatch, `torch` version: {torch_cuda_ver}, env version: {cuda_ver}"
        )

    if torch_cuda_ver < MIN_CUDA_VERSION:
        # raise VerifyDynamoError(
        warnings.warn(
            f"(`torch`) CUDA version not supported: {torch_cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )
    if cuda_ver < MIN_CUDA_VERSION:
        # raise VerifyDynamoError(
        warnings.warn(
            f"(env) CUDA version not supported: {cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )

    return cuda_ver if torch.version.hip is None else "None"


def check_rocm():
    import torch

    if not torch.cuda.is_available() or torch.version.hip is None:
        return None

    # Compare SDK to SDK when both sides have it; else HIP-to-HIP.
    # Truncate to major.minor so 7.14.0 vs 7.14.1 does not warn.
    system_sdk = get_rocm_sdk_version()
    torch_sdk = getattr(torch.version, "rocm", None)
    if system_sdk is not None and torch_sdk:
        torch_rocm_ver = _parse_version(torch_sdk, components=2)
        rocm_ver = _parse_version(str(system_sdk), components=2)
    else:
        torch_rocm_ver = _parse_version(torch.version.hip, components=2)
        rocm_ver = get_hip_version()
    if rocm_ver != torch_rocm_ver:
        warnings.warn(
            f"ROCm version mismatch, `torch` version: {torch_rocm_ver}, env version: {rocm_ver}"
        )
    if torch_rocm_ver < MIN_ROCM_VERSION:
        warnings.warn(
            f"(`torch`) ROCm version not supported: {torch_rocm_ver} "
            f"- minimum requirement: {MIN_ROCM_VERSION}"
        )
    if rocm_ver < MIN_ROCM_VERSION:
        warnings.warn(
            f"(env) ROCm version not supported: {rocm_ver} "
            f"- minimum requirement: {MIN_ROCM_VERSION}"
        )

    return rocm_ver if torch.version.hip else "None"


def check_dynamo(backend, device, err_msg) -> None:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available -- skipping CUDA check on {backend} backend\n")
        return

    try:
        import torch._dynamo as dynamo

        if device == "cuda":
            from torch.utils._triton import has_triton

            if not has_triton():
                print(
                    f"WARNING: CUDA available but triton cannot be used. "
                    f"Your GPU may not be supported. "
                    f"Skipping CUDA check on {backend} backend\n"
                )
                return

        dynamo.reset()

        @dynamo.optimize(backend, nopython=True)
        def fn(x):
            return x + x

        class Module(torch.nn.Module):
            def forward(self, x):
                return x + x

        mod = Module()
        opt_mod = dynamo.optimize(backend, nopython=True)(mod)

        for f in (fn, opt_mod):
            x = torch.randn(10, 10).to(device)
            x.requires_grad = True
            y = f(x)
            torch.testing.assert_close(y, x + x)
            z = y.sum()
            z.backward()
            torch.testing.assert_close(x.grad, 2 * torch.ones_like(x))
    except Exception:
        sys.stderr.write(traceback.format_exc() + "\n" + err_msg + "\n\n")
        sys.exit(1)


_SANITY_CHECK_ARGS = (
    ("eager", "cpu", "CPU eager sanity check failed"),
    ("eager", "cuda", "CUDA eager sanity check failed"),
    ("aot_eager", "cpu", "CPU aot_eager sanity check failed"),
    ("aot_eager", "cuda", "CUDA aot_eager sanity check failed"),
    ("inductor", "cpu", "CPU inductor sanity check failed"),
    (
        "inductor",
        "cuda",
        "CUDA inductor sanity check failed\n"
        + "NOTE: Please check that you installed the correct hash/version of `triton`",
    ),
)


def main() -> None:
    python_ver = check_python()
    torch_ver = check_torch()
    cuda_ver = check_cuda()
    rocm_ver = check_rocm()
    print(
        f"Python version: {python_ver.major}.{python_ver.minor}.{python_ver.micro}\n"
        f"`torch` version: {torch_ver}\n"
        f"CUDA version: {cuda_ver}\n"
        f"ROCM version: {rocm_ver}\n"
    )
    for args in _SANITY_CHECK_ARGS:
        if sys.version_info >= (3, 15):
            warnings.warn("Dynamo not yet supported in Python 3.15.")
        check_dynamo(*args)
    print("All required checks passed")


if __name__ == "__main__":
    main()
