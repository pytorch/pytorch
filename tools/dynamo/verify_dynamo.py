import os
import re
import subprocess
import sys
import traceback
import warnings

from pkg_resources import packaging

MIN_CUDA_VERSION = packaging.version.parse("11.6")
MIN_ROCM_VERSION = packaging.version.parse("5.4")
MIN_PYTHON_VERSION = (3, 8)


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

    return packaging.version.parse(torch.__version__)


# based on torch/utils/cpp_extension.py
def get_cuda_version():
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
    return packaging.version.parse(cuda_str_version)


def get_rocm_version():
    from torch.utils import cpp_extension

    ROCM_HOME = cpp_extension._find_rocm_home()
    if not ROCM_HOME:
        raise VerifyDynamoError(
            "ROCM was not found on the system, please set ROCM_HOME environment variable"
        )

    hipcc = os.path.join(ROCM_HOME, "bin", "hipcc")
    hip_version_str = (
        subprocess.check_output([hipcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )
    hip_version = re.search(r"HIP version: (\d+[.]\d+)", hip_version_str)

    if hip_version is None:
        raise VerifyDynamoError("HIP version not found in `hipcc --version` output")

    hip_str_version = hip_version.group(1)

    return packaging.version.parse(hip_str_version)


def check_cuda():
    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return None

    torch_cuda_ver = packaging.version.parse(torch.version.cuda)

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

    # Extracts main ROCm version from full string
    torch_rocm_ver = packaging.version.parse(
        ".".join(list(torch.version.hip.split(".")[0:2]))
    )

    # check if torch rocm version matches system rocm version
    rocm_ver = get_rocm_version()
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


def check_dynamo(backend, device, err_msg):
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


def main():
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
        if sys.version_info >= (3, 12):
            warnings.warn("Dynamo not yet supported in Python 3.12. Skipping check.")
            continue
        check_dynamo(*args)
    print("All required checks passed")


if __name__ == "__main__":
    main()
