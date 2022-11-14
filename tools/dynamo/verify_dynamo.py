import os
import re
import subprocess
import sys
import traceback
import warnings

from pkg_resources import packaging

MIN_CUDA_VERSION = packaging.version.parse("11.6")
MIN_PYTHON_VERSION = (3, 7)


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


def check_cuda():
    import torch

    if not torch.cuda.is_available():
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

    return cuda_ver


def check_dynamo(backend, device, err_msg):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available -- skipping CUDA check on {backend} backend\n")
        return

    try:
        import torch._dynamo as dynamo

        if device == "cuda":
            import torch._inductor.utils as utils

            if not utils.has_triton():
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
            def __init__(self):
                super().__init__()

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
    print(
        f"Python version: {python_ver.major}.{python_ver.minor}.{python_ver.micro}\n"
        f"`torch` version: {torch_ver}\n"
        f"CUDA version: {cuda_ver}\n"
    )
    for args in _SANITY_CHECK_ARGS:
        check_dynamo(*args)
    print("All required checks passed")


if __name__ == "__main__":
    main()
    sys.exit(1)
