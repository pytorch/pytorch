"""Record complete package identities in original SFT compilation workers."""

import functools
import hashlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def install_trace():
    import triton
    import triton.compiler
    import triton.compiler.compiler as compiler
    from triton._C import libtriton
    from triton.backends.nvidia.compiler import CUDABackend

    import torch

    native = Path(os.environ["PIN_TITAN_CURRENT_NATIVE_SITE"])
    package = Path(os.environ["PIN_TITAN_CURRENT_TRITON_SITE"])
    if torch.version.git_version != os.environ["PIN_TITAN_CURRENT_NATIVE_HEAD"]:
        raise RuntimeError("Native Torch source differs")
    if not Path(torch.__file__).resolve().is_relative_to(native):
        raise RuntimeError("Torch escaped the complete native package")
    if not Path(torch._C.__file__).resolve().is_relative_to(native):
        raise RuntimeError("Torch native library escaped the complete native package")
    if digest(torch._C.__file__) != os.environ["PIN_TITAN_CURRENT_TORCH_NATIVE_SHA256"]:
        raise RuntimeError("Torch native library differs from its original wheel")
    if not Path(triton.__file__).resolve().is_relative_to(package):
        raise RuntimeError("Triton escaped its complete native package")
    if not Path(libtriton.__file__).resolve().is_relative_to(package):
        raise RuntimeError("Compiler escaped its complete Triton package")
    if digest(libtriton.__file__) != os.environ["PIN_TITAN_CURRENT_COMPILER_SHA256"]:
        raise RuntimeError("Compiler differs from its complete official wheel")
    version = importlib.metadata.version("triton")
    if os.environ["PIN_TITAN_CURRENT_TRITON_SHORT"] not in version:
        raise RuntimeError("Triton metadata differs")
    directory = Path(os.environ["PIN_TITAN_CURRENT_WORKERS"])
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "pid": os.getpid(),
        "executable": sys.executable,
        "torch_sha": torch.version.git_version,
        "torch_package": torch.__file__,
        "torch_native": torch._C.__file__,
        "torch_native_sha256": digest(torch._C.__file__),
        "triton_version": version,
        "triton_package": triton.__file__,
        "triton_native": libtriton.__file__,
        "triton_native_sha256": digest(libtriton.__file__),
    }

    def record(event, **data):
        with (directory / f"{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps({"event": event, **provenance, **data}) + "\n")

    record("import")
    original_parse = CUDABackend.parse_options

    @functools.wraps(original_parse)
    def parse_options(backend, options):
        result = original_parse(backend, options)
        record(
            "parse_options",
            requested=options.get("enable_fp_fusion"),
            resolved=result.enable_fp_fusion,
        )
        return result

    CUDABackend.parse_options = parse_options
    original_compile = compiler.compile

    @functools.wraps(original_compile)
    def compile_kernel(*args, **kwargs):
        record("compile_request")
        result = original_compile(*args, **kwargs)
        record(
            "compiled",
            name=result.metadata.name,
            resolved=result.metadata.enable_fp_fusion,
        )
        return result

    compiler.compile = triton.compiler.compile = triton.compile = compile_kernel


if os.environ.get("PIN_TITAN_CURRENT_WORKERS"):
    try:
        install_trace()
    except BaseException:
        traceback.print_exc()
        os._exit(97)
