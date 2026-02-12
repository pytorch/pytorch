"""
ROCm Multi-Architecture Support Utilities
Compile LLVM IR to multi-arch bundles that HIP can load automatically.
"""

import os
import subprocess
from typing import Optional

import torch
from torch.utils.cpp_extension import _join_rocm_home, ROCM_HOME


def get_rocm_compiler() -> str:
    """
    Get path to ROCm's clang compiler.
    Uses PyTorch's ROCM_HOME detection.

    Returns:
        Path to clang compiler

    Raises:
        RuntimeError: If ROCm is not found
    """
    if ROCM_HOME is None:
        raise RuntimeError(
            "ROCm installation not found. "
            "PyTorch was not built with ROCm support or ROCM_HOME is not set."
        )

    # ROCm's clang is at <ROCM_HOME>/llvm/bin/clang
    clang_path = _join_rocm_home("llvm", "bin", "clang")

    if not os.path.exists(clang_path):
        raise RuntimeError(
            f"ROCm clang not found at {clang_path}. ROCM_HOME is set to {ROCM_HOME}"
        )

    return clang_path


def get_rocm_bundler() -> str:
    """
    Get path to clang-offload-bundler.
    Uses PyTorch's ROCM_HOME detection.

    Returns:
        Path to bundler

    Raises:
        RuntimeError: If bundler is not found
    """
    if ROCM_HOME is None:
        raise RuntimeError(
            "ROCm installation not found. "
            "PyTorch was not built with ROCm support or ROCM_HOME is not set."
        )

    # Bundler is at <ROCM_HOME>/llvm/bin/clang-offload-bundler
    bundler_path = _join_rocm_home("llvm", "bin", "clang-offload-bundler")

    if not os.path.exists(bundler_path):
        raise RuntimeError(
            f"clang-offload-bundler not found at {bundler_path}. "
            f"ROCM_HOME is set to {ROCM_HOME}"
        )

    return bundler_path


def get_rocm_target_archs() -> list[str]:
    """
    Get target architectures from environment or config.
    Returns: List of architecture strings (e.g., ['gfx90a', 'gfx942'])
    """
    # Check PYTORCH_ROCM_ARCH environment variable
    env_archs = os.environ.get("PYTORCH_ROCM_ARCH", "").strip()
    if env_archs:
        archs = [arch.strip() for arch in env_archs.replace(";", ",").split(",")]
        archs = [arch for arch in archs if arch]
        if archs:
            return archs

    # Try to get from inductor config
    try:
        from torch._inductor import config

        if hasattr(config, "rocm") and hasattr(config.rocm, "target_archs"):
            archs = config.rocm.target_archs
            if archs:
                return archs

    except Exception:
        pass

    return torch.cuda.get_arch_list()


def compile_llvm_ir_to_code_object(
    llvm_ir_path: str, output_path: str, target_arch: str
) -> bool:
    """
    Compile unbundled LLVM IR to a single-arch code object.

    Args:
        llvm_ir_path: Path to .ll file
        output_path: Where to write .hsaco file
        target_arch: Target architecture (e.g., 'gfx90a')

    Returns:
        True if successful
    """
    if not os.path.exists(llvm_ir_path):
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        clang = get_rocm_compiler()
    except RuntimeError:
        return False

    # Using clang and not hipcc since we are not compiling source code
    # Instead we use the LLVM IR (.ll) provided by triton
    cmd = [
        clang,
        "-target",
        "amdgcn-amd-amdhsa",
        f"-mcpu={target_arch}",
        llvm_ir_path,
        "-o",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not os.path.exists(output_path):
            return False

        return True

    except subprocess.CalledProcessError:
        return False


def create_multiarch_bundle(code_objects: dict, output_bundle_path: str) -> bool:
    """
    Bundle multiple architecture code objects into a single multi-arch bundle.

    Uses clang-offload-bundler to create a fat binary that HIP runtime can load.
    The runtime automatically selects the correct architecture at load time.

    Args:
        code_objects: Dict mapping architecture to code object path
        output_bundle_path: Path for output bundle

    Returns:
        True if successful
    """
    if not code_objects:
        return False

    os.makedirs(os.path.dirname(output_bundle_path), exist_ok=True)

    try:
        bundler = get_rocm_bundler()
    except RuntimeError:
        return False

    # Build targets and inputs lists for clang-offload-bundler
    targets = ["host-x86_64-unknown-linux-gnu"]

    # We include a dummy host entry to satisfy the bundler format
    inputs = ["/dev/null"]

    for arch, path in sorted(code_objects.items()):
        if not os.path.exists(path):
            continue
        # hipv4 = HIP version 4 code object format
        # amdgcn-amd-amdhsa = target triple for ROCm/HSA runtime
        # arch = specific GPU (gfx90a, gfx942, etc.)
        targets.append(f"hipv4-amdgcn-amd-amdhsa--{arch}")
        inputs.append(path)

    if len(inputs) == 1:  # Only host, no device code
        return False

    cmd = [
        bundler,
        "--type=o",
        # CRITICAL: HIP runtime expects 4096-byte alignment for loading bundles
        # Without this, hipModuleLoadData gives segmentation fault
        "-bundle-align=4096",  # CRITICAL: Required by HIP runtime!
        f"--targets={','.join(targets)}",
    ]

    for input_file in inputs:
        cmd.append(f"--input={input_file}")

    cmd.append(f"--output={output_bundle_path}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not os.path.exists(output_bundle_path):
            return False

        return True

    except subprocess.CalledProcessError:
        return False


def compile_multiarch_bundle_from_llvm_ir(
    llvm_ir_path: str, output_bundle_path: str, target_archs: Optional[list[str]] = None
) -> bool:
    """
    Complete workflow: LLVM IR → multiple code objects → bundle.

    This is the main entry point for multi-arch compilation.

    Args:
        llvm_ir_path: Path to .ll file
        output_bundle_path: Where to write bundle
        target_archs: Optional list of architectures

    Returns:
        True if successful
    """
    if target_archs is None:
        # Get architectures from environment variable or config
        target_archs = get_rocm_target_archs()

    # Step 1: Compile LLVM IR to code object for each architecture
    code_objects = {}
    temp_dir = os.path.dirname(output_bundle_path)
    kernel_name = os.path.splitext(os.path.basename(llvm_ir_path))[0]

    for arch in target_archs:
        # Create temporary single-architecture code object
        # Format: kernel_name_gfx90a.co, kernel_name_gfx942.co, etc.
        co_path = os.path.join(temp_dir, f"{kernel_name}_{arch}.co")

        # Compile with clang backend: LLVM IR → GPU machine code
        if compile_llvm_ir_to_code_object(llvm_ir_path, co_path, arch):
            code_objects[arch] = co_path

    if not code_objects:
        return False

    # Step 2: Bundle all code objects together
    # Uses clang-offload-bundler to create fat binary
    success = create_multiarch_bundle(code_objects, output_bundle_path)

    # Step 3: Clean up temporary single-arch code objects
    # The bundle contains all the code, so intermediates are no longer needed
    for co_path in code_objects.values():
        try:
            os.remove(co_path)
        except Exception:
            pass

    return success
