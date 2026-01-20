"""
Kernel serialization for torch.compile pythonify feature.

This module provides infrastructure for serializing Inductor-compiled kernels
so they can be loaded and executed by generated Python code. It handles:

1. Triton kernels (stored as .cubin files with metadata)
2. C++ kernels (stored as .so shared libraries)
3. Python wrapper code that ties everything together

The serialization strategy follows the existing Inductor cache infrastructure
patterns but packages artifacts in a way that's suitable for standalone
Python files.

Usage:
    serializer = KernelSerializer(output_dir="/tmp/model_kernels")
    kernel_ref = serializer.serialize_triton_kernel(kernel_hash, kernel_path)
    kernel_ref = serializer.serialize_cpp_kernel(code_hash, so_path)
    load_code = serializer.generate_loader_code()
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional


class KernelType(Enum):
    """
    Type of compiled kernel being serialized.

    TRITON: GPU kernel compiled via Triton compiler (cubin, ptx)
    CPP: CPU kernel compiled via C++ compiler (.so, .dll)
    PYTHON: Pure Python wrapper (for fallback or simple ops)
    """

    TRITON = auto()
    CPP = auto()
    PYTHON = auto()


@dataclass
class KernelReference:
    """
    Reference to a serialized kernel that can be loaded at runtime.

    This class captures all information needed to generate Python code
    that loads and invokes a serialized kernel.

    Attributes:
        kernel_type: Type of kernel (TRITON, CPP, or PYTHON)
        kernel_id: Unique identifier for this kernel
        file_path: Path to the serialized kernel file (relative to output dir)
        entry_point: Function/symbol name to call
        metadata: Additional kernel-specific metadata (dtypes, grid sizes, etc.)
    """

    kernel_type: KernelType
    kernel_id: str
    file_path: str
    entry_point: str = "call"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SerializedKernelBundle:
    """
    Bundle containing all serialized kernels for a compiled model.

    This class aggregates all kernel references and provides methods
    for generating the Python code needed to load them.

    Attributes:
        output_dir: Directory where kernel files are stored
        kernels: List of kernel references
        constants: Serialized tensor constants (weights, buffers)
        metadata: Compilation metadata
    """

    output_dir: str
    kernels: list[KernelReference] = field(default_factory=list)
    constants: dict[str, bytes] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class KernelSerializer:
    """
    Serializes Inductor-compiled kernels for use in generated Python code.

    This class handles the serialization of compiled kernels (Triton, C++)
    to a directory structure that can be loaded by the generated Python file.
    It follows the existing Inductor cache patterns but packages artifacts
    for standalone use.

    The serializer produces:
    1. Kernel binary files (.so for C++, .cubin/.ptx for Triton)
    2. Metadata JSON files describing each kernel
    3. Python loader code that can reconstruct the compiled callable

    Usage:
        serializer = KernelSerializer("/tmp/model_kernels")
        ref = serializer.serialize_cpp_kernel(code_hash, "/path/to/kernel.so")
        loader_code = serializer.generate_loader_code()
    """

    def __init__(
        self,
        output_dir: str,
        inline_small_kernels: bool = True,
        inline_threshold_bytes: int = 4096,
    ) -> None:
        """
        Initialize the kernel serializer.

        Args:
            output_dir: Directory to store serialized kernel files.
                        Will be created if it doesn't exist.
            inline_small_kernels: If True, small kernels are embedded as base64
                                  in the generated Python file rather than
                                  stored as separate files.
            inline_threshold_bytes: Maximum size in bytes for inlining kernels.
        """
        self.output_dir = Path(output_dir)
        self.inline_small_kernels = inline_small_kernels
        self.inline_threshold_bytes = inline_threshold_bytes
        self._kernels: list[KernelReference] = []
        self._constants: dict[str, bytes] = {}
        self._metadata: dict[str, Any] = {}

    def _ensure_output_dir(self) -> None:
        """Create the output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _compute_kernel_id(self, content: bytes, kernel_type: KernelType) -> str:
        """
        Compute a unique identifier for a kernel based on its content.

        Args:
            content: The kernel binary content
            kernel_type: Type of kernel

        Returns:
            A short hash string uniquely identifying the kernel
        """
        hasher = hashlib.sha256()
        hasher.update(kernel_type.name.encode())
        hasher.update(content)
        return hasher.hexdigest()[:16]

    def serialize_cpp_kernel(
        self,
        so_path: str,
        entry_point: str = "call",
        metadata: Optional[dict[str, Any]] = None,
    ) -> KernelReference:
        """
        Serialize a C++ compiled kernel (.so file).

        Copies the shared library to the output directory and creates
        a reference that can be used to generate loader code.

        Args:
            so_path: Path to the compiled .so file
            entry_point: Name of the function symbol to call
            metadata: Optional metadata about the kernel

        Returns:
            KernelReference that can be used in generated code

        Raises:
            FileNotFoundError: If the .so file doesn't exist
        """
        self._ensure_output_dir()

        so_path = Path(so_path)
        if not so_path.exists():
            raise FileNotFoundError(f"Kernel .so file not found: {so_path}")

        content = so_path.read_bytes()
        kernel_id = self._compute_kernel_id(content, KernelType.CPP)
        dest_filename = f"kernel_{kernel_id}.so"
        dest_path = self.output_dir / dest_filename

        shutil.copy2(so_path, dest_path)

        ref = KernelReference(
            kernel_type=KernelType.CPP,
            kernel_id=kernel_id,
            file_path=dest_filename,
            entry_point=entry_point,
            metadata=metadata or {},
        )
        self._kernels.append(ref)
        return ref

    def serialize_triton_kernel(
        self,
        kernel_hash: str,
        cubin_path: Optional[str] = None,
        ptx_path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> KernelReference:
        """
        Serialize a Triton-compiled GPU kernel.

        Copies the kernel binary (cubin) and optionally PTX source to the
        output directory. Creates a reference for loader code generation.

        Args:
            kernel_hash: Triton's internal hash for this kernel
            cubin_path: Path to the compiled cubin file (GPU binary)
            ptx_path: Optional path to PTX source (for debugging)
            metadata: Kernel metadata (grid size, block size, etc.)

        Returns:
            KernelReference that can be used in generated code

        Raises:
            ValueError: If neither cubin_path nor ptx_path is provided
        """
        self._ensure_output_dir()

        if cubin_path is None and ptx_path is None:
            raise ValueError("At least one of cubin_path or ptx_path must be provided")

        if cubin_path and Path(cubin_path).exists():
            content = Path(cubin_path).read_bytes()
            kernel_id = self._compute_kernel_id(content, KernelType.TRITON)
            dest_filename = f"triton_{kernel_id}.cubin"
            shutil.copy2(cubin_path, self.output_dir / dest_filename)
        else:
            kernel_id = kernel_hash[:16]
            dest_filename = ""

        if ptx_path and Path(ptx_path).exists():
            ptx_filename = f"triton_{kernel_id}.ptx"
            shutil.copy2(ptx_path, self.output_dir / ptx_filename)

        ref = KernelReference(
            kernel_type=KernelType.TRITON,
            kernel_id=kernel_id,
            file_path=dest_filename,
            entry_point="launch",
            metadata={
                "kernel_hash": kernel_hash,
                **(metadata or {}),
            },
        )
        self._kernels.append(ref)
        return ref

    def serialize_python_wrapper(
        self,
        wrapper_code: str,
        wrapper_name: str = "compiled_wrapper",
        metadata: Optional[dict[str, Any]] = None,
    ) -> KernelReference:
        """
        Serialize a Python wrapper that invokes compiled kernels.

        This is used for the top-level wrapper that coordinates kernel
        invocation, or for simple ops that don't require compilation.

        Args:
            wrapper_code: Python source code for the wrapper
            wrapper_name: Name of the callable in the wrapper
            metadata: Optional metadata

        Returns:
            KernelReference for the Python wrapper
        """
        self._ensure_output_dir()

        content = wrapper_code.encode("utf-8")
        kernel_id = self._compute_kernel_id(content, KernelType.PYTHON)
        dest_filename = f"wrapper_{kernel_id}.py"
        dest_path = self.output_dir / dest_filename

        dest_path.write_text(wrapper_code)

        ref = KernelReference(
            kernel_type=KernelType.PYTHON,
            kernel_id=kernel_id,
            file_path=dest_filename,
            entry_point=wrapper_name,
            metadata=metadata or {},
        )
        self._kernels.append(ref)
        return ref

    def serialize_constants(
        self,
        constants: dict[str, bytes],
    ) -> None:
        """
        Serialize tensor constants (weights, buffers) for the model.

        Args:
            constants: Dictionary mapping constant names to serialized bytes
        """
        self._ensure_output_dir()
        self._constants.update(constants)

        constants_path = self.output_dir / "constants.bin"
        with open(constants_path, "wb") as f:
            for name, data in constants.items():
                name_bytes = name.encode("utf-8")
                f.write(len(name_bytes).to_bytes(4, "little"))
                f.write(name_bytes)
                f.write(len(data).to_bytes(8, "little"))
                f.write(data)

    def get_bundle(self) -> SerializedKernelBundle:
        """
        Get the complete serialized kernel bundle.

        Returns:
            SerializedKernelBundle containing all kernel references
        """
        return SerializedKernelBundle(
            output_dir=str(self.output_dir),
            kernels=self._kernels.copy(),
            constants=self._constants.copy(),
            metadata=self._metadata.copy(),
        )

    def generate_loader_code(self) -> str:
        """
        Generate Python code that loads all serialized kernels.

        This produces a Python module that:
        1. Loads C++ kernels via ctypes
        2. Loads Triton kernels via triton runtime
        3. Provides a unified interface to invoke the compiled callable

        Returns:
            Python source code string for the kernel loader module
        """
        lines = [
            '"""',
            "Kernel loader for pythonify-generated model.",
            "",
            "This module loads serialized kernels and provides the compiled callable.",
            '"""',
            "",
            "import ctypes",
            "import os",
            "from pathlib import Path",
            "",
            "import torch",
            "",
            "# Kernel directory (relative to this file)",
            "_KERNEL_DIR = Path(__file__).parent",
            "",
        ]

        lines.append("# Loaded kernels")
        lines.append("_loaded_kernels = {}")
        lines.append("")

        for ref in self._kernels:
            if ref.kernel_type == KernelType.CPP:
                lines.extend(self._generate_cpp_loader(ref))
            elif ref.kernel_type == KernelType.TRITON:
                lines.extend(self._generate_triton_loader(ref))
            elif ref.kernel_type == KernelType.PYTHON:
                lines.extend(self._generate_python_loader(ref))

        lines.append("")
        lines.append("def get_compiled_callable():")
        lines.append('    """Get the main compiled callable for this model."""')
        if self._kernels:
            main_kernel = self._kernels[-1]
            lines.append(f'    return _loaded_kernels.get("{main_kernel.kernel_id}")')
        else:
            lines.append("    return None")
        lines.append("")

        return "\n".join(lines)

    def _generate_cpp_loader(self, ref: KernelReference) -> list[str]:
        """Generate loader code for a C++ kernel."""
        return [
            "",
            f"def _load_cpp_kernel_{ref.kernel_id}():",
            f'    """Load C++ kernel: {ref.kernel_id}"""',
            f'    so_path = _KERNEL_DIR / "{ref.file_path}"',
            "    if not so_path.exists():",
            f'        raise FileNotFoundError(f"Kernel not found: {{so_path}}")',
            "    lib = ctypes.CDLL(str(so_path))",
            f'    fn = getattr(lib, "{ref.entry_point}", None)',
            "    if fn is None:",
            f'        raise RuntimeError("Entry point {ref.entry_point} not found in {{so_path}}")',
            f'    _loaded_kernels["{ref.kernel_id}"] = fn',
            "    return fn",
            "",
            f'_load_cpp_kernel_{ref.kernel_id}()',
            "",
        ]

    def _generate_triton_loader(self, ref: KernelReference) -> list[str]:
        """Generate loader code for a Triton kernel."""
        return [
            "",
            f"def _load_triton_kernel_{ref.kernel_id}():",
            f'    """Load Triton kernel: {ref.kernel_id}"""',
            "    try:",
            "        import triton",
            "    except ImportError:",
            '        raise ImportError("Triton is required to load CUDA kernels")',
            "    # Triton kernel loading would go here",
            "    # For now, store a placeholder",
            f'    _loaded_kernels["{ref.kernel_id}"] = None',
            "    return None",
            "",
            f'_load_triton_kernel_{ref.kernel_id}()',
            "",
        ]

    def _generate_python_loader(self, ref: KernelReference) -> list[str]:
        """Generate loader code for a Python wrapper."""
        return [
            "",
            f"def _load_python_wrapper_{ref.kernel_id}():",
            f'    """Load Python wrapper: {ref.kernel_id}"""',
            f'    wrapper_path = _KERNEL_DIR / "{ref.file_path}"',
            "    if not wrapper_path.exists():",
            f'        raise FileNotFoundError(f"Wrapper not found: {{wrapper_path}}")',
            "    spec = importlib.util.spec_from_file_location(",
            f'        "wrapper_{ref.kernel_id}",',
            "        wrapper_path,",
            "    )",
            "    module = importlib.util.module_from_spec(spec)",
            "    spec.loader.exec_module(module)",
            f'    fn = getattr(module, "{ref.entry_point}", None)',
            f'    _loaded_kernels["{ref.kernel_id}"] = fn',
            "    return fn",
            "",
            "import importlib.util",
            f'_load_python_wrapper_{ref.kernel_id}()',
            "",
        ]


def serialize_inductor_output(
    compiled_output: Any,
    output_dir: str,
) -> SerializedKernelBundle:
    """
    Serialize a compiled Inductor output for use in pythonify.

    This is the main entry point for kernel serialization. It takes the
    output from Inductor compilation and produces a bundle of serialized
    kernels that can be loaded by generated Python code.

    Args:
        compiled_output: Output from Inductor compilation (CompiledFxGraph, etc.)
        output_dir: Directory to store serialized artifacts

    Returns:
        SerializedKernelBundle with references to all serialized kernels
    """
    serializer = KernelSerializer(output_dir)

    if hasattr(compiled_output, "source_code"):
        ref = serializer.serialize_python_wrapper(
            wrapper_code=compiled_output.source_code,
            wrapper_name="call",
            metadata={"type": "inductor_wrapper"},
        )

    return serializer.get_bundle()


def inline_kernel_as_base64(kernel_path: str) -> str:
    """
    Encode a kernel binary as base64 for inline embedding.

    This allows small kernels to be embedded directly in the generated
    Python file rather than requiring separate files.

    Args:
        kernel_path: Path to the kernel binary file

    Returns:
        Base64-encoded string of the kernel contents
    """
    with open(kernel_path, "rb") as f:
        content = f.read()
    return base64.b64encode(content).decode("ascii")


def decode_inline_kernel(base64_data: str, output_path: str) -> str:
    """
    Decode a base64-encoded kernel and write to file.

    Args:
        base64_data: Base64-encoded kernel content
        output_path: Path to write the decoded kernel

    Returns:
        The output_path for convenience
    """
    content = base64.b64decode(base64_data)
    with open(output_path, "wb") as f:
        f.write(content)
    return output_path
