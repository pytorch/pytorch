"""Compress native AOT cubins through CUDA's in-memory fatbinary API."""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any


_FATBIN_MAGIC = 0xBA55ED50
_FATBIN_HEADER = struct.Struct("<IHHQ")
_FATBIN_ENTRY = struct.Struct("<HHIQ")
_FATBIN_ELF = 2
log = logging.getLogger(__name__)


def compress_cubin(cubin: bytes, arch: str) -> bytes:
    """Keep the cubin unchanged when a compressed CUDA image would be larger."""
    from cuda.bindings import nvfatbin
    from cuda.pathfinder import DynamicLibNotFoundError

    try:
        handle = nvfatbin.create(["-64", "-compress-all"], 2)
    except DynamicLibNotFoundError as error:
        raise RuntimeError(
            "native-AOT compression requires libnvfatbin from CUDA Toolkit 13; "
            "check CUDA library paths or set TORCH_NATIVE_AOT=0 to disable native AOT"
        ) from error
    try:
        nvfatbin.add_cubin(handle, cubin, len(cubin), arch.removeprefix("sm_"), "")
        compressed = bytearray(nvfatbin.size(handle))
        nvfatbin.get(handle, compressed)
        return bytes(compressed) if len(compressed) < len(cubin) else cubin
    finally:
        nvfatbin.destroy(handle)


def compress_cute_embedded_cubin(compiled: Any, arch: str | None = None) -> None:
    """Compress the captured cubin for an explicit arch; retain other images."""
    if arch is None:
        return
    from cutlass._mlir import ir

    cubin = compiled.__cubin__
    if isinstance(cubin, str):
        cubin = Path(cubin).read_bytes()
    if not isinstance(cubin, bytes):
        raise RuntimeError("CuTeDSL did not capture its cubin for native AOT")
    module = compiled.ir_module
    matched = False
    with module.context:
        for op in module.body.operations:
            if op.operation.name != "llvm.mlir.global" or "value" not in op.attributes:
                continue
            value = op.attributes["value"]
            if not isinstance(value, ir.StringAttr):
                continue
            data = ir.StringAttr(value).value_bytes
            if len(data) < _FATBIN_HEADER.size:
                continue
            magic, _, header, _ = _FATBIN_HEADER.unpack_from(data)
            if (
                magic != _FATBIN_MAGIC
                or header < _FATBIN_HEADER.size
                or header + _FATBIN_ENTRY.size > len(data)
            ):
                continue
            kind, _, entry_header, size = _FATBIN_ENTRY.unpack_from(data, header)
            payload = header + entry_header
            # Replace only the captured ELF's single-image wrapper.
            if (
                kind != _FATBIN_ELF
                or entry_header < _FATBIN_ENTRY.size
                or payload + size != len(data)
                or data[payload:] != cubin
            ):
                continue
            matched = True
            compressed = compress_cubin(cubin, arch)
            if len(compressed) >= len(cubin):
                continue
            old_type = op.attributes["global_type"]
            symbol = ir.StringAttr(op.attributes["sym_name"]).value
            new_type = ir.TypeAttr.get(
                ir.Type.parse(f"!llvm.array<{len(compressed)} x i8>")
            )
            op.attributes["value"] = ir.StringAttr.get(compressed)
            op.attributes["global_type"] = new_type

            def update_gep_element_type(operation: Any) -> Any:
                if (
                    operation.operation.name == "llvm.getelementptr"
                    and "elem_type" in operation.attributes
                    and operation.attributes["elem_type"] == old_type
                ):
                    owner = operation.operands[0].owner
                    if (
                        owner.operation.name == "llvm.mlir.addressof"
                        and ir.FlatSymbolRefAttr(owner.attributes["global_name"]).value
                        == symbol
                    ):
                        operation.attributes["elem_type"] = new_type
                return ir.WalkResult.ADVANCE

            module.operation.walk(update_gep_element_type)
    if not matched:
        log.debug("No supported CuTeDSL cubin buffer; retaining original images")
