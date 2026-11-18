"""Compress native AOT cubins through CUDA's in-memory fatbinary API."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def compress_cuda_image(cubin: bytes, arch: str) -> bytes:
    """Keep the cubin unchanged when a compressed CUDA image would be larger."""
    from cuda.bindings import nvfatbin

    handle = nvfatbin.create(["-64", "-compress-all"], 2)
    try:
        nvfatbin.add_cubin(handle, cubin, len(cubin), arch.removeprefix("sm_"), "")
        compressed = bytearray(nvfatbin.size(handle))
        nvfatbin.get(handle, compressed)
        return bytes(compressed) if len(compressed) < len(cubin) else cubin
    finally:
        nvfatbin.destroy(handle)


def compress_cute_module(compiled: Any, arch: str | None = None) -> None:
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
    with module.context:
        for op in module.body.operations:
            if op.operation.name != "llvm.mlir.global" or "value" not in op.attributes:
                continue
            value = op.attributes["value"]
            if not isinstance(value, ir.StringAttr):
                continue
            data = ir.StringAttr(value).value_bytes
            if len(data) < 32 or not data.startswith(struct.pack("<I", 0xBA55ED50)):
                continue
            kind, _, header, size = struct.unpack_from("<HHIQ", data, 16)
            # Replace only the captured ELF's single-image wrapper.
            if (
                kind != 2
                or 16 + header + size != len(data)
                or data[16 + header :] != cubin
            ):
                continue
            compressed = compress_cuda_image(cubin, arch)
            if len(compressed) >= len(data):
                continue
            old_type = op.attributes["global_type"]
            symbol = ir.StringAttr(op.attributes["sym_name"]).value
            new_type = ir.TypeAttr.get(
                ir.Type.parse(f"!llvm.array<{len(compressed)} x i8>")
            )
            op.attributes["value"] = ir.StringAttr.get(compressed)
            op.attributes["global_type"] = new_type

            def update_gep(operation: Any) -> Any:
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

            module.operation.walk(update_gep)
