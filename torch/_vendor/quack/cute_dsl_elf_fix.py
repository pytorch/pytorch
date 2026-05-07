# Monkey patch for NVIDIA/cutlass#3161 [CuTeDSL] workaround of MLIR codegen
# emitting duplicate .text sections with conflicting flags.
#
# The CUTLASS MLIR code-gen produces .o files with two .text sections:
# one executable (ALLOC|EXECINSTR) and a small writable-data trampoline
# (WRITE|ALLOC). LLVM's MCJIT backend merges same-named sections and
# misapplies page protections, causing non-deterministic segfaults in
# multi-process torchrun workloads. LLVM JitLink rejects the file outright.
#
# Fix: before constructing the execution engine, rewrite all duplicate
# .text sections to ALLOC|EXECINSTR so JitLink accepts the file, then
# force useJitLink=True for that path. Relocations are applied before
# page protection, so the trampoline does not need WRITE at load time.
#
# Drop this file once the upstream PR lands in the CuTeDSL release quack pins.

import struct

_PATCHED = False


def _fix_elf_dup_text_flags(data: bytes) -> bytes:
    if len(data) < 64 or data[4] != 2 or data[5] != 1:  # ELF64 LE only
        return data
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]
    if not e_shoff or not e_shnum or e_shstrndx >= e_shnum:
        return data
    shstr_hdr = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from("<Q", data, shstr_hdr + 24)[0]
    text_secs: list[tuple[int, int]] = []
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        ni = struct.unpack_from("<I", data, sh)[0]
        ns = shstr_off + ni
        if ns + 6 <= len(data) and data[ns : ns + 6] == b".text\x00":
            text_secs.append((i, sh))
    if len(text_secs) <= 1:
        return data
    result = bytearray(data)
    for _, sh in text_secs[1:]:
        struct.pack_into("<Q", result, sh + 8, 0x6)  # SHF_ALLOC | SHF_EXECINSTR
    return bytes(result)


def patch() -> None:
    """Install the ELF duplicate-.text flags workaround. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from cutlass.base_dsl.export import external_binary_module as _ebm
        from cutlass.base_dsl.common import DSLRuntimeError
    except Exception:
        return  # CuTeDSL not importable; nothing to patch

    cls = _ebm.ExternalBinaryModule
    orig_init = cls.__init__

    def patched_init(self, file_path: str, enable_tvm_ffi: bool = False) -> None:
        # Mirror of upstream __init__ (cutlass.base_dsl.export.external_binary_module)
        # with the PR #3161 fix inserted between useJitLink selection and
        # engine construction.
        self.enable_tvm_ffi = enable_tvm_ffi
        assert self.load_provider is not None, "Load provider is not set for ExternalBinaryModule."
        shared_libs = self.load_provider.dsl._get_dsl().get_shared_libs()
        object_file_content = bytes()
        if file_path.endswith(".so"):
            shared_libs.append(file_path)
        else:
            try:
                with open(file_path, "rb") as f:
                    object_file_content = f.read()
            except Exception as e:
                raise DSLRuntimeError(f"Failed to read object file {file_path}: {e}")

        useJitLink = not enable_tvm_ffi
        if not useJitLink and object_file_content:
            object_file_content = _fix_elf_dup_text_flags(object_file_content)
            useJitLink = True
        self.engine = self.load_provider.execution_engine_constructor(
            object_file_content, shared_libs, useJitLink
        )

    # Fall back to the original __init__ if any check here changes upstream.
    patched_init.__wrapped__ = orig_init  # type: ignore[attr-defined]
    cls.__init__ = patched_init
    _PATCHED = True
