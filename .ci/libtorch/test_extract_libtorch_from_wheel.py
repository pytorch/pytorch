#!/usr/bin/env python3
"""Tests for extract_libtorch_from_wheel.py, focusing on the ELF RPATH rewriter."""

import struct
import tempfile
import unittest
from pathlib import Path

from extract_libtorch_from_wheel import _rewrite_elf_rpath, fix_rpath


def _build_elf64_le(rpath: str, use_runpath: bool = False) -> bytes:
    """Build a minimal ELF64 little-endian binary with the given RPATH.

    Layout:
      [0x000] ELF header  (64 bytes)
      [0x040] PHDR 0: PT_LOAD covering the whole file (56 bytes)
      [0x078] PHDR 1: PT_DYNAMIC pointing to .dynamic  (56 bytes)
      [0x0B0] .dynamic section (3 entries x 16 bytes = 48 bytes)
      [0x0E0] .dynstr section (NUL + rpath string + NUL)
    """
    endian = "<"

    # String table: \0 + rpath + \0
    rpath_offset_in_strtab = 1  # skip leading NUL
    dynstr = b"\x00" + rpath.encode("utf-8") + b"\x00"

    # File offsets
    ehdr_off = 0
    ehdr_size = 64
    phdr_off = ehdr_size  # 0x40
    phdr_size = 56
    phdr_count = 2
    dyn_off = phdr_off + phdr_size * phdr_count  # 0xB0
    dyn_entry_size = 16
    dyn_count = 3  # DT_STRTAB, DT_RPATH/DT_RUNPATH, DT_NULL
    dyn_section_size = dyn_entry_size * dyn_count
    dynstr_off = dyn_off + dyn_section_size  # 0xE0
    total_size = dynstr_off + len(dynstr)

    # Virtual address base (arbitrary, typical for shared libs)
    vaddr_base = 0x0

    # ELF header
    e_ident = b"\x7fELF"  # magic
    e_ident += b"\x02"  # 64-bit
    e_ident += b"\x01"  # little-endian
    e_ident += b"\x01"  # ELF version
    e_ident += b"\x00" * 9  # padding
    ehdr = e_ident + struct.pack(
        f"{endian}HHI QQQ I HHHHHH",
        3,  # e_type: ET_DYN (shared object)
        0x3E,  # e_machine: EM_X86_64
        1,  # e_version
        0,  # e_entry
        phdr_off,  # e_phoff
        0,  # e_shoff (no section headers)
        0,  # e_flags
        ehdr_size,  # e_ehsize
        phdr_size,  # e_phentsize
        phdr_count,  # e_phnum
        0,  # e_shentsize
        0,  # e_shnum
        0,  # e_shstrndx
    )

    # PHDR 0: PT_LOAD covering the whole file
    phdr_load = struct.pack(
        f"{endian}II QQQQQQ",
        1,  # p_type: PT_LOAD
        5,  # p_flags: PF_R | PF_X
        0,  # p_offset
        vaddr_base,  # p_vaddr
        vaddr_base,  # p_paddr
        total_size,  # p_filesz
        total_size,  # p_memsz
        0x1000,  # p_align
    )

    # PHDR 1: PT_DYNAMIC
    phdr_dyn = struct.pack(
        f"{endian}II QQQQQQ",
        2,  # p_type: PT_DYNAMIC
        6,  # p_flags: PF_R | PF_W
        dyn_off,  # p_offset
        vaddr_base + dyn_off,  # p_vaddr
        vaddr_base + dyn_off,  # p_paddr
        dyn_section_size,  # p_filesz
        dyn_section_size,  # p_memsz
        8,  # p_align
    )

    # .dynamic entries
    DT_NULL = 0
    DT_STRTAB = 5
    DT_RPATH = 15
    DT_RUNPATH = 29

    rpath_tag = DT_RUNPATH if use_runpath else DT_RPATH
    dyn = b""
    dyn += struct.pack(f"{endian}qQ", DT_STRTAB, vaddr_base + dynstr_off)
    dyn += struct.pack(f"{endian}qQ", rpath_tag, rpath_offset_in_strtab)
    dyn += struct.pack(f"{endian}qQ", DT_NULL, 0)

    return ehdr + phdr_load + phdr_dyn + dyn + dynstr


def _read_rpath_from_elf(data: bytes) -> str:
    """Read the RPATH/RUNPATH string from a synthetic ELF built by _build_elf64_le."""
    endian = "<"
    # Parse enough to find the string
    phdr_off = struct.unpack_from(f"{endian}Q", data, 32)[0]
    phdr_size = 56

    # Find PT_DYNAMIC
    for i in range(2):
        off = phdr_off + i * phdr_size
        p_type = struct.unpack_from(f"{endian}I", data, off)[0]
        if p_type == 2:
            dyn_off = struct.unpack_from(f"{endian}Q", data, off + 8)[0]
            dyn_filesz = struct.unpack_from(f"{endian}Q", data, off + 32)[0]
            break

    # Find DT_STRTAB and DT_RPATH/DT_RUNPATH
    strtab_addr = 0
    rpath_str_offset = 0
    for i in range(dyn_filesz // 16):
        tag, val = struct.unpack_from(f"{endian}qQ", data, dyn_off + i * 16)
        if tag == 5:
            strtab_addr = val
        elif tag in (15, 29):
            rpath_str_offset = val

    # Find PT_LOAD to convert vaddr to file offset
    for i in range(2):
        off = phdr_off + i * phdr_size
        p_type = struct.unpack_from(f"{endian}I", data, off)[0]
        if p_type == 1:
            p_offset = struct.unpack_from(f"{endian}Q", data, off + 8)[0]
            p_vaddr = struct.unpack_from(f"{endian}Q", data, off + 16)[0]
            p_filesz = struct.unpack_from(f"{endian}Q", data, off + 32)[0]
            if p_vaddr <= strtab_addr < p_vaddr + p_filesz:
                strtab_file_off = p_offset + (strtab_addr - p_vaddr)
                break

    str_pos = strtab_file_off + rpath_str_offset
    end = data.index(b"\x00", str_pos)
    return data[str_pos:end].decode("utf-8")


class TestRewriteElfRpath(unittest.TestCase):
    def _write_elf(self, rpath: str, **kwargs) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".so", delete=False)
        f.write(_build_elf64_le(rpath, **kwargs))
        f.close()
        return Path(f.name)

    def test_rewrite_rpath_basic(self):
        """Rewriting a long RPATH to $ORIGIN should succeed."""
        old_rpath = "$ORIGIN/../../nvidia/nvshmem/lib:$ORIGIN/../../nvidia/cudnn/lib"
        path = self._write_elf(old_rpath)
        try:
            result = _rewrite_elf_rpath(path, "$ORIGIN")
            self.assertTrue(result)
            data = path.read_bytes()
            new_rpath = _read_rpath_from_elf(data)
            self.assertEqual(new_rpath, "$ORIGIN")
        finally:
            path.unlink()

    def test_rewrite_runpath(self):
        """Should also handle DT_RUNPATH (not just DT_RPATH)."""
        old_rpath = "$ORIGIN/../../nvidia/nvshmem/lib"
        path = self._write_elf(old_rpath, use_runpath=True)
        try:
            result = _rewrite_elf_rpath(path, "$ORIGIN")
            self.assertTrue(result)
            data = path.read_bytes()
            new_rpath = _read_rpath_from_elf(data)
            self.assertEqual(new_rpath, "$ORIGIN")
        finally:
            path.unlink()

    def test_nul_padding(self):
        """New RPATH should be NUL-padded to original length."""
        old_rpath = "$ORIGIN/../../nvidia/nvshmem/lib"
        old_len = len(old_rpath)
        path = self._write_elf(old_rpath)
        try:
            _rewrite_elf_rpath(path, "$ORIGIN")
            data = path.read_bytes()
            # Find the rpath string location and check padding
            rpath_start = data.find(b"$ORIGIN")
            self.assertNotEqual(rpath_start, -1)
            # After "$ORIGIN" there should be NULs filling the rest
            region = data[rpath_start : rpath_start + old_len]
            self.assertEqual(region, b"$ORIGIN" + b"\x00" * (old_len - len("$ORIGIN")))
        finally:
            path.unlink()

    def test_non_elf_returns_false(self):
        """Non-ELF files should return False without error."""
        f = tempfile.NamedTemporaryFile(suffix=".so", delete=False)
        f.write(b"not an ELF file at all")
        f.close()
        path = Path(f.name)
        try:
            result = _rewrite_elf_rpath(path, "$ORIGIN")
            self.assertFalse(result)
        finally:
            path.unlink()

    def test_no_rpath_returns_false(self):
        """ELF without RPATH/RUNPATH should return False."""
        # Build an ELF and zero out the rpath tag to DT_NULL
        data = bytearray(_build_elf64_le("$ORIGIN"))
        # Find and zero the DT_RPATH entry (second dynamic entry)
        phdr_off = struct.unpack_from("<Q", data, 32)[0]
        # PT_DYNAMIC is the second phdr
        dyn_off = struct.unpack_from("<Q", data, phdr_off + 56 + 8)[0]
        # Second dynamic entry is the rpath; set tag to DT_NULL (0)
        struct.pack_into("<qQ", data, dyn_off + 16, 0, 0)

        f = tempfile.NamedTemporaryFile(suffix=".so", delete=False)
        f.write(data)
        f.close()
        path = Path(f.name)
        try:
            result = _rewrite_elf_rpath(path, "$ORIGIN")
            self.assertFalse(result)
        finally:
            path.unlink()

    def test_truncation_when_new_rpath_too_long(self):
        """If new RPATH is longer than old, it should be truncated."""
        old_rpath = "ab"
        path = self._write_elf(old_rpath)
        try:
            result = _rewrite_elf_rpath(path, "$ORIGIN")
            self.assertTrue(result)
            data = path.read_bytes()
            new_rpath = _read_rpath_from_elf(data)
            # Truncated to 2 chars
            self.assertEqual(new_rpath, "$O")
        finally:
            path.unlink()


class TestFixRpath(unittest.TestCase):
    def test_skips_non_linux(self):
        """fix_rpath should be a no-op on non-linux platforms."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "libfoo.so"
            old_rpath = "$ORIGIN/../../nvidia/nvshmem/lib"
            p.write_bytes(_build_elf64_le(old_rpath))
            fix_rpath(Path(d), "macos")
            data = p.read_bytes()
            self.assertEqual(_read_rpath_from_elf(data), old_rpath)

    def test_fixes_so_files(self):
        """fix_rpath should rewrite RPATH on .so files."""
        with tempfile.TemporaryDirectory() as d:
            old_rpath = "$ORIGIN/../../nvidia/nvshmem/lib"
            so = Path(d) / "libtorch_nvshmem.so"
            so.write_bytes(_build_elf64_le(old_rpath))
            txt = Path(d) / "readme.txt"
            txt.write_text("not a library")

            fix_rpath(Path(d), "linux")

            self.assertEqual(_read_rpath_from_elf(so.read_bytes()), "$ORIGIN")
            self.assertEqual(txt.read_text(), "not a library")

    def test_skips_non_so_files(self):
        """fix_rpath should not touch files without .so in the name."""
        with tempfile.TemporaryDirectory() as d:
            old_rpath = "$ORIGIN/../../nvidia/nvshmem/lib"
            lib = Path(d) / "libfoo.a"
            lib.write_bytes(_build_elf64_le(old_rpath))
            fix_rpath(Path(d), "linux")
            self.assertEqual(_read_rpath_from_elf(lib.read_bytes()), old_rpath)


if __name__ == "__main__":
    unittest.main()
