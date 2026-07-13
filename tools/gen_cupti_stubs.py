"""Generate ``torch/profiler/_cupti/_cupti_stubs.py`` from the CUPTI ABI.

The v2 / user-defined-record CUPTI path selects activity records by *field id*
(``CUpti_Activity*FieldIds``) and is configured via *attribute* selectors
(``CUpti_ActivityAttribute``). cupti-python exposes neither enum, so the monitor
previously hard-coded the integer ids/attrs by hand. This script parses them straight
out of ``cupti_activity.h`` (shipped by the ``nvidia-cuda-cupti`` build dependency) and
emits a Python module of the same constants -- the per-kind ``Field`` catalogs plus
``ActivityAttr`` -- so they can never drift from the header. ``records.py`` curates
*which* fields the monitor selects; this module is only the ABI source of truth.

Parsing uses libclang (the ``clang`` python bindings) so the C frontend -- not a
fragile regex -- computes every enumerator value. The ``libclang`` wheel (a build
dependency) bundles ``libclang.so`` and cindex loads it automatically; ``LIBCLANG_PATH``
can override with a specific one. Field string-ness (which documented field is
``const char*``) is read from each enumerator's doc comment, the header's only record
of the C type.

Run standalone for debugging:

    python tools/gen_cupti_stubs.py --output torch/profiler/_cupti/_cupti_stubs.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


_FIELDIDS_PREFIX = "CUpti_Activity"
_FIELDIDS_SUFFIX = "FieldIds"
# The activity-attribute enum (cuptiActivitySetAttribute_v2 selectors: UDR, kernel-latency
# timestamps, the timestamp callback, ...). Emitted as ActivityAttr so cupti_python need not
# hardcode the (cupti-python-renumbered) ints.
_ATTR_ENUM = "CUpti_ActivityAttribute"
_ATTR_PREFIX = "CUPTI_ACTIVITY_ATTR_"
# A field's doc comment opens with its C declaration (``<type> <name>;``), the only place
# the C type is recorded: the record structs can't supply it (the enum->struct name mapping
# is irregular -- e.g. Memcpy2FieldIds -> CUpti_ActivityMemcpyPtoP4 -- and positional mapping
# breaks on the structs' pad/reserved members). We read the leading type token to pick a
# Ctype (decode interpretation); the byte width comes from CUPTI's captured layout, not here.
_CHAR_PTR_DECL_RE = re.compile(r"(?:const\s+)?char\s*\*")
# Trailing count sentinel each field enum ends with (``*_FIELD_MAX``); not a real field.
_SENTINEL_SUFFIXES = ("MAX", "FORCE_INT")
# CUPTI_API_VERSION lives in the sibling cupti_version.h; stamped into the generated
# module (with the header's sha256) so its provenance -- which CUPTI ABI it came from
# -- is self-evident.
_CUPTI_API_VERSION_RE = re.compile(
    r"^\s*#\s*define\s+CUPTI_API_VERSION\s+(\d+)", re.MULTILINE
)


def _field_ctype(raw_comment: str | None) -> str:
    """The ``Ctype`` member name for a field, from its doc-comment declaration line.
    ``const char*`` -> CSTR, ``float``/``double`` -> FLOAT, signed ints -> INT, everything
    else (unsigned ints, and enum/struct typedefs whose width the runtime layout supplies)
    -> UINT. Struct fields need nothing more: the decoder skips non-1/2/4/8 sizes."""
    for line in (raw_comment or "").splitlines():
        decl = line.lstrip(" */")
        if not decl.endswith(";"):  # only the leading declaration line
            continue
        if _CHAR_PTR_DECL_RE.match(decl):
            return "CSTR"
        tok = decl.split()[0]
        if tok in ("float", "double"):
            return "FLOAT"
        if tok.startswith(("uint", "unsigned", "size_t")):
            return "UINT"
        if tok.startswith("int"):  # int, int8_t .. int64_t
            return "INT"
        return "UINT"
    return "UINT"


@dataclass(frozen=True)
class _FieldDef:
    name: str  # attribute name, e.g. "REGISTERS_PER_THREAD"
    value: int
    ctype: str  # Ctype member name: "UINT" | "INT" | "FLOAT" | "CSTR"


def _load_cindex():  # type: ignore[no-untyped-def]
    """Import ``clang.cindex`` and ensure a ``libclang.so`` is loadable. The ``libclang`` wheel
    (a build dependency) bundles ``libclang.so`` and cindex loads it automatically; set
    ``LIBCLANG_PATH`` to override with a specific one."""
    try:
        # pyrefly: ignore [missing-import]
        import clang.cindex as cindex
    except ImportError as e:
        raise SystemExit(
            "the CUPTI field-id codegen requires the clang python bindings; install the "
            "'libclang' wheel into the build environment"
        ) from e

    explicit = os.environ.get("LIBCLANG_PATH")
    if explicit and Path(explicit).exists():
        cindex.Config.set_library_file(explicit)
    try:
        cindex.Index.create()  # force the load now, so failure is clean (not a mid-parse traceback)
    except cindex.LibclangError as e:
        raise SystemExit(
            "CUPTI stub generation failed: libclang.so could not be loaded. Install the "
            "'libclang' wheel (bundles libclang.so), or set LIBCLANG_PATH to a libclang.so."
        ) from e
    return cindex


def parse_header(
    header: Path,
) -> tuple[dict[str, list[_FieldDef]], list[tuple[str, int]]]:
    """Parse the header via libclang. Returns (``<X>FieldIds`` class name -> its fields, in
    declaration order) and the CUpti_ActivityAttribute enumerators (name, value)."""
    cindex = _load_cindex()
    args = ["-x", "c", f"-I{header.parent}"]
    if cuda_home := (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")):
        args.append(f"-I{Path(cuda_home) / 'include'}")
    # A missing <cuda.h> only yields diagnostics; libclang parses past it and the
    # FieldIds / attribute enums (plain int enums) are recovered regardless.
    tu = cindex.Index.create().parse(
        str(header),
        args=args,
        options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
    )

    fields_by_class: dict[str, list[_FieldDef]] = {}
    attrs: list[tuple[str, int]] = []
    for cur in tu.cursor.walk_preorder():
        if cur.kind != cindex.CursorKind.TYPEDEF_DECL:
            continue
        name = cur.spelling
        if name == _ATTR_ENUM:
            enum = cur.underlying_typedef_type.get_declaration()
            if enum.kind == cindex.CursorKind.ENUM_DECL:
                for c in enum.get_children():
                    if c.kind != cindex.CursorKind.ENUM_CONSTANT_DECL:
                        continue
                    an = c.spelling.removeprefix(_ATTR_PREFIX)
                    if an.endswith(_SENTINEL_SUFFIXES):
                        continue
                    attrs.append((an, c.enum_value))
            continue
        if not (name.startswith(_FIELDIDS_PREFIX) and name.endswith(_FIELDIDS_SUFFIX)):
            continue
        enum = cur.underlying_typedef_type.get_declaration()
        if enum.kind != cindex.CursorKind.ENUM_DECL:
            continue
        fields: list[_FieldDef] = []
        for c in enum.get_children():
            if c.kind != cindex.CursorKind.ENUM_CONSTANT_DECL:
                continue
            attr = c.spelling.split("_FIELD_", 1)[-1]
            if attr in _SENTINEL_SUFFIXES:
                continue
            fields.append(_FieldDef(attr, c.enum_value, _field_ctype(c.raw_comment)))
        if fields:
            # CUpti_ActivityKernelFieldIds -> Kernel (matches the names records.py uses)
            cls_name = name.removeprefix(_FIELDIDS_PREFIX).removesuffix(
                _FIELDIDS_SUFFIX
            )
            fields_by_class[cls_name] = fields

    if not fields_by_class:
        raise SystemExit(f"no CUpti_Activity*FieldIds enums parsed from {header}")
    if not attrs:
        raise SystemExit(f"no {_ATTR_ENUM} enumerators parsed from {header}")
    return fields_by_class, attrs


def render(
    fields_by_class: dict[str, list[_FieldDef]],
    attrs: list[tuple[str, int]],
    header: Path,
) -> str:
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    version_header = header.parent / "cupti_version.h"
    match = (
        _CUPTI_API_VERSION_RE.search(version_header.read_text())
        if version_header.is_file()
        else None
    )
    version = match.group(1) if match else "unknown"
    lines = [
        "# @" + "generated by tools/gen_cupti_stubs.py -- do not edit.",
        f"# Source: {header.name} (CUPTI ABI; nvidia-cuda-cupti build dependency).",
        f"# BUILD_CUPTI_API_VERSION: {version}   Source sha256: {digest}",
        '"""CUPTI ABI enum catalogs, generated from cupti_activity.h.',
        "",
        "One class per activity kind: each attribute is a :class:`Field` (its",
        "``CUpti_Activity*FieldIds`` id plus its :class:`Ctype` for decode); ``records``",
        "curates which of these the monitor selects per kind. ``ActivityAttr`` holds the",
        "``CUpti_ActivityAttribute`` selectors (cuptiActivitySetAttribute_v2).",
        '"""',
        "",
        "from torch.profiler._cupti._records_base import Ctype, Field",
        "",
        "",
    ]
    for cls, fields in fields_by_class.items():
        lines.append(f"class {cls}:")
        for f in fields:
            lines.append(f"    {f.name} = Field({f.value}, Ctype.{f.ctype})")
        lines.append("")
        lines.append("")
    lines.append("class ActivityAttr:")
    lines.append(
        '    """CUpti_ActivityAttribute selectors (cuptiActivitySetAttribute_v2), by name."""'
    )
    for an, val in attrs:
        lines.append(f"    {an} = {val}")
    return "\n".join(lines).rstrip() + "\n"


def generate(output: Path, header: Path) -> None:
    """Write the generated module from ``header`` (already resolved and version-gated by the
    caller via find_cupti_header). ``main`` skips this call entirely when no suitable header
    exists, so non-CUPTI builds proceed without the module."""
    fields_by_class, attrs = parse_header(header)
    content = render(fields_by_class, attrs, header)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text() != content:
        output.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Run as a script, sys.path[0] is tools/, not the repo root -- add the latter so the
    # tools.setup_helpers package (the header resolver) imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.setup_helpers.cupti import find_cupti_header

    header = find_cupti_header()
    if header is None:
        # No sufficiently-new CUPTI header (find_cupti_header applies the version floor).
        # Skip rather than error: the build gates on whether the output file exists. Clear any
        # stale output so that gate reflects reality (the file is generated / gitignored).
        args.output.unlink(missing_ok=True)
        print("no sufficiently-new CUPTI header found; skipping CUPTI stub generation")
        return
    generate(args.output, header)
    print(f"Generated {args.output} from {header}")
    # Machine-readable (own stdout line) so the build can track the resolved header as a
    # configure dependency and regenerate when it changes.
    print(f"CUPTI_MONITOR_STUBS_HEADER={header}")


if __name__ == "__main__":
    main()
