"""Inline PTX support for the CuteDSL backend.

Inductor's `inline_asm_elementwise` HOP is how callers reach instructions the
compiler will not emit on its own, such as TorchAO's
`cvt.rp.satfinite.ue8m0x2.f32` MX scale conversion. Triton lowers the HOP to
`tl.inline_asm_elementwise`; this module provides the CuteDSL equivalent by
emitting one `llvm.inline_asm` per element and rebuilding the fragment.

The asm string and constraint list are the caller's contract, exactly as in the
Triton backend. Values are bitcast, never converted, into the register class each
constraint letter names, so a Float32 input under an `r` constraint arrives as
its bit pattern rather than a rounded integer.
"""

import functools
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import dsl_user_op, T


@functools.cache
def inline_asm_cache_key() -> str:
    """Include this module's source in generated kernel cache keys."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# PTX register classes, as accepted by the NVPTX inline-asm backend.
CONSTRAINT_TYPES = {
    "h": T.i16,
    "r": T.i32,
    "l": T.i64,
    "f": T.f32,
    "d": T.f64,
}


def split_constraints(constraints: str) -> tuple[list[str], list[str]]:
    """Split an LLVM constraint list into output and input register classes."""
    outputs, inputs = [], []
    for entry in constraints.split(","):
        entry = entry.strip()
        target = outputs if entry.startswith("=") else inputs
        letter = entry.lstrip("=&")
        if letter not in CONSTRAINT_TYPES:
            raise NotImplementedError(
                f"CuteDSL inline asm supports constraints {sorted(CONSTRAINT_TYPES)}, "
                f"got {entry!r}"
            )
        target.append(letter)
    if len(outputs) != 1:
        raise NotImplementedError(
            f"CuteDSL inline asm requires exactly one output constraint, got {constraints!r}"
        )
    return outputs, inputs


def bitcast_to(value: ir.Value, target: ir.Type) -> ir.Value:
    """Reinterpret a value's bits as an equally wide type."""
    if value.type == target:
        return value
    return llvm.bitcast(target, value)


def fragment_elements(source, index: int) -> ir.Value:
    """Extract element `index` from a TensorSSA, or the value itself if scalar."""
    if isinstance(source, cute.TensorSSA):
        return vector.extract(source.ir_value(), [], [index])
    return cutlass.Float32(source).ir_value() if isinstance(source, float) else source


@dsl_user_op
def inline_asm_elementwise_intrinsic(
    *sources,
    asm: str,
    constraints: str,
    result_type,
    is_pure: bool = True,
    pack: int = 1,
    loc=None,
    ip=None,
):
    """Apply an inline PTX block elementwise across a TensorSSA fragment.

    Args:
        sources: TensorSSA fragments or scalars, all of the same length.
        asm: PTX text using `$N` operand syntax, output first.
        constraints: LLVM constraint list, e.g. `"=h,r"`.
        result_type: cutlass numeric type of the produced fragment.
        is_pure: Whether the block may be reordered and CSEd.
        pack: Elements consumed per asm invocation. Only 1 is supported.
    """
    if pack != 1:
        raise NotImplementedError(
            f"CuteDSL inline asm supports pack=1, got pack={pack}. Use a named "
            "intrinsic for packed conversions."
        )
    output_letters, input_letters = split_constraints(constraints)
    if len(input_letters) != len(sources):
        raise ValueError(
            f"inline asm expects {len(input_letters)} inputs for constraints "
            f"{constraints!r}, got {len(sources)}"
        )

    output_type = CONSTRAINT_TYPES[output_letters[0]]()
    fragments = [s for s in sources if isinstance(s, cute.TensorSSA)]
    if not fragments:
        raise ValueError("inline asm requires at least one TensorSSA input")
    shape = fragments[0].shape
    count = ir.VectorType(fragments[0].ir_value().type).shape[0]

    converted = []
    for index in range(count):
        operands = [
            bitcast_to(fragment_elements(source, index), CONSTRAINT_TYPES[letter]())
            for source, letter in zip(sources, input_letters)
        ]
        produced = llvm.inline_asm(
            output_type,
            operands,
            asm,
            constraints,
            has_side_effects=not is_pure,
            is_align_stack=False,
        )
        converted.append(bitcast_to(produced, result_type.mlir_type))

    vector_type = ir.VectorType.get([count], result_type.mlir_type)
    return cute.TensorSSA(
        vector.from_elements(vector_type, converted), shape, result_type
    )
