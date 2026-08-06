"""Inline PTX support for the CuteDSL backend.

Inductor's `inline_asm_elementwise` HOP is how callers reach instructions the
compiler will not emit on its own, such as TorchAO's
`cvt.rp.satfinite.ue8m0x2.f32` MX scale conversion. Triton lowers the HOP to
`tl.inline_asm_elementwise`; this module provides the CuteDSL equivalent by
emitting one `llvm.inline_asm` per element and rebuilding the fragment.

The asm string and constraint list are the caller's contract, exactly as in the
Triton backend. Inputs are bitcast into the register class each constraint letter
names, so a Float32 input under an `r` constraint arrives as its bit pattern.
Wider integer outputs are truncated before being bitcast to the requested type.
E8M0 bit patterns are decoded exactly to Float32 for fused consumers while the
HOP's logical dtype still controls eventual storage.
"""

import functools
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
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
    return arith.bitcast(target, value)


def narrow_integer(value: ir.Value, width: int) -> ir.Value:
    """Truncate an integer register to `width` bits."""
    if not ir.IntegerType.isinstance(value.type):
        raise TypeError(f"expected an integer inline-asm result, got {value.type}")
    if ir.IntegerType(value.type).width > width:
        return arith.trunci(ir.IntegerType.get_signless(width), value)
    return value


def decode_e8m0(value: ir.Value) -> ir.Value:
    """Decode one biased E8M0 bit pattern to its exact Float32 value."""
    code = arith.extui(T.i32(), narrow_integer(value, 8))
    bits = arith.shli(code, arith.constant(T.i32(), 23))
    is_zero = arith.cmpi(arith.CmpIPredicate.eq, code, arith.constant(T.i32(), 0))
    bits = arith.select(is_zero, arith.constant(T.i32(), 0x00400000), bits)
    is_nan = arith.cmpi(arith.CmpIPredicate.eq, code, arith.constant(T.i32(), 255))
    bits = arith.select(is_nan, arith.constant(T.i32(), 0x7FC00000), bits)
    return arith.bitcast(T.f32(), bits)


def convert_output(value: ir.Value, result_type) -> ir.Value:
    """Convert a register result to the requested narrower element type."""
    target = result_type.mlir_type
    if value.type == target:
        return value
    if ir.IntegerType.isinstance(value.type):
        value = narrow_integer(value, result_type.width)
    return bitcast_to(value, target)


def fragment_element(source, index: int) -> ir.Value:
    """Extract element `index` from a TensorSSA, or unwrap a scalar value."""
    if isinstance(source, cute.TensorSSA):
        return vector.extract(source.ir_value(), [], [index])
    if isinstance(source, ir.Value):
        return source
    if isinstance(source, (int, float)):
        return cutlass.Float32(source).ir_value()
    return source.ir_value()


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
        result_type: Logical cutlass numeric type of the produced fragment.
            E8M0 integer results are decoded to Float32 for fused consumers.
        is_pure: Whether the block may be reordered and CSEd.
        pack: Elements consumed per asm invocation. Only 1 is supported.
    """
    if pack != 1:
        raise NotImplementedError(
            f"CuteDSL inline asm supports pack=1, got pack={pack}"
        )
    output_letters, input_letters = split_constraints(constraints)
    if len(input_letters) != len(sources):
        raise ValueError(
            f"inline asm expects {len(input_letters)} inputs for constraints "
            f"{constraints!r}, got {len(sources)}"
        )

    output_type = CONSTRAINT_TYPES[output_letters[0]]()
    compute_type = (
        cutlass.Float32 if result_type == cutlass.Float8E8M0FNU else result_type
    )

    def invoke(index: int) -> ir.Value:
        operands = [
            bitcast_to(fragment_element(source, index), CONSTRAINT_TYPES[letter]())
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
        return (
            decode_e8m0(produced)
            if result_type == cutlass.Float8E8M0FNU
            else convert_output(produced, result_type)
        )

    fragments = [source for source in sources if isinstance(source, cute.TensorSSA)]
    if not fragments:
        return compute_type(invoke(0))

    shape = fragments[0].shape
    count = ir.VectorType(fragments[0].ir_value().type).shape[0]
    converted = [invoke(index) for index in range(count)]
    vector_type = ir.VectorType.get([count], compute_type.mlir_type)
    return cute.TensorSSA(
        vector.from_elements(vector_type, converted), shape, compute_type
    )
