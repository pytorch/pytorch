"""`GEMMDesc`: how to compute a GEMM, written down precisely enough to pin the bits.

Floating-point addition is not associative, so the order a GEMM adds its k products
in decides its bits. `GEMMDesc` writes that order down: two runs on the same
architecture that follow the same `GEMMDesc` owe you the same bits. It carries only
what changes that order -- never a speed knob, never a library name.

A speed knob can still move the bits indirectly: a tile too small for an instruction
makes the compiler fall back to a different one. That runs through `algorithm`,
which is a field, so a producer must check that its launch really runs the
instruction the descriptor names rather than assume it.

Two relations live here and they are not the same. `a == b` says the two
descriptions are identical. `produces_same_bits(a, b, known)` says one machine
cannot show the difference between them, which is weaker and is measured per
architecture.

See gemm_desc.md, beside this file, for the long form: the worked examples, the
measurements these rules came from, and what is deliberately not a field.
"""

from __future__ import annotations

import dataclasses
from enum import auto, Enum

import torch


class GEMMAlgorithm(Enum):
    """How operands become an update to the accumulator.

    The matrix instructions are named one by one, not lumped into a single "tensor
    core" value, because on fp8 they do not agree. AMD's `v_mfma` and `v_wmma` and
    NVIDIA's older `wmma.mma.sync` have no value here -- add one rather than borrow
    a name; see gemm_desc.md.
    """

    # `mma.sync.aligned`: one warp, accumulator in registers. Volta onward.
    MMA_SYNC = auto()

    # `wgmma.mma_async`: one warpgroup, operands from shared memory. Hopper only.
    WGMMA = auto()

    # `tcgen05.mma`: one thread, accumulator in tensor memory. Blackwell onward.
    TCGEN05_MMA = auto()

    # `acc = fma(a, b, acc)`: one rounding per k element.
    SCALAR_FUSED_MULTIPLY_ADD = auto()

    # `acc = acc + (a * b)`: two roundings per k element.
    SCALAR_MULTIPLY_THEN_ADD = auto()


# The algorithms that sum several k elements in one rounding.
MATRIX_INSTRUCTIONS: frozenset[GEMMAlgorithm] = frozenset(
    {
        GEMMAlgorithm.MMA_SYNC,
        GEMMAlgorithm.WGMMA,
        GEMMAlgorithm.TCGEN05_MMA,
    }
)


class InputPrecision(Enum):
    """What the operands are rounded to before they are multiplied.

    fp32 operands can be fed to a matrix instruction in more than one way, and the
    choice changes the result while the dtype stays fp32.
    """

    # Operands multiplied at their own precision; the only choice for non-fp32 ones.
    IEEE = auto()

    # fp32 operands rounded to tf32, a 10-bit mantissa, before the multiply.
    TF32 = auto()

    # Each fp32 operand split into three tf32 pieces and the products summed.
    TF32X3 = auto()


class KCutLayout(Enum):
    """Which k elements one part of a k cut owns.

    Two layouts can hand out parts of the same size and still be different sums,
    because which k elements share an accumulator is what decides the sum.
    """

    # Part i owns the run k in [i * span, (i + 1) * span). The last part may be short.
    CONTIGUOUS = auto()

    # `span`-long tiles dealt out: part i owns every tile j with j % count == i.
    STRIDED = auto()


class MergeOrder(Enum):
    """The order the finished parts of a k cut are added together in.

    The two agree at two parts or fewer, so a producer that has only seen two-part
    cuts has not yet learned which one a kernel does.
    """

    # (((p0 + p1) + p2) + p3), which is what a loop over the parts does.
    SEQUENTIAL = auto()

    # ((p0 + p1) + (p2 + p3)), a balanced tree in index order -- a count-down butterfly
    # is not this one, write it as nested two-part cuts; see gemm_desc.md.
    PAIRWISE_TREE = auto()


class EpilogueOrder(Enum):
    """Where the single rounding to `output_dtype` sits relative to the epilogue.

    A bias added to the accumulator before that rounding and the same bias added
    after it give different bits.
    """

    # Nothing after the k sum: the accumulator is rounded once, on the store.
    NONE = auto()

    # Every epilogue step runs in `accumulate_dtype`, then one rounding on the store.
    IN_ACCUMULATOR = auto()

    # Rounded first, then the epilogue runs in `output_dtype` -- but Inductor rounds in
    # the MIDDLE of a fused epilogue, which is neither value; see gemm_desc.md.
    AFTER_ROUNDING = auto()


# An operand may be read at any of these. An unlisted dtype is rejected, not guessed.
FLOAT_OPERAND_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    }
)
INT_OPERAND_DTYPES: frozenset[torch.dtype] = frozenset({torch.int8, torch.uint8})
OPERAND_DTYPES: frozenset[torch.dtype] = FLOAT_OPERAND_DTYPES | INT_OPERAND_DTYPES

# What a k sum may be kept in. fp8 is absent on purpose: no hardware accumulates in it.
ACCUMULATE_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float16, torch.float32, torch.float64, torch.int32, torch.int64}
)

OUTPUT_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int8,
        torch.int32,
    }
)

# What a partial sum may be stored in, and what a merge may add in.
MERGE_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    }
)


def _dtype_names(dtypes: frozenset[torch.dtype]) -> str:
    return ", ".join(sorted(str(d) for d in dtypes))


@dataclasses.dataclass(frozen=True)
class KCut:
    """One cut of the k axis, not one part of it.

    Slice the k range handed to this level into parts, give each part its own
    accumulator starting at zero, then add the part totals. Cuts nest, outermost
    first; the innermost part has no cut under it and one accumulator walks it in
    index order.
    """

    # k elements in one part (CONTIGUOUS) or in one tile (STRIDED). A length, not a
    # part count -- the count a library is asked for is not the count it performs.
    span: int

    # The dtype a finished part is written at, and the dtype the running merge sum is
    # kept in -- merge_dtype does not follow partial_dtype; see gemm_desc.md.
    partial_dtype: torch.dtype
    merge_dtype: torch.dtype

    layout: KCutLayout = KCutLayout.CONTIGUOUS
    merge: MergeOrder = MergeOrder.SEQUENTIAL

    # How many parts. STRIDED only; for CONTIGUOUS it follows from `span`.
    count: int | None = None

    # How a STRIDED cut labels a tile when the label does not repeat every `count`
    # tiles. Unset means the plain rotation, tile j to part j % count.
    period: int | None = None
    divisor: int = 1

    def __post_init__(self) -> None:
        if self.span < 1:
            raise ValueError(f"span must be at least 1, got {self.span}")
        contiguous = self.layout is KCutLayout.CONTIGUOUS
        if contiguous and self.count is not None:
            raise ValueError(
                "count must be None for a CONTIGUOUS cut: the number of parts "
                f"follows from span and the length being cut, got count={self.count}"
            )
        if not contiguous and (self.count is None or self.count < 2):
            raise ValueError(
                "a STRIDED cut needs count >= 2, since a cut into one part is not a "
                f"cut, got count={self.count}"
            )
        self._check_label()
        for name, dtype in (
            ("partial_dtype", self.partial_dtype),
            ("merge_dtype", self.merge_dtype),
        ):
            if dtype not in MERGE_DTYPES:
                raise ValueError(
                    f"unsupported {name} {dtype}; "
                    f"supported: {_dtype_names(MERGE_DTYPES)}"
                )

    def _check_label(self) -> None:
        contiguous = self.layout is KCutLayout.CONTIGUOUS
        for name, value in (("period", self.period), ("divisor", self.divisor)):
            unset = value is None or (name == "divisor" and value == 1)
            if contiguous and not unset:
                raise ValueError(
                    f"{name} must be unset for a CONTIGUOUS cut: it labels a tile, "
                    f"and a contiguous cut has no tiles, got {name}={value}"
                )
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.period is None and self.divisor != 1:
            raise ValueError(
                "divisor needs a period: it selects a digit of the label, and "
                "without a period there is no label to read, got "
                f"divisor={self.divisor}"
            )


@dataclasses.dataclass(frozen=True)
class GEMMDesc:
    """Everything about a GEMM that decides its bits, and nothing else.

    The k sum is described from the outside in: `k_cuts` cuts the k axis into parts
    that each get their own accumulator, and inside the innermost part one
    accumulator runs a chain of steps in index order. It is frozen and compares by
    its fields, so equal descriptors mean equal bits and it works as a cache key.

    Five algorithms, and four fields that only mean anything for some of them. The
    whole rule is::

        MMA_SYNC                   mma.sync, one warp, Volta onward
        WGMMA                      wgmma, one warpgroup, Hopper only
        TCGEN05_MMA                tcgen05.mma, one thread, Blackwell onward
            instruction_k     required   how many k one instruction folds in
            use_fast_accum    required   where the instruction's result meets acc
            k_loop_step       optional   only when the short turn comes first
            input_precision   IEEE, or tf32 / tf32x3 when the operands are fp32

        SCALAR_FUSED_MULTIPLY_ADD  fma(a, b, acc), one rounding per k element
        SCALAR_MULTIPLY_THEN_ADD   a * b, then the add, two roundings per element
            instruction_k     must be None
            use_fast_accum    must be None
            k_loop_step       must be None
            input_precision   IEEE only

    A scalar algorithm rounds once per k element, so it has no instruction to size,
    no second place to put the add, and no instruction groups for a short first turn
    to shift; tf32 is a matrix instruction's input format. Construction rejects every
    other combination. Everything else applies to all five: the three dtypes,
    `k_cuts`, and `epilogue`.
    """

    # The dtype both operands are read at; two operand dtypes are out of scope.
    operand_dtype: torch.dtype

    # The dtype the k sum is kept in. Not implied by the operand dtype.
    accumulate_dtype: torch.dtype

    # The dtype the finished value is rounded to on the store.
    output_dtype: torch.dtype

    algorithm: GEMMAlgorithm

    input_precision: InputPrecision = InputPrecision.IEEE

    # k elements one instruction folds into one rounding -- not the loop's k step.
    instruction_k: int | None = None

    # True is `tl.dot(a, b, acc)`, one rounding; False is `acc + tl.dot(a, b)`, two.
    use_fast_accum: bool | None = None

    # The mainloop's k step, set only when the short turn runs FIRST; None means last.
    k_loop_step: int | None = None

    # The nest of k cuts, outermost first. Empty is the ordinary non-split GEMM.
    k_cuts: tuple[KCut, ...] = ()

    epilogue: EpilogueOrder = EpilogueOrder.NONE

    def __post_init__(self) -> None:
        self._check_dtypes()
        self._check_algorithm()
        self._check_cuts()

    @property
    def is_floating_point(self) -> bool:
        """Whether this GEMM rounds at all. An integer GEMM does not."""
        return self.operand_dtype.is_floating_point

    def _check_dtypes(self) -> None:
        for name, dtype, allowed in (
            ("operand_dtype", self.operand_dtype, OPERAND_DTYPES),
            ("accumulate_dtype", self.accumulate_dtype, ACCUMULATE_DTYPES),
            ("output_dtype", self.output_dtype, OUTPUT_DTYPES),
        ):
            if dtype not in allowed:
                raise ValueError(
                    f"unsupported {name} {dtype}; supported: {_dtype_names(allowed)}."
                    " An unlisted dtype is rejected rather than run under some other"
                    " dtype's recipe, which would return a wrong answer quietly."
                )
        kinds = {
            self.operand_dtype.is_floating_point,
            self.accumulate_dtype.is_floating_point,
            self.output_dtype.is_floating_point,
        }
        if len(kinds) != 1:
            raise ValueError(
                f"operand_dtype {self.operand_dtype}, accumulate_dtype "
                f"{self.accumulate_dtype} and output_dtype {self.output_dtype} must "
                "all be floating point or all be integer"
            )

    def _check_algorithm(self) -> None:
        name = self.algorithm.name
        matrix = self.algorithm in MATRIX_INSTRUCTIONS
        if matrix and (self.instruction_k is None or self.instruction_k < 1):
            raise ValueError(
                f"{name} is a matrix instruction and needs instruction_k >= 1, the "
                "number of k elements one instruction sums in a single rounding, got "
                f"{self.instruction_k}"
            )
        if not matrix and self.instruction_k is not None:
            raise ValueError(
                f"instruction_k must be None for {name}: a scalar step always covers "
                f"exactly one k element, got {self.instruction_k}"
            )
        if not matrix and self.k_loop_step is not None:
            raise ValueError(
                f"k_loop_step must be None for {name}: with one rounding per k "
                "element there are no instruction groups for a short first turn to "
                "shift"
            )
        if matrix and self.use_fast_accum is None:
            raise ValueError(
                f"{name} is a matrix instruction and needs use_fast_accum: adding "
                "into the accumulator and adding into zero then merging are two "
                "different sums, and there is no safe default to pick for you"
            )
        if not matrix and self.use_fast_accum is not None:
            raise ValueError(
                f"use_fast_accum must be None for {name}: a scalar algorithm already "
                "says whether the multiply and the add round together, got "
                f"{self.use_fast_accum}"
            )
        if self.k_loop_step is not None and self.k_loop_step < 1:
            raise ValueError(f"k_loop_step must be at least 1, got {self.k_loop_step}")
        self._check_input_precision(matrix)

    def _check_input_precision(self, matrix: bool) -> None:
        if self.input_precision is InputPrecision.IEEE:
            return
        if self.operand_dtype is not torch.float32:
            raise ValueError(
                f"input_precision {self.input_precision.name} needs float32 operands,"
                f" got {self.operand_dtype}: it names how fp32 is cut down before the"
                " multiply"
            )
        if not matrix:
            raise ValueError(
                f"input_precision {self.input_precision.name} needs a matrix "
                f"instruction, got {self.algorithm.name}: tf32 exists only as a "
                "matrix instruction input"
            )

    def _check_cuts(self) -> None:
        if not isinstance(self.k_cuts, tuple):
            raise ValueError(
                "k_cuts must be a tuple so the descriptor stays hashable, got "
                f"{type(self.k_cuts).__name__}"
            )
        for level, part in enumerate(self.k_cuts):
            if not isinstance(part, KCut):
                raise ValueError(
                    f"k_cuts[{level}] must be a KCut, got {type(part).__name__}"
                )
            self._check_cut_dtypes(level, part)

    def _check_cut_dtypes(self, level: int, part: KCut) -> None:
        for name, dtype in (
            ("partial_dtype", part.partial_dtype),
            ("merge_dtype", part.merge_dtype),
        ):
            if dtype.is_floating_point != self.is_floating_point:
                raise ValueError(
                    f"k_cuts[{level}].{name} is {dtype}, which does not match "
                    f"an operand_dtype of {self.operand_dtype}"
                )


@dataclasses.dataclass(frozen=True)
class Equivalences:
    """What one machine treats as the same computation.

    Which `GEMMDesc` differences a machine cannot show you. That is measured one
    architecture at a time, so this module ships no instances; `Equivalences()`
    declares nothing, and under it `produces_same_bits` is exactly `==`.
    """

    # Groups of algorithms this machine cannot tell apart. Groups must not overlap.
    interchangeable_instructions: tuple[frozenset[GEMMAlgorithm], ...] = ()

    # Whether the two values of `use_fast_accum` give the same bits here.
    fast_accum_is_free: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.interchangeable_instructions, tuple):
            raise ValueError(
                "interchangeable_instructions must be a tuple so this stays "
                f"hashable, got {type(self.interchangeable_instructions).__name__}"
            )
        claimed: set[GEMMAlgorithm] = set()
        for index, group in enumerate(self.interchangeable_instructions):
            self._check_group(index, group)
            repeated = claimed & group
            if repeated:
                names = ", ".join(sorted(a.name for a in repeated))
                raise ValueError(
                    f"interchangeable_instructions[{index}] names {names}, which an "
                    "earlier group already claimed: overlapping groups would mean a "
                    "wider equivalence than either of them states"
                )
            claimed |= group

    @staticmethod
    def _check_group(index: int, group: frozenset[GEMMAlgorithm]) -> None:
        if not isinstance(group, frozenset):
            raise ValueError(
                f"interchangeable_instructions[{index}] must be a frozenset, got "
                f"{type(group).__name__}"
            )
        for member in group:
            if not isinstance(member, GEMMAlgorithm):
                raise ValueError(
                    f"interchangeable_instructions[{index}] must hold GEMMAlgorithm "
                    f"values, got {type(member).__name__}"
                )
        if len(group) < 2:
            raise ValueError(
                f"interchangeable_instructions[{index}] needs at least 2 algorithms, "
                f"since a group of one says nothing, got {len(group)}"
            )


def produces_same_bits(a: GEMMDesc, b: GEMMDesc, known: Equivalences) -> bool:
    """Whether the machine `known` describes returns the same bits for `a` and `b`.

    The weaker of the two relations: equal descriptors always pass, different ones
    only where `known` says the difference does not show. It fails closed -- every
    field not declared free is compared, including a field added later.
    """
    return _normalised(a, known) == _normalised(b, known)


def _normalised(desc: GEMMDesc, known: Equivalences) -> dict[str, object]:
    """`desc` with the differences `known` declares free rewritten to one form.

    A dict and not another `GEMMDesc`: it must keep comparing fields added later, and
    a canonical form need not be a legal descriptor.
    """
    values: dict[str, object] = {
        field.name: getattr(desc, field.name) for field in dataclasses.fields(desc)
    }
    values["algorithm"] = _canonical_algorithm(desc.algorithm, known)
    if known.fast_accum_is_free and desc.use_fast_accum is not None:
        values["use_fast_accum"] = True
    return values


def _canonical_algorithm(
    algorithm: GEMMAlgorithm, known: Equivalences
) -> GEMMAlgorithm:
    for group in known.interchangeable_instructions:
        if algorithm in group:
            # Any fixed member will do; groups cannot overlap, so a pair lands together.
            return min(group, key=lambda member: member.value)
    return algorithm
