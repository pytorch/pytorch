# Owner(s): ["module: inductor"]
from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable

import torch
from torch._inductor.gemm_desc import (
    EpilogueOrder,
    Equivalences,
    GEMMAlgorithm,
    GEMMDesc,
    InputPrecision,
    KCut,
    KCutLayout,
    MATRIX_INSTRUCTIONS,
    MergeOrder,
    produces_same_bits,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _fp16_matmul(**overrides: object) -> GEMMDesc:
    """An ordinary fp16 tensor-core GEMM. Every case below bends one of its fields."""
    fields: dict[str, object] = {
        "operand_dtype": torch.float16,
        "accumulate_dtype": torch.float32,
        "output_dtype": torch.float16,
        "algorithm": GEMMAlgorithm.MMA_SYNC,
        "instruction_k": 16,
        "use_fast_accum": True,
    }
    fields.update(overrides)
    return GEMMDesc(**fields)  # type: ignore[arg-type]


def _int8_matmul(**overrides: object) -> GEMMDesc:
    fields: dict[str, object] = {
        "operand_dtype": torch.int8,
        "accumulate_dtype": torch.int32,
        "output_dtype": torch.int32,
    }
    fields.update(overrides)
    return _fp16_matmul(**fields)


def _scalar_matmul(**overrides: object) -> GEMMDesc:
    """A scalar-loop GEMM: no instruction_k and no use_fast_accum."""
    fields: dict[str, object] = {
        "algorithm": GEMMAlgorithm.SCALAR_FUSED_MULTIPLY_ADD,
        "instruction_k": None,
        "use_fast_accum": None,
    }
    fields.update(overrides)
    return _fp16_matmul(**fields)


def _cut(**overrides: object) -> KCut:
    fields: dict[str, object] = {
        "span": 1024,
        "partial_dtype": torch.float32,
        "merge_dtype": torch.float32,
    }
    fields.update(overrides)
    return KCut(**fields)  # type: ignore[arg-type]


def _equivalences(**overrides: object) -> Equivalences:
    """Loose kwargs, so a case below can hand a field the wrong type on purpose."""
    return Equivalences(**overrides)  # type: ignore[arg-type]


# Written out here rather than imported from the module, so that dropping a dtype
# from the module turns these red instead of quietly shrinking the sweep.
_FLOAT_OPERAND_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
_FLOAT_ACCUMULATE_DTYPES = [torch.float16, torch.float32, torch.float64]
_FLOAT_OUTPUT_DTYPES = _FLOAT_OPERAND_DTYPES

_INT_OPERAND_DTYPES = [torch.int8, torch.uint8]
_INT_ACCUMULATE_DTYPES = [torch.int32, torch.int64]
_INT_OUTPUT_DTYPES = [torch.int8, torch.int32]

# Every triple the descriptor accepts: 6 x 3 x 6 floating plus 2 x 2 x 2 integer.
_SUPPORTED_DTYPE_TRIPLES = list(
    itertools.product(
        _FLOAT_OPERAND_DTYPES, _FLOAT_ACCUMULATE_DTYPES, _FLOAT_OUTPUT_DTYPES
    )
) + list(
    itertools.product(_INT_OPERAND_DTYPES, _INT_ACCUMULATE_DTYPES, _INT_OUTPUT_DTYPES)
)

# Written out here, like the dtypes above, so that adding an instruction to the
# module or dropping one turns these red instead of quietly shrinking the sweep.
_MATRIX_ALGORITHMS = [
    GEMMAlgorithm.MMA_SYNC,
    GEMMAlgorithm.WGMMA,
    GEMMAlgorithm.TCGEN05_MMA,
]

_SCALAR_ALGORITHMS = [
    GEMMAlgorithm.SCALAR_FUSED_MULTIPLY_ADD,
    GEMMAlgorithm.SCALAR_MULTIPLY_THEN_ADD,
]

_ALGORITHMS = _MATRIX_ALGORITHMS + _SCALAR_ALGORITHMS

_INPUT_PRECISIONS = [InputPrecision.IEEE, InputPrecision.TF32, InputPrecision.TF32X3]

_EPILOGUES = [EpilogueOrder.IN_ACCUMULATOR, EpilogueOrder.AFTER_ROUNDING]


_FP32_OPERANDS: dict[str, object] = {
    "operand_dtype": torch.float32,
    "output_dtype": torch.float32,
}

_SCALAR: dict[str, object] = {
    "algorithm": GEMMAlgorithm.SCALAR_FUSED_MULTIPLY_ADD,
    "instruction_k": None,
    "use_fast_accum": None,
}

# What only a matrix instruction has to obey, as (case, overrides, words the message
# must carry). Each is run against every instruction, so an instruction cannot arrive
# with the rules unwired for it.
_MATRIX_RULES: list[tuple[str, dict[str, object], str]] = [
    ("no_instruction_k", {"instruction_k": None}, "needs instruction_k"),
    ("instruction_k_zero", {"instruction_k": 0}, "needs instruction_k"),
    ("instruction_k_negative", {"instruction_k": -16}, "needs instruction_k"),
    ("no_use_fast_accum", {"use_fast_accum": None}, "needs use_fast_accum"),
]

# What only a scalar algorithm has to obey, run against both of them. Every message
# here ends just before the algorithm's name, so the test can check it is named.
_SCALAR_RULES: list[tuple[str, dict[str, object], str]] = [
    ("instruction_k", {"instruction_k": 1}, "instruction_k must be None for"),
    ("use_fast_accum", {"use_fast_accum": True}, "use_fast_accum must be None for"),
    (
        "use_fast_accum_false",
        {"use_fast_accum": False},
        "use_fast_accum must be None for",
    ),
    ("k_loop_step", {"k_loop_step": 64}, "k_loop_step must be None for"),
    (
        "tf32",
        {**_FP32_OPERANDS, "input_precision": InputPrecision.TF32},
        "needs a matrix instruction, got",
    ),
    (
        "tf32x3",
        {**_FP32_OPERANDS, "input_precision": InputPrecision.TF32X3},
        "needs a matrix instruction, got",
    ),
]

_MATRIX_RULE_CASES = [
    (f"{instruction.name}_{case}", instruction, overrides, expected)
    for instruction in _MATRIX_ALGORITHMS
    for case, overrides, expected in _MATRIX_RULES
]

_SCALAR_RULE_CASES = [
    (f"{algorithm.name}_{case}", algorithm, overrides, expected)
    for algorithm in _SCALAR_ALGORITHMS
    for case, overrides, expected in _SCALAR_RULES
]

# Two descriptors that differ in exactly one field must not compare equal. Each entry
# is (field, overrides for the left descriptor, overrides for the right one).
_ONE_FIELD_CHANGES: list[tuple[str, dict[str, object], dict[str, object]]] = [
    ("operand_dtype", {}, {"operand_dtype": torch.bfloat16}),
    ("accumulate_dtype", {}, {"accumulate_dtype": torch.float16}),
    ("output_dtype", {}, {"output_dtype": torch.float32}),
    # Both sides are scalar, so this really is one field: the two scalar algorithms
    # differ only in whether the multiply and the add round together.
    (
        "algorithm",
        _SCALAR,
        {**_SCALAR, "algorithm": GEMMAlgorithm.SCALAR_MULTIPLY_THEN_ADD},
    ),
    # Two matrix instructions, which is the same field again. Kept apart from the
    # scalar pair because a machine can declare these two interchangeable and can
    # never declare the scalar pair to be the same descriptor.
    ("matrix_instruction", {}, {"algorithm": GEMMAlgorithm.WGMMA}),
    (
        "input_precision",
        _FP32_OPERANDS,
        {**_FP32_OPERANDS, "input_precision": InputPrecision.TF32},
    ),
    ("instruction_k", {}, {"instruction_k": 8}),
    ("use_fast_accum", {}, {"use_fast_accum": False}),
    ("k_loop_step", {}, {"k_loop_step": 64}),
    ("k_cuts", {}, {"k_cuts": (_cut(),)}),
    ("epilogue", {}, {"epilogue": EpilogueOrder.IN_ACCUMULATOR}),
]

_STRIDED: dict[str, object] = {"layout": KCutLayout.STRIDED, "count": 4}

_ONE_CUT_FIELD_CHANGES: list[tuple[str, dict[str, object], dict[str, object]]] = [
    ("span", {}, {"span": 512}),
    ("partial_dtype", {}, {"partial_dtype": torch.float16}),
    ("merge_dtype", {}, {"merge_dtype": torch.float16}),
    ("merge", {}, {"merge": MergeOrder.PAIRWISE_TREE}),
    # layout and count move together: a CONTIGUOUS cut may not carry a count.
    ("layout", {}, _STRIDED),
    ("count", _STRIDED, {**_STRIDED, "count": 8}),
    # An unset period is the plain rotation, so stating one has to be a different
    # value -- otherwise a descriptor written before the field existed would silently
    # become one that means something else.
    ("period", _STRIDED, {**_STRIDED, "period": 37}),
    (
        "divisor",
        {**_STRIDED, "period": 37},
        {**_STRIDED, "period": 37, "divisor": 4},
    ),
]

# Every combination the descriptor refuses, and the words its message must carry.
# Building a descriptor is the only thing under test, so each case is a callable.
_REJECTED: dict[str, tuple[Callable[[], object], str]] = {
    "operand_dtype_unlisted": (
        lambda: _fp16_matmul(operand_dtype=torch.complex64),
        "unsupported operand_dtype",
    ),
    "operand_dtype_bool": (
        lambda: _fp16_matmul(operand_dtype=torch.bool),
        "unsupported operand_dtype",
    ),
    "operand_dtype_int32": (
        lambda: _fp16_matmul(
            operand_dtype=torch.int32,
            accumulate_dtype=torch.int32,
            output_dtype=torch.int32,
        ),
        "unsupported operand_dtype",
    ),
    "accumulate_dtype_fp8": (
        lambda: _fp16_matmul(accumulate_dtype=torch.float8_e4m3fn),
        "unsupported accumulate_dtype",
    ),
    "accumulate_dtype_bfloat16": (
        lambda: _fp16_matmul(accumulate_dtype=torch.bfloat16),
        "unsupported accumulate_dtype",
    ),
    "output_dtype_unlisted": (
        lambda: _fp16_matmul(output_dtype=torch.complex64),
        "unsupported output_dtype",
    ),
    "output_dtype_int64": (
        lambda: _int8_matmul(output_dtype=torch.int64),
        "unsupported output_dtype",
    ),
    "integer_operand_float_accumulate": (
        lambda: _fp16_matmul(operand_dtype=torch.int8, output_dtype=torch.int8),
        "floating point or all be integer",
    ),
    "float_operand_integer_accumulate": (
        lambda: _fp16_matmul(accumulate_dtype=torch.int32),
        "floating point or all be integer",
    ),
    "integer_operand_float_output": (
        lambda: _int8_matmul(output_dtype=torch.float16),
        "floating point or all be integer",
    ),
    "matrix_instruction_without_instruction_k": (
        lambda: _fp16_matmul(instruction_k=None),
        "MMA_SYNC is a matrix instruction and needs instruction_k",
    ),
    "instruction_k_zero": (
        lambda: _fp16_matmul(instruction_k=0),
        "MMA_SYNC is a matrix instruction and needs instruction_k",
    ),
    "matrix_instruction_without_use_fast_accum": (
        lambda: _fp16_matmul(use_fast_accum=None),
        "MMA_SYNC is a matrix instruction and needs use_fast_accum",
    ),
    "instruction_k_on_scalar_algorithm": (
        lambda: _scalar_matmul(instruction_k=1),
        "instruction_k must be None",
    ),
    "use_fast_accum_on_scalar_algorithm": (
        lambda: _scalar_matmul(use_fast_accum=True),
        "use_fast_accum must be None",
    ),
    "use_fast_accum_false_on_scalar_algorithm": (
        lambda: _scalar_matmul(use_fast_accum=False),
        "use_fast_accum must be None",
    ),
    "k_loop_step_on_scalar_algorithm": (
        lambda: _scalar_matmul(
            algorithm=GEMMAlgorithm.SCALAR_MULTIPLY_THEN_ADD, k_loop_step=64
        ),
        "k_loop_step must be None",
    ),
    "k_loop_step_zero": (
        lambda: _fp16_matmul(k_loop_step=0),
        "k_loop_step must be at least 1",
    ),
    "tf32_on_fp16_operands": (
        lambda: _fp16_matmul(input_precision=InputPrecision.TF32),
        "needs float32 operands",
    ),
    "tf32x3_on_fp16_operands": (
        lambda: _fp16_matmul(input_precision=InputPrecision.TF32X3),
        "needs float32 operands",
    ),
    "tf32_on_scalar_algorithm": (
        lambda: _scalar_matmul(
            **_FP32_OPERANDS, input_precision=InputPrecision.TF32  # type: ignore[arg-type]
        ),
        "needs a matrix instruction",
    ),
    "tf32x3_on_scalar_algorithm": (
        lambda: _scalar_matmul(
            **_FP32_OPERANDS, input_precision=InputPrecision.TF32X3  # type: ignore[arg-type]
        ),
        "needs a matrix instruction",
    ),
    "k_cuts_as_list": (lambda: _fp16_matmul(k_cuts=[_cut()]), "must be a tuple"),
    "k_cuts_holding_a_number": (lambda: _fp16_matmul(k_cuts=(1024,)), "must be a KCut"),
    "k_cuts_holding_a_number_at_level_one": (
        lambda: _fp16_matmul(k_cuts=(_cut(), 64)),
        r"k_cuts\[1\] must be a KCut",
    ),
    "cut_partial_dtype_wrong_kind": (
        lambda: _fp16_matmul(k_cuts=(_cut(partial_dtype=torch.int32),)),
        "partial_dtype is torch.int32, which does not match",
    ),
    "cut_merge_dtype_wrong_kind": (
        lambda: _fp16_matmul(k_cuts=(_cut(merge_dtype=torch.int64),)),
        "merge_dtype is torch.int64, which does not match",
    ),
    "integer_gemm_with_float_cut": (
        lambda: _int8_matmul(k_cuts=(_cut(),)),
        "partial_dtype is torch.float32, which does not match",
    ),
    "cut_span_zero": (lambda: _cut(span=0), "span must be at least 1"),
    "cut_span_negative": (lambda: _cut(span=-8), "span must be at least 1"),
    "contiguous_cut_with_a_count": (
        lambda: _cut(count=4),
        "count must be None for a CONTIGUOUS cut",
    ),
    "strided_cut_without_a_count": (
        lambda: _cut(layout=KCutLayout.STRIDED),
        "STRIDED cut needs count",
    ),
    "strided_cut_with_one_part": (
        lambda: _cut(layout=KCutLayout.STRIDED, count=1),
        "STRIDED cut needs count",
    ),
    "cut_partial_dtype_fp8": (
        lambda: _cut(partial_dtype=torch.float8_e5m2),
        "unsupported partial_dtype",
    ),
    "cut_merge_dtype_fp8": (
        lambda: _cut(merge_dtype=torch.float8_e4m3fn),
        "unsupported merge_dtype",
    ),
    "cut_merge_dtype_int8": (
        lambda: _cut(merge_dtype=torch.int8),
        "unsupported merge_dtype",
    ),
    "contiguous_cut_with_a_period": (
        lambda: _cut(period=8),
        "period must be unset for a CONTIGUOUS cut",
    ),
    "contiguous_cut_with_a_divisor": (
        lambda: _cut(divisor=2),
        "divisor must be unset for a CONTIGUOUS cut",
    ),
    "divisor_without_a_period": (
        lambda: _cut(layout=KCutLayout.STRIDED, count=4, divisor=4),
        "divisor needs a period",
    ),
    "cut_period_zero": (
        lambda: _cut(layout=KCutLayout.STRIDED, count=4, period=0),
        "period must be at least 1",
    ),
    "cut_divisor_zero": (
        lambda: _cut(layout=KCutLayout.STRIDED, count=4, period=8, divisor=0),
        "divisor must be at least 1",
    ),
    "equivalence_groups_as_a_list": (
        lambda: _equivalences(
            interchangeable_instructions=[frozenset(_MATRIX_ALGORITHMS)]
        ),
        "must be a tuple",
    ),
    "equivalence_group_as_a_plain_set": (
        lambda: _equivalences(interchangeable_instructions=(set(_MATRIX_ALGORITHMS),)),
        r"interchangeable_instructions\[0\] must be a frozenset",
    ),
    "equivalence_group_of_one": (
        lambda: Equivalences(
            interchangeable_instructions=(frozenset({GEMMAlgorithm.WGMMA}),)
        ),
        "needs at least 2 algorithms",
    ),
    "equivalence_group_of_none": (
        lambda: Equivalences(interchangeable_instructions=(frozenset(),)),
        "needs at least 2 algorithms",
    ),
    "equivalence_group_of_names": (
        lambda: _equivalences(
            interchangeable_instructions=(frozenset({"MMA_SYNC", "WGMMA"}),)
        ),
        "must hold GEMMAlgorithm values",
    ),
    "equivalence_groups_overlap": (
        lambda: Equivalences(
            interchangeable_instructions=(
                frozenset({GEMMAlgorithm.MMA_SYNC, GEMMAlgorithm.WGMMA}),
                frozenset({GEMMAlgorithm.WGMMA, GEMMAlgorithm.TCGEN05_MMA}),
            )
        ),
        r"interchangeable_instructions\[1\] names WGMMA",
    ),
}


# Made-up machines. What a real one declares is measured per architecture and belongs
# to the producer that measured it, never to the module under test.
_NOTHING_DECLARED = Equivalences()

_TWO_INSTRUCTIONS_AGREE = Equivalences(
    interchangeable_instructions=(
        frozenset({GEMMAlgorithm.MMA_SYNC, GEMMAlgorithm.TCGEN05_MMA}),
    )
)

_FAST_ACCUM_FREE = Equivalences(fast_accum_is_free=True)

_EVERYTHING_DECLARED = Equivalences(
    interchangeable_instructions=(frozenset(_MATRIX_ALGORITHMS),),
    fast_accum_is_free=True,
)

_MACHINES = [
    ("nothing_declared", _NOTHING_DECLARED),
    ("two_instructions_agree", _TWO_INSTRUCTIONS_AGREE),
    ("fast_accum_free", _FAST_ACCUM_FREE),
    ("everything_declared", _EVERYTHING_DECLARED),
]

# The only two entries of _ONE_FIELD_CHANGES that _EVERYTHING_DECLARED makes free.
# Every other one must survive it, however much is declared.
_FREE_UNDER_EVERYTHING = {"matrix_instruction", "use_fast_accum"}


def _case_name(name: str, *_: object) -> str:
    return name


@instantiate_parametrized_tests
class TestGEMMDesc(TestCase):
    def test_plain_matmul_reads_back(self):
        desc = _fp16_matmul()
        self.assertEqual(desc.operand_dtype, torch.float16)
        self.assertEqual(desc.accumulate_dtype, torch.float32)
        self.assertEqual(desc.output_dtype, torch.float16)
        self.assertEqual(desc.algorithm, GEMMAlgorithm.MMA_SYNC)
        self.assertEqual(desc.instruction_k, 16)
        self.assertEqual(desc.use_fast_accum, True)
        # The defaults are the ordinary GEMM: no cut, no epilogue, full precision.
        self.assertEqual(desc.input_precision, InputPrecision.IEEE)
        self.assertEqual(desc.k_loop_step, None)
        self.assertEqual(desc.k_cuts, ())
        self.assertEqual(desc.epilogue, EpilogueOrder.NONE)
        self.assertTrue(desc.is_floating_point)

    @parametrize("algorithm", _ALGORITHMS, name_fn=lambda a: a.name)
    def test_every_algorithm(self, algorithm):
        matrix = algorithm in MATRIX_INSTRUCTIONS
        desc = _fp16_matmul(
            algorithm=algorithm,
            instruction_k=16 if matrix else None,
            use_fast_accum=False if matrix else None,
        )
        self.assertEqual(desc.algorithm, algorithm)
        self.assertEqual(desc.instruction_k, 16 if matrix else None)
        self.assertEqual(desc.use_fast_accum, False if matrix else None)

    def test_the_algorithms_are_five_distinct_values(self):
        """Each instruction is its own value, and the module agrees which are which."""
        self.assertEqual(len(set(_ALGORITHMS)), 5)
        self.assertEqual(MATRIX_INSTRUCTIONS, frozenset(_MATRIX_ALGORITHMS))
        self.assertEqual(MATRIX_INSTRUCTIONS & frozenset(_SCALAR_ALGORITHMS), set())

    @parametrize("instruction", _MATRIX_ALGORITHMS, name_fn=lambda a: a.name)
    def test_every_matrix_instruction_builds(self, instruction):
        desc = _fp16_matmul(algorithm=instruction)
        self.assertEqual(desc.algorithm, instruction)
        self.assertEqual(desc.instruction_k, 16)
        self.assertEqual(desc.use_fast_accum, True)
        # Two descriptors that name different instructions are different descriptors.
        others = [i for i in _MATRIX_ALGORITHMS if i is not instruction]
        for other in others:
            self.assertNotEqual(desc, _fp16_matmul(algorithm=other))
        self.assertEqual(len({desc, *(_fp16_matmul(algorithm=o) for o in others)}), 3)

    @parametrize(
        "case,instruction,overrides,expected", _MATRIX_RULE_CASES, name_fn=_case_name
    )
    def test_matrix_rule_bites_for_every_instruction(
        self, case, instruction, overrides, expected
    ):
        message = rf"{instruction.name} is a matrix instruction and {expected}"
        with self.assertRaisesRegex(ValueError, message):
            _fp16_matmul(algorithm=instruction, **overrides)

    @parametrize(
        "case,algorithm,overrides,expected", _SCALAR_RULE_CASES, name_fn=_case_name
    )
    def test_scalar_rule_bites_for_every_scalar_algorithm(
        self, case, algorithm, overrides, expected
    ):
        with self.assertRaisesRegex(ValueError, rf"{expected} {algorithm.name}"):
            _scalar_matmul(algorithm=algorithm, **overrides)

    @parametrize("use_fast_accum", [True, False])
    def test_matrix_instruction_accumulate_choice_reads_back(self, use_fast_accum):
        desc = _fp16_matmul(use_fast_accum=use_fast_accum)
        self.assertEqual(desc.use_fast_accum, use_fast_accum)

    def test_one_cut(self):
        desc = _fp16_matmul(k_cuts=(_cut(span=4096),))
        self.assertEqual(len(desc.k_cuts), 1)
        self.assertEqual(desc.k_cuts[0].span, 4096)
        self.assertEqual(desc.k_cuts[0].layout, KCutLayout.CONTIGUOUS)
        self.assertEqual(desc.k_cuts[0].count, None)
        self.assertEqual(desc.k_cuts[0].merge, MergeOrder.SEQUENTIAL)

    def test_two_nested_cuts(self):
        """Slices of k, and inside a slice an accumulator closed every 64 elements."""
        desc = _fp16_matmul(k_cuts=(_cut(span=4096), _cut(span=64)))
        self.assertEqual(len(desc.k_cuts), 2)
        self.assertEqual(desc.k_cuts[0].span, 4096)
        self.assertEqual(desc.k_cuts[1].span, 64)
        self.assertEqual(desc.k_cuts[1].merge, MergeOrder.SEQUENTIAL)

    def test_split_k_partials_kept_in_the_output_dtype(self):
        """The merge dtypes are fields, so this is sayable rather than assumed."""
        desc = _fp16_matmul(
            k_cuts=(
                KCut(
                    span=2048,
                    partial_dtype=torch.float16,
                    merge_dtype=torch.float16,
                ),
            ),
        )
        self.assertEqual(desc.k_cuts[0].partial_dtype, torch.float16)
        self.assertEqual(desc.k_cuts[0].merge_dtype, torch.float16)

    def test_residue_first_matmul(self):
        desc = _fp16_matmul(instruction_k=16, k_loop_step=32)
        self.assertEqual(desc.k_loop_step, 32)

    def test_strided_cut_with_a_lane_tree(self):
        desc = _scalar_matmul(
            k_cuts=(
                KCut(
                    span=4,
                    partial_dtype=torch.float32,
                    merge_dtype=torch.float32,
                    layout=KCutLayout.STRIDED,
                    count=32,
                    merge=MergeOrder.PAIRWISE_TREE,
                ),
            ),
        )
        cut = desc.k_cuts[0]
        self.assertEqual(cut.span, 4)
        self.assertEqual(cut.layout, KCutLayout.STRIDED)
        self.assertEqual(cut.count, 32)
        self.assertEqual(cut.merge, MergeOrder.PAIRWISE_TREE)

    def test_integer_gemm_with_integer_cuts(self):
        desc = _int8_matmul(
            k_cuts=(_cut(partial_dtype=torch.int32, merge_dtype=torch.int32),)
        )
        self.assertFalse(desc.is_floating_point)
        self.assertEqual(desc.k_cuts[0].partial_dtype, torch.int32)

    @parametrize("operand,accumulate,output", _SUPPORTED_DTYPE_TRIPLES)
    def test_every_supported_dtype_triple(self, operand, accumulate, output):
        desc = _fp16_matmul(
            operand_dtype=operand,
            accumulate_dtype=accumulate,
            output_dtype=output,
        )
        self.assertEqual(desc.operand_dtype, operand)
        self.assertEqual(desc.accumulate_dtype, accumulate)
        self.assertEqual(desc.output_dtype, output)
        self.assertEqual(desc.is_floating_point, operand.is_floating_point)

    @parametrize("input_precision", _INPUT_PRECISIONS, name_fn=lambda p: p.name)
    def test_fp32_input_precision(self, input_precision):
        desc = _fp16_matmul(**_FP32_OPERANDS, input_precision=input_precision)
        self.assertEqual(desc.input_precision, input_precision)

    @parametrize("epilogue", _EPILOGUES, name_fn=lambda e: e.name)
    def test_epilogue(self, epilogue):
        desc = _fp16_matmul(epilogue=epilogue)
        self.assertEqual(desc.epilogue, epilogue)

    def test_equal_descriptors_are_equal_and_hash_alike(self):
        a = _fp16_matmul(k_cuts=(_cut(),))
        b = _fp16_matmul(k_cuts=(_cut(),))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    def test_equal_cuts_are_equal_and_hash_alike(self):
        a, b = _cut(), _cut()
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    @parametrize("field,left,right", _ONE_FIELD_CHANGES, name_fn=_case_name)
    def test_one_changed_field_differs(self, field, left, right):
        a, b = _fp16_matmul(**left), _fp16_matmul(**right)
        self.assertNotEqual(a, b)
        self.assertEqual(len({a, b}), 2)

    @parametrize("field,left,right", _ONE_CUT_FIELD_CHANGES, name_fn=_case_name)
    def test_one_changed_cut_field_differs(self, field, left, right):
        a, b = _cut(**left), _cut(**right)
        self.assertNotEqual(a, b)
        self.assertNotEqual(_fp16_matmul(k_cuts=(a,)), _fp16_matmul(k_cuts=(b,)))

    def test_cut_nesting_order_matters(self):
        outer_first = _fp16_matmul(k_cuts=(_cut(span=4096), _cut(span=64)))
        inner_first = _fp16_matmul(k_cuts=(_cut(span=64), _cut(span=4096)))
        self.assertNotEqual(outer_first, inner_first)
        self.assertEqual(len({outer_first, inner_first}), 2)

    def test_usable_as_a_dict_key(self):
        table = {
            _fp16_matmul(): "matmul",
            _fp16_matmul(k_cuts=(_cut(),)): "split k",
        }
        self.assertEqual(len(table), 2)
        self.assertEqual(table[_fp16_matmul()], "matmul")
        self.assertEqual(table[_fp16_matmul(k_cuts=(_cut(),))], "split k")

    def test_descriptor_is_frozen(self):
        desc = _fp16_matmul()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            desc.instruction_k = 8
        self.assertEqual(desc.instruction_k, 16)

    def test_cut_is_frozen(self):
        cut = _cut()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cut.span = 1
        self.assertEqual(cut.span, 1024)

    def test_replace_builds_a_new_value_and_revalidates(self):
        base = _fp16_matmul()
        wider = dataclasses.replace(base, instruction_k=32)
        self.assertEqual(base.instruction_k, 16)
        self.assertEqual(wider.instruction_k, 32)
        with self.assertRaisesRegex(ValueError, "MMA_SYNC is a matrix instruction"):
            dataclasses.replace(base, instruction_k=0)
        with self.assertRaisesRegex(ValueError, "unsupported output_dtype"):
            dataclasses.replace(base, output_dtype=torch.complex64)

    def test_replace_on_a_cut_revalidates(self):
        base = _cut()
        shorter = dataclasses.replace(base, span=16)
        self.assertEqual(shorter.span, 16)
        with self.assertRaisesRegex(ValueError, "span must be at least 1"):
            dataclasses.replace(base, span=0)
        with self.assertRaisesRegex(ValueError, "CONTIGUOUS cut"):
            dataclasses.replace(base, count=4)

    @parametrize("case", sorted(_REJECTED))
    def test_rejects(self, case):
        build, expected = _REJECTED[case]
        with self.assertRaisesRegex(ValueError, expected):
            build()

    def test_the_one_field_sweep_covers_every_field(self):
        """A field added to GEMMDesc turns this red until the sweep covers it.

        That matters more than it looks: the sweep below is what shows a new field
        is significant to `produces_same_bits` until someone measures otherwise.
        """
        covered = {field for field, _, _ in _ONE_FIELD_CHANGES}
        self.assertEqual(
            covered - {"matrix_instruction"},
            {field.name for field in dataclasses.fields(GEMMDesc)},
        )

    def test_the_one_cut_field_sweep_covers_every_cut_field(self):
        covered = {field for field, _, _ in _ONE_CUT_FIELD_CHANGES}
        self.assertEqual(covered, {field.name for field in dataclasses.fields(KCut)})

    def test_equivalences_is_a_value(self):
        self.assertEqual(Equivalences(), Equivalences())
        self.assertEqual(hash(Equivalences()), hash(Equivalences()))
        self.assertNotEqual(_NOTHING_DECLARED, _FAST_ACCUM_FREE)
        self.assertEqual(len({_NOTHING_DECLARED, _TWO_INSTRUCTIONS_AGREE}), 2)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            _NOTHING_DECLARED.fast_accum_is_free = True

    def test_nothing_declared_is_the_default(self):
        self.assertEqual(Equivalences().interchangeable_instructions, ())
        self.assertFalse(Equivalences().fast_accum_is_free)

    @parametrize("machine,known", _MACHINES, name_fn=_case_name)
    def test_produces_same_bits_is_reflexive(self, machine, known):
        for desc in (
            _fp16_matmul(),
            _fp16_matmul(use_fast_accum=False),
            _fp16_matmul(algorithm=GEMMAlgorithm.WGMMA),
            _fp16_matmul(algorithm=GEMMAlgorithm.TCGEN05_MMA),
            _scalar_matmul(),
            _scalar_matmul(algorithm=GEMMAlgorithm.SCALAR_MULTIPLY_THEN_ADD),
            _int8_matmul(),
            _fp16_matmul(k_cuts=(_cut(), _cut(span=64))),
        ):
            self.assertTrue(produces_same_bits(desc, desc, known))
            # A separately built copy, so this is value equality and not identity.
            twin = dataclasses.replace(desc)
            self.assertIsNot(desc, twin)
            self.assertTrue(produces_same_bits(desc, twin, known))

    @parametrize("field,left,right", _ONE_FIELD_CHANGES, name_fn=_case_name)
    def test_nothing_declared_means_plain_equality(self, field, left, right):
        a, b = _fp16_matmul(**left), _fp16_matmul(**right)
        self.assertFalse(produces_same_bits(a, b, _NOTHING_DECLARED))
        self.assertFalse(produces_same_bits(b, a, _NOTHING_DECLARED))

    @parametrize("field,left,right", _ONE_FIELD_CHANGES, name_fn=_case_name)
    def test_only_the_declared_differences_are_free(self, field, left, right):
        a, b = _fp16_matmul(**left), _fp16_matmul(**right)
        free = field in _FREE_UNDER_EVERYTHING
        self.assertEqual(produces_same_bits(a, b, _EVERYTHING_DECLARED), free)
        self.assertEqual(produces_same_bits(b, a, _EVERYTHING_DECLARED), free)

    def test_a_declared_pair_of_instructions_agrees(self):
        a = _fp16_matmul(algorithm=GEMMAlgorithm.MMA_SYNC)
        b = _fp16_matmul(algorithm=GEMMAlgorithm.TCGEN05_MMA)
        self.assertNotEqual(a, b)
        self.assertTrue(produces_same_bits(a, b, _TWO_INSTRUCTIONS_AGREE))
        self.assertTrue(produces_same_bits(b, a, _TWO_INSTRUCTIONS_AGREE))
        self.assertFalse(produces_same_bits(a, b, _NOTHING_DECLARED))
        self.assertFalse(produces_same_bits(a, b, _FAST_ACCUM_FREE))

    def test_an_instruction_outside_the_declared_group_does_not_agree(self):
        """The Blackwell fp8 case: mma.sync and wgmma are simply two answers here."""
        a = _fp16_matmul(algorithm=GEMMAlgorithm.MMA_SYNC)
        b = _fp16_matmul(algorithm=GEMMAlgorithm.WGMMA)
        self.assertFalse(produces_same_bits(a, b, _TWO_INSTRUCTIONS_AGREE))
        self.assertFalse(produces_same_bits(b, a, _TWO_INSTRUCTIONS_AGREE))
        self.assertTrue(produces_same_bits(a, b, _EVERYTHING_DECLARED))

    def test_two_declared_instructions_still_differ_in_another_field(self):
        a = _fp16_matmul(algorithm=GEMMAlgorithm.MMA_SYNC, instruction_k=16)
        b = _fp16_matmul(algorithm=GEMMAlgorithm.TCGEN05_MMA, instruction_k=32)
        self.assertFalse(produces_same_bits(a, b, _EVERYTHING_DECLARED))

    def test_fast_accum_is_free_where_it_is_declared(self):
        a = _fp16_matmul(use_fast_accum=True)
        b = _fp16_matmul(use_fast_accum=False)
        self.assertNotEqual(a, b)
        self.assertTrue(produces_same_bits(a, b, _FAST_ACCUM_FREE))
        self.assertTrue(produces_same_bits(b, a, _FAST_ACCUM_FREE))
        self.assertFalse(produces_same_bits(a, b, _NOTHING_DECLARED))
        self.assertFalse(produces_same_bits(a, b, _TWO_INSTRUCTIONS_AGREE))

    def test_a_free_fast_accum_and_a_free_instruction_together(self):
        a = _fp16_matmul(algorithm=GEMMAlgorithm.MMA_SYNC, use_fast_accum=True)
        b = _fp16_matmul(algorithm=GEMMAlgorithm.TCGEN05_MMA, use_fast_accum=False)
        self.assertFalse(produces_same_bits(a, b, _TWO_INSTRUCTIONS_AGREE))
        self.assertFalse(produces_same_bits(a, b, _FAST_ACCUM_FREE))
        self.assertTrue(produces_same_bits(a, b, _EVERYTHING_DECLARED))

    def test_fast_accum_free_says_nothing_about_a_scalar_algorithm(self):
        """use_fast_accum is None there, so there is no second form to be free of."""
        fused = _scalar_matmul()
        separate = _scalar_matmul(algorithm=GEMMAlgorithm.SCALAR_MULTIPLY_THEN_ADD)
        self.assertFalse(produces_same_bits(fused, separate, _EVERYTHING_DECLARED))
        self.assertTrue(produces_same_bits(fused, fused, _EVERYTHING_DECLARED))
        # Nor does it pull a scalar algorithm and a matrix instruction together.
        for use_fast_accum in (True, False):
            matrix = _fp16_matmul(use_fast_accum=use_fast_accum)
            self.assertFalse(produces_same_bits(fused, matrix, _EVERYTHING_DECLARED))

    @parametrize("field,left,right", _ONE_CUT_FIELD_CHANGES, name_fn=_case_name)
    def test_a_changed_cut_field_is_never_free(self, field, left, right):
        a = _fp16_matmul(k_cuts=(_cut(**left),))
        b = _fp16_matmul(k_cuts=(_cut(**right),))
        self.assertFalse(produces_same_bits(a, b, _EVERYTHING_DECLARED))
        # Also one level down, where only the inner cut differs.
        outer = _cut(span=4096)
        self.assertFalse(
            produces_same_bits(
                _fp16_matmul(k_cuts=(outer, _cut(**left))),
                _fp16_matmul(k_cuts=(outer, _cut(**right))),
                _EVERYTHING_DECLARED,
            )
        )

    def test_matching_cuts_do_not_block_a_free_instruction(self):
        a = _fp16_matmul(algorithm=GEMMAlgorithm.MMA_SYNC, k_cuts=(_cut(),))
        b = _fp16_matmul(
            algorithm=GEMMAlgorithm.TCGEN05_MMA, k_cuts=(_cut(),)  # rebuilt, not shared
        )
        self.assertTrue(produces_same_bits(a, b, _TWO_INSTRUCTIONS_AGREE))

    def test_cuts_of_different_depths_are_never_free(self):
        one = _fp16_matmul(k_cuts=(_cut(),))
        two = _fp16_matmul(k_cuts=(_cut(), _cut(span=64)))
        swapped = _fp16_matmul(k_cuts=(_cut(span=64), _cut()))
        self.assertFalse(produces_same_bits(one, two, _EVERYTHING_DECLARED))
        self.assertFalse(produces_same_bits(two, swapped, _EVERYTHING_DECLARED))

    def test_a_declared_scalar_pair_agrees(self):
        """Integer arithmetic is exact, so a machine may declare the scalar pair."""
        exact = Equivalences(
            interchangeable_instructions=(frozenset(_SCALAR_ALGORITHMS),)
        )
        fused = _int8_matmul(**_SCALAR)
        separate = _int8_matmul(
            **{**_SCALAR, "algorithm": GEMMAlgorithm.SCALAR_MULTIPLY_THEN_ADD}
        )
        self.assertNotEqual(fused, separate)
        self.assertTrue(produces_same_bits(fused, separate, exact))
        self.assertFalse(produces_same_bits(fused, separate, _EVERYTHING_DECLARED))


if __name__ == "__main__":
    run_tests()
