# Owner(s): ["module: inductor"]

import operator

import torch
from torch._inductor.kernel.flex.flex_flydsl_mask import lower_flydsl_mask_graph
from torch._inductor.test_case import TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)


def _mask_graph(mask_mod):
    indices = tuple(torch.tensor(0, dtype=torch.int32) for _ in range(4))
    return make_fx(mask_mod)(*indices)


def _evaluate_mask_program(program, batch_index, head_index, query_index, kv_index):
    values = [batch_index, head_index, query_index, kv_index]
    for instruction in program.instructions:
        opcode = instruction[0]
        if opcode in ("const_i32", "const_bool"):
            values.append(instruction[1])
            continue
        lhs = values[instruction[1]]
        rhs = values[instruction[2]]
        values.append(
            {
                "add": operator.add,
                "sub": operator.sub,
                "mul": operator.mul,
                "remainder": operator.mod,
                "ge": operator.ge,
                "lt": operator.lt,
                "eq": operator.eq,
                "and": operator.and_,
            }[opcode](lhs, rhs)
        )
    return values[program.output]


@instantiate_parametrized_tests
class TestFlyDSLFlexMaskLowering(TestCase):
    def test_mask_lowering_rejects_float_constant(self):
        graph_module = torch.fx.symbolic_trace(
            lambda batch_index, head_index, query_index, kv_index: query_index + -0.5
            >= kv_index
        )
        program, reason = lower_flydsl_mask_graph(graph_module, ())
        self.assertIsNone(program)
        self.assertIn("scalar constant -0.5 is unsupported", reason)

    @parametrize(
        "mask_mod,query_index,kv_index,expected",
        [
            subtest(
                (
                    lambda batch_index, head_index, query_index, kv_index: torch.add(
                        query_index, 64, alpha=2
                    )
                    >= kv_index,
                    0,
                    100,
                    True,
                ),
                name="add_alpha",
            ),
            subtest(
                (
                    lambda batch_index, head_index, query_index, kv_index: torch.sub(
                        query_index, 64, alpha=2
                    )
                    >= kv_index,
                    200,
                    100,
                    False,
                ),
                name="sub_alpha",
            ),
            subtest(
                (
                    lambda batch_index, head_index, query_index, kv_index: query_index
                    + 8188
                    >= kv_index,
                    3,
                    8191,
                    True,
                ),
                name="decode_boundary_inside",
            ),
            subtest(
                (
                    lambda batch_index, head_index, query_index, kv_index: query_index
                    + 8188
                    >= kv_index,
                    2,
                    8191,
                    False,
                ),
                name="decode_boundary_outside",
            ),
            subtest(
                (
                    lambda batch_index, head_index, query_index, kv_index: (
                        query_index >= kv_index
                    )
                    & (query_index - kv_index < 96),
                    95,
                    0,
                    True,
                ),
                name="window_boundary_inside",
            ),
            subtest(
                (
                    lambda batch_index, head_index, query_index, kv_index: (
                        query_index >= kv_index
                    )
                    & (query_index - kv_index < 96),
                    96,
                    0,
                    False,
                ),
                name="window_boundary_outside",
            ),
            subtest(
                (
                    lambda batch_index, head_index, query_index, kv_index: (
                        query_index - kv_index
                    )
                    % 8
                    == 1,
                    1,
                    8,
                    True,
                ),
                name="negative_lhs_remainder",
            ),
        ],
    )
    def test_mask_lowering_edge_cases(self, mask_mod, query_index, kv_index, expected):
        program, reason = lower_flydsl_mask_graph(_mask_graph(mask_mod), ())
        self.assertIsNotNone(program, reason)
        self.assertEqual(
            _evaluate_mask_program(program, 0, 0, query_index, kv_index), expected
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
