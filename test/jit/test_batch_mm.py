# Owner(s): ["oncall: jit"]

import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestBatchMM(JitTestCase):
    @staticmethod
    def _get_test_tensors(n: int):
        return [
            torch.tensor([[1 + x, 2 + x, 3 + x], [4 + x, 5 + x, 6 + x]])
            if x % 2 == 0
            else torch.tensor([[1 + x, 2 + x], [3 + x, 4 + x], [5 + x, 6 + x]])
            for x in range(n)
        ]

    def test_batch_mm_no_mutation(self):
        def test_batch_mm(
            T1: torch.Tensor,
            T2: torch.Tensor,
            T3: torch.Tensor,
            T4: torch.Tensor,
            T5: torch.Tensor,
            T6: torch.Tensor,
            T7: torch.Tensor,
            T8: torch.Tensor,
        ):
            return (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )

        test_batch_mm_scripted = torch.jit.script(test_batch_mm)

        tensors = TestBatchMM._get_test_tensors(8)
        expected = test_batch_mm(*tensors)

        FileCheck().check_count("aten::mm", 4, exactly=True).run(
            test_batch_mm_scripted.graph
        )
        self.run_pass("batch_mm", test_batch_mm_scripted.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).run(
            test_batch_mm_scripted.graph
        )

        actual = test_batch_mm_scripted(*tensors)
        self.assertEqual(expected, actual, atol=1e-9, rtol=1e-9)

    def test_batch_mm_permitted_mutation(self):
        def test_batch_mm(
            T1: torch.Tensor,
            T2: torch.Tensor,
            T3: torch.Tensor,
            T4: torch.Tensor,
            T5: torch.Tensor,
            T6: torch.Tensor,
            T7: torch.Tensor,
            T8: torch.Tensor,
        ):
            result = {}
            result["product"] = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )
            result["constant"] = torch.tensor([42.0])
            return result

        test_batch_mm_scripted = torch.jit.script(test_batch_mm)

        tensors = TestBatchMM._get_test_tensors(8)
        expected = test_batch_mm(*tensors)

        FileCheck().check_count("aten::mm", 4, exactly=True).run(
            test_batch_mm_scripted.graph
        )
        self.run_pass("batch_mm", test_batch_mm_scripted.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).run(
            test_batch_mm_scripted.graph
        )

        actual = test_batch_mm_scripted(*tensors)
        self.assertEqual(expected, actual, atol=1e-9, rtol=1e-9)

    def test_batch_mm_prohibited_mutation(self):
        @torch.jit.script
        def test_batch_mm(n: int):
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            torch.relu_(T1)
            result = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )
            return result

        FileCheck().check_count("aten::mm", 4, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 4, exactly=True).check_not(
            "prim::MMTreeReduce"
        ).run(test_batch_mm.graph)

    def test_batch_mm_prohibited_mutation_multiple_adds(self):
        @torch.jit.script
        def test_batch_mm(n: int):
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            torch.relu_(T1)
            result = {}
            result["no_mutated_parameters"] = (
                torch.mm(T2, T3)
                + torch.mm(T4, T5)
                + torch.mm(T6, T7)
                + torch.mm(T8, T9)
            )
            result["all_parameters"] = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
                + torch.mm(T9, T10)
            )
            return result

        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).check_count(
            "aten::mm", 5, exactly=True
        ).run(test_batch_mm.graph)

    def test_batch_mm_prohibited_mutation_if_node(self):
        @torch.jit.script
        def test_batch_mm(n: int, use_t1: bool):
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            if use_t1:
                torch.relu_(T1)
                return (
                    torch.mm(T1, T2)
                    + torch.mm(T3, T4)
                    + torch.mm(T5, T6)
                    + torch.mm(T7, T8)
                    + torch.mm(T9, T10)
                )
            else:
                return (
                    torch.mm(T2, T3)
                    + torch.mm(T4, T5)
                    + torch.mm(T6, T7)
                    + torch.mm(T8, T9)
                )

        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 5, exactly=True).check_count(
            "prim::MMTreeReduce", 1, exactly=True
        ).run(test_batch_mm.graph)

    def test_batch_mm_side_permitted_mutation(self):
        @torch.jit.script
        def test_batch_mm(n: int):
            result = {}
            A = torch.zeros((n, n))
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            return result

        FileCheck().check_count("aten::mm", 8, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("prim::MMBatchSide", 1, exactly=True).check_not(
            "aten::mm"
        ).run(test_batch_mm.graph)

    def test_batch_mm_side_prohibited_mutation_uncommon_side(self):
        @torch.jit.script
        def test_batch_mm(n: int):
            A = torch.zeros((n, n))
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            torch.relu_(T1)
            result = {}
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            result["T9"] = torch.mm(A, T9)
            result["T10"] = torch.mm(A, T10)
            return result

        FileCheck().check_count("aten::mm", 10, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)

        FileCheck().check_count("aten::mm", 1, exactly=True).run(test_batch_mm.graph)
        FileCheck().check_count("prim::MMBatchSide", 1, exactly=True).run(
            test_batch_mm.graph
        )

    def test_batch_mm_side_prohibited_mutation_common_side(self):
        @torch.jit.script
        def test_batch_mm(n: int):
            A = torch.zeros((n, n))
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            torch.relu_(A)
            result = {}
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            result["T9"] = torch.mm(A, T9)
            result["T10"] = torch.mm(A, T10)
            return result

        FileCheck().check_count("aten::mm", 10, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 10, exactly=True).check_not(
            "prim::MMBatchSide"
        ).run(test_batch_mm.graph)
