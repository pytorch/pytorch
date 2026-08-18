# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../torch/csrc/production_debt.py",
)
spec = importlib.util.spec_from_file_location("pytorch_aten_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["pytorch_aten_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtATenGate = production_debt_mod.ProductionDebtATenGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtATenGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtATenGate(
            never_equate_intent_to_approval=True,
            max_acceptable_adi=12.0,
        )

    def test_clean_tensor_operation_passes_readiness(self) -> None:
        report = self.gate.evaluate_tensor_operation(
            tensor_id="aten_matmul_cuda_c10_allocated",
            allocated_c10_bytes=32000000000,
            peak_fragmented_bytes=33600000000,
            tensor_alloc_latency_ms=3.8,
            dispatch_fallback_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.adi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_tensor_operation_fails_debt(self) -> None:
        report = self.gate.evaluate_tensor_operation(
            tensor_id="uncalibrated_aten_tensor_graph",
            allocated_c10_bytes=32000000000,
            peak_fragmented_bytes=90000000000,  # 2.81x C10 fragmentation sprawl
            tensor_alloc_latency_ms=45.0,  # High alloc latency
            dispatch_fallback_stalls=3,  # 3 dispatch fallback stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.adi_score, 50.0)
        self.assertIn("HIGH_C10_ALLOCATOR_FRAGMENTATION_2.81X", report.critical_smells)
        self.assertIn("HIGH_TENSOR_ALLOC_LATENCY_45.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_ATEN_DISPATCH_FALLBACK_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_INPLACE_AUTOGRAD_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_tensor_operation("tensor-1")
        self.gate.evaluate_tensor_operation("tensor-2")
        self.gate.evaluate_tensor_operation("tensor-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
