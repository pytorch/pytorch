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
    "../../torch/_dynamo/production_debt.py",
)
spec = importlib.util.spec_from_file_location("dynamo_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["dynamo_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtDynamoGate = production_debt_mod.ProductionDebtDynamoGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtDynamoGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtDynamoGate(
            never_equate_intent_to_approval=True,
            max_acceptable_ddi=12.0,
        )

    def test_clean_evaluated_frame_passes_readiness(self) -> None:
        report = self.gate.evaluate_evaluated_frame(
            frame_id="torch_compile_dynamo_forward_frame",
            allocated_graph_count=1,
            actual_subgraph_count=1,
            frame_hook_latency_ms=5.8,
            guard_invalidation_recompilations=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.ddi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_evaluated_frame_fails_debt(self) -> None:
        report = self.gate.evaluate_evaluated_frame(
            frame_id="uncalibrated_dynamo_frame",
            allocated_graph_count=1,
            actual_subgraph_count=6,  # 6.0x graph break fragmentation sprawl
            frame_hook_latency_ms=85.0,  # High frame hook latency
            guard_invalidation_recompilations=3,  # 3 guard invalidation recompilations
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.ddi_score, 50.0)
        self.assertIn("HIGH_GRAPH_BREAK_FRAGMENTATION_6.00X", report.critical_smells)
        self.assertIn("HIGH_FRAME_HOOK_LATENCY_85.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_GUARD_INVALIDATION_RECOMPILATIONS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_FRAME_EVAL_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_evaluated_frame("frame-1")
        self.gate.evaluate_evaluated_frame("frame-2")
        self.gate.evaluate_evaluated_frame("frame-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
