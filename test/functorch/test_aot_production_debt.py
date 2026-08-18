# Copyright (c) Facebook, Inc. and its affiliates.
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
    "../../torch/_functorch/production_debt.py",
)
spec = importlib.util.spec_from_file_location("aot_autograd_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["aot_autograd_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtAOTAutogradGate = production_debt_mod.ProductionDebtAOTAutogradGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtAOTAutogradGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtAOTAutogradGate(
            never_equate_intent_to_approval=True,
            max_acceptable_adi=12.0,
        )

    def test_clean_joint_graph_passes_readiness(self) -> None:
        report = self.gate.evaluate_joint_graph(
            graph_id="aot_autograd_joint_forward_backward_graph",
            base_activation_bytes=16000000000,
            peak_stashed_bytes=16800000000,
            joint_partition_latency_ms=6.2,
            defunctional_view_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.adi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_joint_graph_fails_debt(self) -> None:
        report = self.gate.evaluate_joint_graph(
            graph_id="uncalibrated_aot_graph",
            base_activation_bytes=16000000000,
            peak_stashed_bytes=45000000000,  # 2.81x activation stashing sprawl
            joint_partition_latency_ms=85.0,  # High joint partition latency
            defunctional_view_stalls=3,  # 3 defunctional view stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.adi_score, 50.0)
        self.assertIn("HIGH_ACTIVATION_STASHING_SPRAWL_2.81X", report.critical_smells)
        self.assertIn("HIGH_JOINT_PARTITION_LATENCY_85.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_DEFUNCTIONAL_VIEW_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_AOT_GRAPH_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_joint_graph("graph-1")
        self.gate.evaluate_joint_graph("graph-2")
        self.gate.evaluate_joint_graph("graph-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
