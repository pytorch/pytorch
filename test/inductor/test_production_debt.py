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
    "../../torch/_inductor/production_debt.py",
)
spec = importlib.util.spec_from_file_location("inductor_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["inductor_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtInductorGate = production_debt_mod.ProductionDebtInductorGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtInductorGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtInductorGate(
            never_equate_intent_to_approval=True,
            max_acceptable_idi=12.0,
        )

    def test_clean_compiled_graph_passes_readiness(self) -> None:
        report = self.gate.evaluate_compiled_graph(
            graph_id="torch_compile_inductor_transformer_fx_graph",
            allocated_graph_bytes=12000000000,
            peak_buffer_bytes=12600000000,
            codegen_latency_ms=34.5,
            fusion_rejections=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.idi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_compiled_graph_fails_debt(self) -> None:
        report = self.gate.evaluate_compiled_graph(
            graph_id="uncalibrated_inductor_graph",
            allocated_graph_bytes=12000000000,
            peak_buffer_bytes=34000000000,  # 2.83x buffer aliasing sprawl
            codegen_latency_ms=190.0,  # High codegen compile latency
            fusion_rejections=3,  # 3 kernel fusion rejections
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.idi_score, 50.0)
        self.assertIn("HIGH_BUFFER_ALIASING_SPRAWL_2.83X", report.critical_smells)
        self.assertIn("HIGH_CODEGEN_COMPILE_LATENCY_190.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_KERNEL_FUSION_REJECTIONS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_GRAPH_LOWERING_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_compiled_graph("graph-1")
        self.gate.evaluate_compiled_graph("graph-2")
        self.gate.evaluate_compiled_graph("graph-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
