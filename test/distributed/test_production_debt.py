import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../torch/distributed/production_debt.py",
)
spec = importlib.util.spec_from_file_location("pytorch_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["pytorch_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtFSDPGate = production_debt_mod.ProductionDebtFSDPGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtFSDPGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtFSDPGate(
            never_equate_intent_to_approval=True,
            max_acceptable_pdi=12.0,
        )

    def test_clean_fsdp_training_passes_readiness(self) -> None:
        report = self.gate.evaluate_fsdp_training_step(
            model_id="meta-llama/Llama-3-405B-FSDP2",
            allocated_cuda_bytes=70000000000,
            reserved_caching_bytes=72000000000,
            allgather_barrier_latency_us=95.0,
            un_synced_gradient_steps=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.pdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_fsdp_training_fails_debt(self) -> None:
        report = self.gate.evaluate_fsdp_training_step(
            model_id="uncalibrated_fsdp_run",
            allocated_cuda_bytes=50000000000,
            reserved_caching_bytes=130000000000,  # High memory fragmentation (2.6x)
            allgather_barrier_latency_us=650.0,  # High barrier latency
            un_synced_gradient_steps=3,  # 3 un-synced barriers
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.pdi_score, 50.0)
        self.assertIn("HIGH_CUDA_CACHING_FRAGMENTATION_2.60X", report.critical_smells)
        self.assertIn("HIGH_FSDP_ALLGATHER_BARRIER_LATENCY_650.0US", report.critical_smells)
        self.assertIn("DETECTED_3_UNSYNCED_GRADIENT_BARRIERS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_STATE_DICT_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_fsdp_training_step("model-1")
        self.gate.evaluate_fsdp_training_step("model-2")
        self.gate.evaluate_fsdp_training_step("model-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
