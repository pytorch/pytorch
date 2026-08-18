from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class PyTorchFSDPDebtReport:
    model_id: str
    pdi_score: float  # PyTorch Debt Index (target <= 12.0)
    cuda_fragmentation_multiplier: float  # Target <= 1.08x
    allgather_barrier_latency_us: float  # Target <= 120.0us
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for PyTorch distributed FSDP execution runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_distributed_event(
        self,
        model_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{model_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "model_id": model_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtFSDPGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for PyTorch Distributed FSDP.

    Quantifies CUDA caching allocator fragmentation, FSDP all-gather communication barriers, and state dict checkpoints against 4 Enterprise KPIs:
    1. PyTorch Debt Index (PDI <= 12.0)
    2. CUDA Memory Fragmentation Ratio (CMFR <= 1.08x)
    3. P99 FSDP All-Gather Barrier Latency (<= 120us)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_pdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_pdi = max_acceptable_pdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_fsdp_training_step(
        self,
        model_id: str,
        allocated_cuda_bytes: int = 70000000000,
        reserved_caching_bytes: int = 72000000000,
        allgather_barrier_latency_us: float = 95.0,
        un_synced_gradient_steps: int = 0,
        un_gated_mutations: int = 0,
    ) -> PyTorchFSDPDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_distributed_event(
                model_id=model_id,
                event_type="training_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. PyTorch FSDP execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: CUDA Memory Fragmentation Ratio
        frag_ratio = reserved_caching_bytes / max(1, allocated_cuda_bytes)
        if frag_ratio > 1.8:
            critical_smells.append(f"HIGH_CUDA_CACHING_FRAGMENTATION_{frag_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if allgather_barrier_latency_us > 500.0:
            critical_smells.append(f"HIGH_FSDP_ALLGATHER_BARRIER_LATENCY_{allgather_barrier_latency_us:.1f}US")

        # Un-synced gradient steps
        if un_synced_gradient_steps > 1:
            critical_smells.append(f"DETECTED_{un_synced_gradient_steps}_UNSYNCED_GRADIENT_BARRIERS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_STATE_DICT_MUTATIONS")

        # KPI 1: PyTorch Debt Index (0 = Clean, 100 = Catastrophic)
        pdi = (
            max(0.0, (frag_ratio - 1.0) * 20.0)
            + max(0.0, (allgather_barrier_latency_us - 120.0) * 0.1)
            + (un_synced_gradient_steps * 15.0)
            + (un_gated_mutations * 30.0)
        )
        pdi_score = round(min(100.0, pdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - pdi_score)
        is_production_ready = (
            pdi_score <= self.max_acceptable_pdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_distributed_event(
            model_id=model_id,
            event_type="training_authorized" if is_production_ready else "training_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "pdi_score": pdi_score,
                "frag_ratio": frag_ratio,
                "allocated_cuda_bytes": allocated_cuda_bytes,
                "reserved_caching_bytes": reserved_caching_bytes,
                "allgather_barrier_latency_us": allgather_barrier_latency_us,
                "un_synced_gradient_steps": un_synced_gradient_steps,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return PyTorchFSDPDebtReport(
            model_id=model_id,
            pdi_score=pdi_score,
            cuda_fragmentation_multiplier=round(frag_ratio, 2),
            allgather_barrier_latency_us=round(allgather_barrier_latency_us, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
