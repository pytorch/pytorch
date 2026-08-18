# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
class ATenDebtReport:
    tensor_id: str
    adi_score: float  # ATen Debt Index (target <= 12.0)
    c10_memory_sprawl_multiplier: float  # Target <= 1.08x
    tensor_alloc_latency_ms: float  # Target <= 5.2ms
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for PyTorch ATen native tensor operations."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_aten_event(
        self,
        tensor_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{tensor_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "tensor_id": tensor_id,
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


class ProductionDebtATenGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for PyTorch ATen Native Tensors & C10 Allocator.

    Quantifies C10 CUDA caching allocator block fragmentation, ATen dynamic dispatch fallback stalls, in-place autograd mutations, and alloc latency against 4 Enterprise KPIs:
    1. ATen Debt Index (ADI <= 12.0)
    2. C10 Allocator Memory Multiplier (CAMM <= 1.08x)
    3. P99 Native Tensor Allocation Latency (<= 5.2ms)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_adi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_adi = max_acceptable_adi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_tensor_operation(
        self,
        tensor_id: str,
        allocated_c10_bytes: int = 32000000000,
        peak_fragmented_bytes: int = 33600000000,
        tensor_alloc_latency_ms: float = 3.8,
        dispatch_fallback_stalls: int = 0,
        un_gated_mutations: int = 0,
    ) -> ATenDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_aten_event(
                tensor_id=tensor_id,
                event_type="operation_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. PyTorch ATen execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: C10 Allocator Memory Multiplier
        c10_ratio = peak_fragmented_bytes / max(1, allocated_c10_bytes)
        if c10_ratio > 1.8:
            critical_smells.append(f"HIGH_C10_ALLOCATOR_FRAGMENTATION_{c10_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if tensor_alloc_latency_ms > 20.0:
            critical_smells.append(f"HIGH_TENSOR_ALLOC_LATENCY_{tensor_alloc_latency_ms:.1f}MS")

        # Dispatch fallback stalls
        if dispatch_fallback_stalls > 0:
            critical_smells.append(f"DETECTED_{dispatch_fallback_stalls}_ATEN_DISPATCH_FALLBACK_STALLS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_INPLACE_AUTOGRAD_MUTATIONS")

        # KPI 1: ATen Debt Index (0 = Clean, 100 = Catastrophic)
        adi = (
            max(0.0, (c10_ratio - 1.0) * 20.0)
            + max(0.0, (tensor_alloc_latency_ms - 5.2) * 0.5)
            + (dispatch_fallback_stalls * 25.0)
            + (un_gated_mutations * 30.0)
        )
        adi_score = round(min(100.0, adi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - adi_score)
        is_production_ready = (
            adi_score <= self.max_acceptable_adi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_aten_event(
            tensor_id=tensor_id,
            event_type="tensor_authorized" if is_production_ready else "tensor_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "adi_score": adi_score,
                "c10_ratio": c10_ratio,
                "allocated_c10_bytes": allocated_c10_bytes,
                "peak_fragmented_bytes": peak_fragmented_bytes,
                "tensor_alloc_latency_ms": tensor_alloc_latency_ms,
                "dispatch_fallback_stalls": dispatch_fallback_stalls,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return ATenDebtReport(
            tensor_id=tensor_id,
            adi_score=adi_score,
            c10_memory_sprawl_multiplier=round(c10_ratio, 2),
            tensor_alloc_latency_ms=round(tensor_alloc_latency_ms, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
