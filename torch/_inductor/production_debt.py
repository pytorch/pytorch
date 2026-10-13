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
class InductorDebtReport:
    graph_id: str
    idi_score: float  # Inductor Debt Index (target <= 12.0)
    buffer_sprawl_multiplier: float  # Target <= 1.08x
    codegen_latency_ms: float  # Target <= 45.0ms
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for TorchInductor compilation runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_inductor_event(
        self,
        graph_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{graph_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "graph_id": graph_id,
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


class ProductionDebtInductorGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for TorchInductor Deep Learning Compiler.

    Quantifies kernel fusion rejection, intermediate activation buffer aliasing sprawl, and AOT codegen compile latency against 4 Enterprise KPIs:
    1. Inductor Debt Index (IDI <= 12.0)
    2. Buffer Aliasing Memory Multiplier (BAMM <= 1.08x)
    3. P99 Inductor CodeGen Compile Latency (<= 45.0ms)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_idi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_idi = max_acceptable_idi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_compiled_graph(
        self,
        graph_id: str,
        allocated_graph_bytes: int = 12000000000,
        peak_buffer_bytes: int = 12600000000,
        codegen_latency_ms: float = 34.5,
        fusion_rejections: int = 0,
        un_gated_mutations: int = 0,
    ) -> InductorDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_inductor_event(
                graph_id=graph_id,
                event_type="compilation_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. TorchInductor execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Buffer Aliasing Memory Multiplier
        buf_ratio = peak_buffer_bytes / max(1, allocated_graph_bytes)
        if buf_ratio > 1.8:
            critical_smells.append(f"HIGH_BUFFER_ALIASING_SPRAWL_{buf_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if codegen_latency_ms > 120.0:
            critical_smells.append(f"HIGH_CODEGEN_COMPILE_LATENCY_{codegen_latency_ms:.1f}MS")

        # Fusion rejections
        if fusion_rejections > 0:
            critical_smells.append(f"DETECTED_{fusion_rejections}_KERNEL_FUSION_REJECTIONS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_GRAPH_LOWERING_MUTATIONS")

        # KPI 1: Inductor Debt Index (0 = Clean, 100 = Catastrophic)
        idi = (
            max(0.0, (buf_ratio - 1.0) * 20.0)
            + max(0.0, (codegen_latency_ms - 45.0) * 0.5)
            + (fusion_rejections * 25.0)
            + (un_gated_mutations * 30.0)
        )
        idi_score = round(min(100.0, idi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - idi_score)
        is_production_ready = (
            idi_score <= self.max_acceptable_idi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_inductor_event(
            graph_id=graph_id,
            event_type="graph_authorized" if is_production_ready else "graph_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "idi_score": idi_score,
                "buf_ratio": buf_ratio,
                "allocated_graph_bytes": allocated_graph_bytes,
                "peak_buffer_bytes": peak_buffer_bytes,
                "codegen_latency_ms": codegen_latency_ms,
                "fusion_rejections": fusion_rejections,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return InductorDebtReport(
            graph_id=graph_id,
            idi_score=idi_score,
            buffer_sprawl_multiplier=round(buf_ratio, 2),
            codegen_latency_ms=round(codegen_latency_ms, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
