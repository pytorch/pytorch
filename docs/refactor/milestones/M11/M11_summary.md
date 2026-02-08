# 📌 Milestone Summary — M11: Distributed Protocol Version Guardrail

**Project:** PyTorch Refactoring Program  
**Phase:** Phase 2 — Test Infrastructure  
**Milestone:** M11 — Distributed Protocol Version Guardrail  
**Timeframe:** 2026-02-08 → OPEN  
**Status:** 🟡 **OPEN — Awaiting CI Approval**  
**Baseline:** `5933293e0b3` (M08 closeout)  
**Refactor Posture:** Behavior-Preserving

---

## 1. Milestone Objective

Introduce an **explicit, machine-checkable protocol version guardrail** for `torch.distributed` to prevent **silent cross-version incompatibility** between workers during distributed training.

### Problem Addressed

| Issue | Description |
|-------|-------------|
| INV-030 | Collective Wire Protocol Compatibility was 🔴 MISSING |
| Risk I06 | Implicit Distributed Protocol — cross-version mismatches fail late and opaquely |
| Production Impact | Rolling updates in clusters can mix PyTorch versions |

### What Would Remain Unsafe Without This Refactor

- Cross-version incompatibilities would continue to cause silent corruption or late, opaque failures
- No explicit signal that distributed versions are compatible
- INV-030 would remain unverified

---

## 2. Scope Definition

### In Scope

| Component | Details |
|-----------|---------|
| Modules touched | `torch/distributed/_protocol_version.py` (NEW), `torch/distributed/distributed_c10d.py` |
| Entrypoints affected | `init_process_group()` — internal initialization path |
| Contracts | Protocol version exchange via rendezvous Store |
| CI workflows | None modified (new test file only) |
| Documentation | M11 milestone artifacts |

### Out of Scope

| Exclusion | Rationale |
|-----------|-----------|
| RPC protocol version (INV-031) | Explicitly deferred per locked decisions (D11) |
| MPI backend enforcement | No store path available; documented exclusion |
| Backend protocol redesign | Not a redesign; minimal guardrail only |
| Wire-format changes | No changes beyond version handshake |
| Performance optimization | Not applicable |
| DDP/FSDP logic | Untouched |

### Scope Changes During Execution

None. Scope remained as defined in M11_plan.md throughout implementation.

---

## 3. Refactor Classification

### Change Type

**Semantic Refactor** — Logic added without intended behavior change for matching versions.

- **Matching versions:** Behavior identical to baseline
- **Mismatched versions:** New early `RuntimeError` (intentional guardrail behavior)

### Observability

| Surface | Observable Change |
|---------|-------------------|
| Matching-version distributed training | **None** |
| Mismatched-version distributed training | **New:** Early `RuntimeError` with clear message |
| API responses | **None** |
| CLI output | **None** |
| File formats | **None** |

---

## 4. Work Executed

### Implementation Summary

| Action | Details |
|--------|---------|
| Created `_protocol_version.py` | 60 lines — `PROTOCOL_VERSION = 1`, `get_protocol_version()` with env override |
| Modified `distributed_c10d.py` | 87 lines added — `_check_protocol_version()` helper, hook in `init_process_group` |
| Created test file | 274 lines — 6 test cases (3 unit, 3 multi-process) |
| Created milestone docs | M11_plan.md, M11_toolcalls.md, M11_run1.md |

### Key Design Decisions (Per Locked Decisions)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| D1. Version representation | Single integer | Simplest guardrail-first posture |
| D2. Initial value | `1` | Baseline for future increments |
| D3. Compatibility rule | Exact match only | Simplest; ranges are future work |
| D4. Exchange mechanism | Store-based | Works before backend init; uses proven pattern |
| D5. MPI handling | Excluded | No store path; documented |
| D6. Exception type | `RuntimeError` | No new public exception |
| D8. Test override | Env var | `TORCH_DISTRIBUTED_PROTOCOL_VERSION_OVERRIDE` |
| D9. Test style | `spawn` | Windows compatible |
| D11. RPC scope | Deferred | Collectives only for M11 |

### Files Changed

| File | Lines | Type |
|------|-------|------|
| `torch/distributed/_protocol_version.py` | +60 | NEW |
| `torch/distributed/distributed_c10d.py` | +87 | MODIFIED |
| `test/distributed/test_protocol_version.py` | +274 | NEW |
| `docs/refactor/milestones/M11/*` | ~700 | NEW |

### Migration Steps

None required. New module is internal. No existing code paths changed for matching-version scenarios.

---

## 5. Invariants & Compatibility

### Declared Invariants

| Invariant | Description | Verification Method |
|-----------|-------------|---------------------|
| Distributed correctness | Matching-version runs behave identically | Existing distributed tests |
| Failure clarity | Mismatches fail early with clear message | New mismatch test |
| Compatibility | No existing code breaks with same version | All distributed tests |
| Isolation | No protocol version logic in non-distributed paths | Code review |

### Compatibility Notes

| Aspect | Status | Evidence |
|--------|--------|----------|
| Backward compatibility preserved | ✅ Yes | Matching versions unchanged |
| Breaking changes introduced | ❌ No | Only new failure mode for mismatches |
| Deprecations introduced | ❌ No | No deprecations |

---

## 6. Validation & Evidence

### Status: ⏳ PENDING

CI workflows have not executed due to administrative blockers (CLA signature, fork PR approval).

### Expected Validation

| Evidence Type | Tool/Workflow | Expected Result | Notes |
|---------------|---------------|-----------------|-------|
| Unit tests | pytest | PASS | 3 unit tests in new file |
| Integration tests | pytest + spawn | PASS | 3 multi-process tests |
| Existing distributed tests | CI suite | PASS | No regression expected |
| Lint | Lint workflow | PASS | py_compile verified locally |
| BC check | BC Lint | PASS | No public API changes |

### Failures Encountered

| Failure | Resolution |
|---------|------------|
| EasyCLA | ⏳ Awaiting CLA signature |
| Workflow approval | ⏳ Awaiting maintainer action |

### Evidence Gaps

- **CI execution:** Cannot verify until workflows approved
- **Coverage metrics:** Cannot measure until CI runs
- **Distributed test interaction:** Cannot verify until CI runs

---

## 7. CI / Automation Impact

### Workflows Affected

| Workflow | Change |
|----------|--------|
| Distributed tests | Will run new test file |
| Lint | Will check new files |
| BC Lint | Will analyze for compatibility |

### Checks Added/Removed/Reclassified

- **Added:** `test/distributed/test_protocol_version.py` (6 new tests)
- **Removed:** None
- **Reclassified:** None

### Enforcement Changes

- **Unchanged** — No CI enforcement changes

### CI Assessment

| Question | Answer |
|----------|--------|
| Did CI block incorrect changes? | ⏳ PENDING |
| Did CI validate correct changes? | ⏳ PENDING |
| Did CI fail to observe relevant risk? | ⏳ PENDING |

---

## 8. Issues, Exceptions, and Guardrails

### Issues Encountered

| Issue | Root Cause | Resolution | Tracking | Guardrail |
|-------|------------|------------|----------|-----------|
| CLA not signed | Fork PR requirement | Sign via EasyCLA link | PR #174577 | N/A (administrative) |
| Workflows need approval | Fork PR security | Await maintainer | PR #174577 | N/A (administrative) |

### New Guardrails Added

| Guardrail | Location | Purpose |
|-----------|----------|---------|
| `_check_protocol_version()` | `distributed_c10d.py` | Prevent silent cross-version incompatibility |
| Protocol version tests | `test_protocol_version.py` | Verify guardrail works |

---

## 9. Deferred Work

| Item | Why Deferred | Pre-existed? | Status Changed? |
|------|--------------|--------------|-----------------|
| INV-031 (RPC Serialization) | Out of scope per D11 | Yes (from M00 audit) | No |
| MPI protocol version enforcement | No store path available | No (new) | N/A |
| Forward/backward version ranges | Exact match is simpler | No (design choice) | N/A |

---

## 10. Governance Outcomes

### What Is Now Provably True

| Before M11 | After M11 |
|------------|-----------|
| INV-030 marked 🔴 MISSING | INV-030 has verification (pending CI) |
| Cross-version issues fail silently/late | Cross-version issues fail early with clear message |
| No protocol version enforcement | Explicit version check during init |

### Governance Improvements

1. **Invariant now has verification** — INV-030 transitions from theory to practice
2. **Clear failure semantics** — Error message includes versions, backend, and guidance
3. **Test coverage established** — Both success and failure paths tested

---

## 11. Exit Criteria Evaluation

| Criterion | Status | Evidence |
|-----------|--------|----------|
| New mismatch test fails if version check removed | ⏳ PENDING | Test designed to verify this |
| CI distributed tests pass | ⏳ PENDING | Awaiting CI execution |
| No distributed behavior change for matching versions | ⏳ PENDING | Code review confirms; CI will verify |
| Audit verdict: Safe, guardrail added, no drift | ⏳ PENDING | Audit drafted, awaiting CI |

---

## 12. Final Verdict

**Verdict:** 🟡 **OPEN — Awaiting CI Approval**

Implementation is complete and appears structurally correct. Final verdict cannot be issued until CI workflows execute and confirm:
1. All existing distributed tests pass
2. New protocol version tests pass
3. No regressions introduced

---

## 13. Authorized Next Step

**Authorized:**
1. Sign EasyCLA to resolve CLA blocker
2. Obtain maintainer approval for workflow execution
3. Analyze CI results in `M11_run2.md`
4. Issue final verdict and close milestone (with permission)

**Not Authorized:**
- Merge without CI verification
- Close milestone without explicit permission
- Proceed to M12 until M11 is closed

---

## 14. Canonical References

| Type | Reference |
|------|-----------|
| PR | [#174577](https://github.com/pytorch/pytorch/pull/174577) |
| Branch | `m11-protocol-version-guardrail` |
| Commits | `d76b31b8f21`, `fdc546b7bc2`, `ea8c960829a`, `0655d7f9ed5`, `f4e3c86ea33` |
| Baseline | `5933293e0b3` (M08 closeout) |
| Plan | [`M11_plan.md`](M11_plan.md) |
| Toolcalls | [`M11_toolcalls.md`](M11_toolcalls.md) |
| CI Analysis | [`M11_run1.md`](M11_run1.md) |
| Audit | [`M11_audit.md`](M11_audit.md) |

---

**Summary Status:** 🟡 **DRAFT — PENDING CI**  
**Author:** Cursor (AI Agent)  
**Date:** 2026-02-08  
**Next Update:** After CI workflows complete

