# PyTorch Refactoring Program — Living Log

**Repository:** pytorch/pytorch (fork: m-cahill/pytorch)  
**Program Start:** 2026-02-08  
**Baseline Commit:** c5f1d40892292ef79cb583a8df00ceb1c8812a12  
**Current Phase:** Phase 1 In Progress (M03 Complete)

---

## Project Context — AI-Native Refactoring Program

This project applies an **AI-native, audit-first refactoring methodology** to an existing, large-scale production codebase.

The program is **not greenfield**. The default posture is **behavior preservation**, with all changes evaluated against explicitly declared **invariants**, **blast-radius awareness**, and **verifiable evidence** (tests, CI signals, documented gaps).

Work proceeds in **small, non-overlapping milestones**, each with:

* a clearly stated intent and scope boundary,
* explicit non-goals,
* defined verification methods,
* and a safe rollback plan.

A completed **Phase 0 (M00–M02) foundation** establishes the authoritative starting point:

* system surfaces,
* invariants,
* risks,
* CI posture,
* security and supply-chain considerations,
* accepted evidence gaps,
* and this governance framework.

All subsequent milestones (M03+) must:

* reference the Phase 0 audit,
* follow the governance rules in this document,
* avoid implicit behavior changes,
* treat CI as a truth signal (not a suggestion),
* and produce objective proof that correctness is preserved.

Cursor is expected to operate as a **refactoring assistant under governance**, not as a code generator or optimizer. When in doubt, prefer **restraint, documentation, and explicit deferral** over speculative fixes.

---

## Repository Facts (Baseline)

| Metric | Value | Source |
|--------|-------|--------|
| **Total Tracked Files** | 20,440 | `git ls-files` (2026-02-08) |
| **Python Files** | 4,216 | File count |
| **C/C++ Files** | 4,403 | File count |
| **CUDA Files** | 345 | File count |
| **Python Test Files** | 1,353+ | `test/` directory |
| **C++ Test Files** | 279+ | `test/cpp/`, `c10/test/`, etc. |
| **CI Workflows** | 130+ | `.github/workflows/` |
| **Estimated Test Coverage** | 70-75% | CI inference |
| **Python Version** | 3.12.10 | Development environment |

---

# Governance Framework

This section defines the rules, processes, and constraints that govern all refactoring work. It is the authoritative reference for how the program operates.

---

## Governance Model

### Document Hierarchy

The refactoring program operates under a three-tier authority hierarchy:

1. **Audit Pack** (`docs/refactor/audit/`) — Immutable baseline facts
   - Captures the state of the codebase at commit c5f1d40 (2026-02-08)
   - Documents what was observed, not what should be
   - Cannot be modified after Phase 0 closeout (except to fix factual errors)

2. **REFACTOR.md** (this document) — Authoritative living governance
   - Defines how the refactoring program operates
   - Updated after each milestone
   - Source of truth for current procedures and constraints

3. **Milestone Documents** (`docs/refactor/milestones/MNN/`) — Local execution detail
   - Capture plans, decisions, and outcomes for specific milestones
   - Subordinate to REFACTOR.md for procedural questions

### Conflict Resolution

- **Factual conflicts about baseline state:** The audit pack wins.
- **Procedural or governance questions:** REFACTOR.md is authoritative.
- **Milestone-specific execution questions:** Consult the milestone's plan and summary.

### Key References

- **Baseline Audit:** [`docs/refactor/audit/BASELINE_AUDIT.md`](docs/refactor/audit/BASELINE_AUDIT.md)
- **System Surfaces:** [`docs/refactor/audit/SYSTEM_SURFACES.md`](docs/refactor/audit/SYSTEM_SURFACES.md)
- **Invariants Catalog:** [`docs/refactor/audit/INVARIANTS_CATALOG.md`](docs/refactor/audit/INVARIANTS_CATALOG.md)
- **Milestone Roadmap:** [`docs/refactor/audit/REFACTOR_PHASE_MAP.md`](docs/refactor/audit/REFACTOR_PHASE_MAP.md)
- **Next Actions:** [`docs/refactor/audit/NEXT_ACTIONS.md`](docs/refactor/audit/NEXT_ACTIONS.md)
- **Audit Pack Index:** [`docs/refactor/audit/README.md`](docs/refactor/audit/README.md)

---

## Milestone Lifecycle

Every milestone follows a defined lifecycle from opening to closeout.

### Opening a Milestone

A milestone may be opened when:

1. All blocking dependencies (prior milestones) are complete
2. A plan document exists (`MNN_plan.md`) with:
   - Intent and scope boundaries
   - Invariants to protect or verify
   - Verification plan
   - Rollback strategy
3. The milestone has been explicitly authorized to begin

### Required Artifacts

Each milestone must produce the following artifacts:

| Artifact | Purpose | When Created |
|----------|---------|--------------|
| `MNN_plan.md` | Scope, intent, verification approach | Before execution |
| `MNN_toolcalls.md` | Tool invocation log for recovery | During execution |
| `MNN_runN.md` | CI analysis (if applicable) | After each CI run |
| `MNN_audit.md` | Compliance and verification report | At closeout |
| `MNN_summary.md` | Executive summary of outcomes | At closeout |

### Execution Rules

During execution:

1. Log all tool calls **before** execution (not after)
2. Stay within declared scope boundaries
3. Stop and ask for confirmation when encountering:
   - Scope ambiguity
   - Unexpected failures
   - Potential invariant violations
4. Update REFACTOR.md milestone entry with progress

### Stopping, Deferring, or Aborting

- **Stop:** Pause execution, document current state, wait for guidance
- **Defer:** Move work to a future milestone, document reason and target
- **Abort:** Rollback changes, document why, close milestone as cancelled

A milestone should be stopped when:

- Scope creep is detected
- Invariant violations are discovered
- Blocking issues arise that require governance decisions

### Closeout

A milestone is closed when:

1. All deliverables are complete
2. All verification methods have passed
3. REFACTOR.md is updated with milestone outcome
4. Audit and summary documents are created
5. Explicit closeout permission is granted

---

## Change Classes

All changes fall into one of the following classes. Each class has different governance requirements.

### Documentation-Only

**Definition:** Changes to markdown, comments, or non-executable files.

**Requirements:**
- No verification required beyond review
- No CI enforcement (unless docs are tested)
- May be closed without explicit permission

**Examples:** Audit pack creation, README updates, governance additions

### Verification-Only

**Definition:** Addition of tests, linters, or verification tooling that do not modify production behavior.

**Requirements:**
- New tests must pass
- CI must remain green
- No production code modified

**Examples:** M01 (import smoke test), adding new test files

### Mechanical Refactor

**Definition:** Structural changes that preserve behavior exactly.

**Requirements:**
- All affected invariants must be verified
- CI must pass
- Rollback plan must exist
- Evidence of equivalence required

**Examples:** Renaming, moving files, code reorganization

### Behavioral Change

**Definition:** Any change that could alter runtime behavior, output, or API semantics.

**Requirements:**
- **Explicitly disallowed** unless approved through governance
- Requires documented justification
- Requires extended verification (beyond CI)
- Requires explicit approval before merge

**Examples:** Bug fixes, feature additions, performance changes

### CI Changes

Two sub-classes with different risk profiles:

**CI Wiring (Low Risk):**
- Adding new workflows
- Adjusting triggers
- Improving feedback

**CI Weakening (High Risk):**
- Adding `continue-on-error`
- Removing required checks
- Loosening constraints

CI weakening requires explicit approval and documented justification.

---

## Invariant Handling

Invariants are properties that must remain true across all refactoring work.

### Introducing New Invariants

New invariants may be proposed when:

1. An existing property is discovered that should be protected
2. A new verification capability is added
3. A risk is identified that requires explicit protection

New invariants must be documented in the Invariants Catalog with:
- Unique identifier (INV-NNN)
- Description
- Priority (P0-P3)
- Verification method
- Blast radius if violated

### Verifying Invariants

Before closing any milestone that touches production code:

1. Identify all invariants in scope
2. Run or confirm verification method for each
3. Document verification outcome in milestone summary

### Handling Violations

If an invariant violation is detected:

1. **Stop immediately** — Do not proceed
2. **Document the violation** — What, where, how discovered
3. **Assess blast radius** — What else might be affected
4. **Report and wait** — Do not attempt to fix without approval

### Proposing Invariant Changes

Existing invariants may be modified or retired when:

1. The protected property is no longer relevant
2. A better verification method exists
3. The invariant is demonstrably too strict

Changes to invariants require explicit approval and documentation.

---

## Deferral & Risk Registry Rules

### What Qualifies as a Deferral

Work is deferred when:

1. It is out of scope for the current milestone
2. It requires capabilities not yet available
3. It introduces risk that should be isolated
4. It would cause scope creep

### Required Metadata for Deferrals

Every deferral must document:

| Field | Description |
|-------|-------------|
| **What** | Description of deferred work |
| **Why** | Reason for deferral |
| **Risk** | What remains unsafe without this work |
| **Revisit** | Target milestone or condition for revisiting |

### Where Deferrals Live

- **Current milestone:** In `MNN_summary.md` under "Deferred Work"
- **Program-level:** Referenced in REFACTOR.md Active Risks table if P0-P1

### Prohibition on Silent Deferral

**Silent deferral is not permitted.**

If work is not completed, it must be explicitly documented as deferred with the required metadata. Omitting work without documentation is a governance violation.

---

## AI Agent Operating Rules

These rules govern how Cursor and other AI agents operate within this program.

### Expected Posture

1. **Operate as assistant, not author** — Follow governance, don't invent it
2. **Prefer restraint over speculation** — When uncertain, stop and ask
3. **Document before acting** — Log tool calls before execution
4. **Verify before claiming success** — Run tests, check CI, confirm outcomes

### When to Stop and Ask

Stop and request confirmation when:

1. Scope boundaries are unclear
2. A change might affect invariants not explicitly in scope
3. CI fails unexpectedly
4. A decision has governance implications
5. Work would exceed estimated effort by >50%

### Prohibited Actions

AI agents must NOT:

1. Merge PRs without explicit permission
2. Push to main without explicit permission
3. Modify audit pack documents (except for typo fixes)
4. Make behavioral changes without approval
5. Skip tool logging
6. Ignore failing CI

### Recovery Protocol

If a session is interrupted:

1. Read the milestone's `MNN_toolcalls.md` for last recorded action
2. Report: what was being done, what step was next, completion status
3. Wait for confirmation before resuming

---

## Canonical Milestone Template

All milestones should follow this structure. This template may be extracted to a separate file if it grows large.

```markdown
# MNN_plan — [Milestone Name]

## Intent / Target

[One paragraph: What does this milestone accomplish? What would remain unsafe without it?]

## Scope Boundaries

**In Scope:**
- [Specific deliverables]

**Out of Scope:**
- [Explicit non-goals]

## Invariants

[Which invariants from INVARIANTS_CATALOG.md are protected or verified?]

## Verification Plan

[How will success be measured? Tests, CI, manual verification?]

## Implementation Steps

[Ordered list of steps, each reversible]

## Risk & Rollback

**Risks:** [What could go wrong]
**Mitigation:** [How to reduce risk]
**Rollback:** [How to undo if needed]

## Deliverables

[List of files/artifacts that will be created or modified]

## Definition of Done

[Explicit checklist for closeout]
```

---

## Phase Boundaries

The refactoring program is organized into phases, each with a specific focus.

### Phase 0: Foundation (M00–M02) ✅ COMPLETE

**Focus:** Establish the baseline and governance framework.

| Milestone | Purpose | Status |
|-----------|---------|--------|
| M00 | Baseline Audit — Document what exists | ✅ Complete |
| M01 | Import Smoke Test — First executable verification | ✅ Complete |
| M02 | Governance Hardening — Formalize how work proceeds | ✅ Complete |

**Phase 0 Outcome:** 
- Audit pack is immutable baseline
- REFACTOR.md is authoritative governance surface
- First verification tool operational
- Program ready for execution-focused work

**Transition:** Phase 1 begins only after governance is explicit (M02 complete).

### Phase 1: CI Health & Guardrails (M03–M10)

**Focus:** Improve CI reliability, eliminate silent failures, secure supply chain.

See [`docs/refactor/audit/REFACTOR_PHASE_MAP.md`](docs/refactor/audit/REFACTOR_PHASE_MAP.md) for full plan.

### Phase 2: Test Infrastructure (M11–M14)

**Focus:** Close critical testing gaps, add protocol versioning.

### Phase 3: Verification Infrastructure (M15–M19)

**Focus:** Add verification for critical invariants (ABI, determinism, state dict).

### Phase 4: Structural Refactors (M20+)

**Focus:** Architectural improvements, enabled by safety infrastructure from Phases 1–3.

---

# Milestone History

---

## Phase 0: Foundation ✅ COMPLETE

### M00 — Baseline Audit ✅ COMPLETE

**Status:** 🔒 Closed and Locked  
**Date:** 2026-02-08  
**Mode:** BASELINE RESET  
**Scope:** Documentation & audit only

**Intent:**
Establish an authoritative, evidence-based baseline for the PyTorch refactoring program through comprehensive audit of codebase, CI infrastructure, testing, security posture, and module boundaries.

**Deliverables:**
- ✅ 11 audit documents (~15,000 lines) in `docs/refactor/audit/`
- ✅ Top 7 issues identified and triaged (P0-P2)
- ✅ 22 milestones planned (M00-M22)
- ✅ 80+ invariants cataloged with verification methods
- ✅ Phase 0 non-goals explicitly stated
- ✅ 7 evidence gaps acknowledged and documented
- ✅ Governance structure established

**Key Findings:**
- **Wins:** Production-grade CI (130+ workflows), comprehensive tests (1,600+ files), modern dependency management, modular architecture
- **Top Risks:** No local build environment (CI-only), 130+ workflows with potential silent failures, mixed action pinning (supply chain risk), large third-party footprint
- **Coverage:** 70-75% estimated (actual metrics deferred to M01+)
- **Security Score:** 5.2/10 baseline (target 7/10 after Phase 1)

**Non-Goals (Honored):**
- ❌ No behavior changes
- ❌ No refactors
- ❌ No formatting sweeps
- ❌ No dependency upgrades
- ❌ No CI enforcement changes
- ❌ No performance work
- ❌ No test rewrites

**Evidence Gaps (Accepted):**
- No local build/test execution (per AGENTS.md constraints)
- No runtime coverage metrics (estimated from structure)
- No fork-level CI execution (observational only)
- No SBOM generated (planned for M08)
- CI behavior inferred from workflow structure
- No dependency version audit (planned for M08-M09)
- No ABI compatibility baseline (planned for M15)

**Outcome:**
- Baseline locked at commit c5f1d40
- Audit pack is now immutable reference
- Ready to proceed to M01 (Import Smoke Test)

**Verification:**
- ✅ All scope boundaries honored (documentation-only)
- ✅ All invariants preserved (zero behavioral impact)
- ✅ All quality gates passed
- ✅ Third-party reviewable

**Closeout Artifacts:**
- [`docs/refactor/milestones/M00/M00_summary.md`](docs/refactor/milestones/M00/M00_summary.md)
- [`docs/refactor/milestones/M00/M00_audit.md`](docs/refactor/milestones/M00/M00_audit.md)
- [`docs/refactor/milestones/M00/M00_closeout.md`](docs/refactor/milestones/M00/M00_closeout.md)

**Effort:** ~40 hours (audit pack creation + closeout)

---

### M01 — Import Smoke Test ✅ COMPLETE

**Status:** 🔒 Closed and Locked  
**Date:** 2026-02-08  
**Effort:** 4 hours  
**Merge:** PR #1 → `d72fd100459`

**Outcome:**
Created static import-graph checker (`tools/refactor/import_smoke_static.py`) that validates Python import integrity without C++ build. First executable verification surface for the refactoring program.

**Deliverables:**
- `tools/refactor/import_smoke_static.py` — Static AST-based import analyzer
- `test/test_import_smoke_static.py` — Test suite (8 tests)
- `.github/workflows/refactor-smoke.yml` — Isolated CI workflow

**Invariant Verified:**
- INV-050: Import Path Stability (2,372 files, 21,254 imports, 0 unresolved)

**CI Progression:**
- Run 1: FAILED (pytest not installed) → Fixed: use stdlib unittest
- Run 2: FAILED (test/ not a package) → Fixed: run file directly
- Run 3: SUCCESS

**Closeout Artifacts:**
- [`M01_summary.md`](docs/refactor/milestones/M01/M01_summary.md)
- [`M01_audit.md`](docs/refactor/milestones/M01/M01_audit.md)
- [`M01_run1.md`](docs/refactor/milestones/M01/M01_run1.md)

---

### M02 — Governance & Living-Log Hardening ✅ COMPLETE

**Status:** 🔒 Closed and Locked  
**Date:** 2026-02-08  
**Effort:** 2 hours  
**Change Class:** Documentation-Only

**Intent:**
Harden and formalize the governance spine of the refactoring program so that all future milestones operate under explicit, enforceable rules rather than convention.

**Deliverables:**
- ✅ REFACTOR.md updated with explicit governance sections:
  - Governance Model (authority hierarchy, conflict resolution)
  - Milestone Lifecycle (opening, execution, closeout rules)
  - Change Classes (documentation, verification, mechanical, behavioral)
  - Invariant Handling (introduction, verification, violation handling)
  - Deferral & Risk Registry Rules
  - AI Agent Operating Rules
- ✅ Canonical Milestone Template added
- ✅ Phase Boundaries clarified (Phase 0 = M00–M02)
- ✅ Placeholder sections merged into governance framework

**Non-Goals (Honored):**
- ❌ No code changes
- ❌ No test changes
- ❌ No CI changes
- ❌ No new tracking artifacts (registry tables, etc.)
- ❌ No rewriting of M00/M01 conclusions

**Outcome:**
- Governance framework is now explicit and auditable
- AI agents can answer "how do I safely work here?" without inference
- Phase 0 is complete; Phase 1 (M03+) may begin

**Verification:**
- ✅ All governance sections present and readable
- ✅ No contradictions with audit pack or prior milestones
- ✅ No non-documentation files modified

**Closeout Artifacts:**
- [`docs/refactor/milestones/M02/M02_plan.md`](docs/refactor/milestones/M02/M02_plan.md)
- [`docs/refactor/milestones/M02/M02_toolcalls.md`](docs/refactor/milestones/M02/M02_toolcalls.md)
- [`docs/refactor/milestones/M02/M02_summary.md`](docs/refactor/milestones/M02/M02_summary.md)
- [`docs/refactor/milestones/M02/M02_audit.md`](docs/refactor/milestones/M02/M02_audit.md)

---

## Phase 1: CI Health & Guardrails (In Progress)

### M03 — CI Workflow Audit for Silent Failures ✅ COMPLETE

**Status:** 🔒 Closed and Locked  
**Date:** 2026-02-08  
**Effort:** 4 hours  
**Change Class:** Documentation-Only (Audit)

**Intent:**
Perform a systematic audit of PyTorch's 142 CI workflows to identify silent failure patterns, false confidence signals, and structural gaps.

**Deliverables:**
- ✅ `M03_audit.md` — Full taxonomy, inventory, and risk register
- ✅ `M03_summary.md` — Executive summary with top 5 risks
- ✅ `M03_toolcalls.md` — Tool invocation log (13 entries)

**Key Findings:**
- **4 high-severity silent failure risks** identified
- **Target Determination** can fail silently (`continue-on-error: true`)
- **LLM TD Retrieval** has job-level `continue-on-error`
- **Executorch build** disabled with `if: false` in `trunk.yml`
- **tools-unit-tests.yml** tests always pass due to `continue-on-error`

**Non-Goals (Honored):**
- ❌ No CI workflow modifications
- ❌ No fixes applied
- ❌ No enforcement changes

**Verification:**
- ✅ All 142 workflow files inventoried
- ✅ Workflows classified by signal strength
- ✅ Silent failure patterns documented with evidence
- ✅ Cross-referenced with M00 known risks
- ✅ No code or CI files modified

**Closeout Artifacts:**
- [`docs/refactor/milestones/M03/M03_plan.md`](docs/refactor/milestones/M03/M03_plan.md)
- [`docs/refactor/milestones/M03/M03_toolcalls.md`](docs/refactor/milestones/M03/M03_toolcalls.md)
- [`docs/refactor/milestones/M03/M03_audit.md`](docs/refactor/milestones/M03/M03_audit.md)
- [`docs/refactor/milestones/M03/M03_summary.md`](docs/refactor/milestones/M03/M03_summary.md)

---

### M04 — Fix High-Priority CI Silent Failures ✅ COMPLETE

**Status:** 🔒 Closed and Locked  
**Date:** 2026-02-08  
**Effort:** 2 hours  
**Change Class:** CI Configuration (Behavior-Preserving)  
**Merge:** PR #2 → `760d459c4fb`

**Intent:**
Address the 5 high-severity silent failure risks identified in M03. Surgical, targeted fixes only.

**Deliverables:**
- ✅ `target_determination.yml` — Remove `continue-on-error` from Do TD step (M03-R01)
- ✅ `llm_td_retrieval.yml` — Remove job-level `continue-on-error` (M03-R02)
- ✅ `trunk.yml` — Remove disabled executorch jobs (M03-R03)
- ✅ `tools-unit-tests.yml` — Remove `continue-on-error` from test steps (M03-R04)
- ✅ `scorecards.yml` — Re-enable OSSF security scoring for upstream (M03-R05)

**Invariant Introduced:**
- **INV-060** — CI Critical Path Integrity: If a correctness-critical step fails, CI must fail visibly.

**Evidence Constraint:**
Fork CI is guarded by `github.repository_owner == 'pytorch'`. Verification was performed via diff-based semantic proof. Upstream CI verification deferred as M04-V01.

**Non-Goals (Honored):**
- ❌ No new CI jobs
- ❌ No workflow re-architecture
- ❌ No action pinning (M06)
- ❌ No YAML linting (M05)

**Verification:**
- ✅ One commit per risk ID (granular rollback)
- ✅ Only 5 workflow files modified
- ✅ Semantic proof documented in M04_audit.md
- ✅ No scope creep

**Closeout Artifacts:**
- [`docs/refactor/milestones/M04/M04_plan.md`](docs/refactor/milestones/M04/M04_plan.md)
- [`docs/refactor/milestones/M04/M04_toolcalls.md`](docs/refactor/milestones/M04/M04_toolcalls.md)
- [`docs/refactor/milestones/M04/M04_audit.md`](docs/refactor/milestones/M04/M04_audit.md)
- [`docs/refactor/milestones/M04/M04_summary.md`](docs/refactor/milestones/M04/M04_summary.md)

---

### M05 — CI Workflow Linting & Structural Guardrails ✅ COMPLETE

**Status:** 🔒 Closed and Locked  
**Date:** 2026-02-08  
**Effort:** 2 hours  
**Change Class:** Verification-Only  
**Merge:** PR #174557 (pending)

**Intent:**
Add `actionlint` as a non-blocking CI workflow to detect workflow anti-patterns before they ship, establishing an early warning system for CI integrity.

**Deliverables:**
- ✅ `.github/workflows/refactor-actionlint.yml` — actionlint CI workflow
- ✅ `M05_audit.md` — Findings classification (0 errors in 144 files)
- ✅ `M05_summary.md` — Executive summary

**Invariant Introduced:**
- **INV-070** — CI Structural Validity: All CI workflows must be syntactically valid and analyzable by static tooling.

**Key Finding:**
- **Zero actionlint errors** across 144 workflow files
- PyTorch's existing CI workflows are structurally clean
- Clean baseline established for INV-070

**Non-Goals (Honored):**
- ❌ No workflow fixes
- ❌ No SARIF/Security tab integration
- ❌ No shellcheck (scope limitation)
- ❌ No action pinning (M06)
- ❌ No enforcement (observational only)

**Verification:**
- ✅ actionlint v1.7.7 executed locally
- ✅ 144 workflows scanned, 0 errors found
- ✅ No existing workflows modified
- ✅ CI workflow added (non-blocking)

**Closeout Artifacts:**
- [`docs/refactor/milestones/M05/M05_plan.md`](docs/refactor/milestones/M05/M05_plan.md)
- [`docs/refactor/milestones/M05/M05_toolcalls.md`](docs/refactor/milestones/M05/M05_toolcalls.md)
- [`docs/refactor/milestones/M05/M05_audit.md`](docs/refactor/milestones/M05/M05_audit.md)
- [`docs/refactor/milestones/M05/M05_summary.md`](docs/refactor/milestones/M05/M05_summary.md)

---

### M06 — Action Pinning & Supply-Chain Hardening 🔵 NEXT

**Status:** Ready to Start  
**Priority:** P1  
**Blockers:** None (M05 complete)

**Intent:**
Pin all GitHub Actions to full SHA for supply-chain security.

---

### M07-M10 — CI Health (Planned)

See [`docs/refactor/audit/REFACTOR_PHASE_MAP.md`](docs/refactor/audit/REFACTOR_PHASE_MAP.md) for full Phase 1 plan.

---

# Program Status

---

## Active Risks (P0-P1)

From baseline audit, top risks requiring mitigation:

| ID | Risk | Priority | Status | Milestone |
|----|------|---------|--------|-----------|
| **I01** | No Working Build Environment | P0 | 🟡 Mitigated | M01 (static checks) |
| **I02** | Empty REFACTOR.md | P0 | ✅ Resolved | M00-M02 |
| **I03** | 130+ CI Workflows (Maintenance) | P1 | ✅ Mitigated | M03-M05 (audit + linting) |
| **I04** | Mixed Action Pinning | P1 | 🔵 Active | M06-M07 |
| **I05** | Third-Party Supply Chain Risk | P1 | 🔵 Active | M08-M10 |
| **I06** | Implicit Distributed Protocol | P2 | 🔵 Active | M11-M12 |
| **I07** | No Pre-Commit Hooks | P2 | 🔵 Active | M13-M14 |

---

## Deferred Verification

| ID | Description | Discovered | Deferred To | Exit Criteria |
|----|-------------|------------|-------------|---------------|
| **M04-V01** | Upstream CI execution verification for M04 changes | M04 | Upstream PR | TD failure propagates; tools-unit-tests fails on pytest failure; scorecards runs cleanly |

---

## Milestone Progress

| Phase | Milestones | Complete | In Progress | Planned |
|-------|-----------|----------|-------------|---------|
| **Phase 0** | M00-M02 | 3 (M00, M01, M02) | 0 | 0 |
| **Phase 1** | M03-M10 | 3 (M03, M04, M05) | 0 | 5 |
| **Phase 2** | M11-M14 | 0 | 0 | 4 |
| **Phase 3** | M15-M19 | 0 | 0 | 5 |
| **Phase 4** | M20+ | 0 | 0 | TBD |

**Program Progress:** 6/22 milestones complete (27%)  
**Phase 0:** ✅ Complete  
**Phase 1:** 🔄 In Progress (3/8)  
**Estimated Remaining:** ~155 hours (M06-M19)

---

## Score Trend

| Date | Phase | Architecture | Tests | CI | Security | Velocity |
|------|-------|-------------|-------|----|---------| ---------|
| 2026-02-08 | M00 (Baseline) | 8/10 | 7/10 | 8/10 | 6/10 | N/A |
| 2026-02-08 | M02 (Phase 0 Complete) | 8/10 | 7/10 | 8/10 | 6/10 | Established |
| 2026-02-08 | M03 (CI Audit Complete) | 8/10 | 7/10 | 7/10* | 6/10 | Maintained |
| 2026-02-08 | M04 (Silent Failures Fixed) | 8/10 | 7/10 | 7.5/10 | 6/10 | Maintained |
| 2026-02-08 | M05 (Actionlint Added) | 8/10 | 7/10 | 7.5/10 | 6/10 | Maintained |

*CI score maintained: Actionlint confirms clean baseline (0 errors in 144 files). Full recovery to 8/10 expected after M06 (action pinning).

**Targets (Post-Phase 3):**
- Architecture: Maintain 8/10
- Tests: Improve to 8/10
- CI: Maintain 8/10
- Security: Improve to 7/10

---

## Tool Logging

All tool invocations are logged in: [`docs/refactor/toolcalls.md`](docs/refactor/toolcalls.md)

Milestone-specific logs are in: `docs/refactor/milestones/MNN/MNN_toolcalls.md`

---

## Recovery Protocol

If session is interrupted:

1. Read the current milestone's `MNN_toolcalls.md` for last recorded action
2. Report: what was being done, what step was next, completion status
3. Wait for confirmation before resuming

For program-level recovery, consult: [`docs/refactor/toolcalls.md`](docs/refactor/toolcalls.md)

---

## Document Version

**Last Updated:** 2026-02-08 (M05 closeout)  
**Next Update:** M06 completion  
**Baseline Locked:** Commit c5f1d40  
**Phase 0:** ✅ Complete  
**Phase 1:** 🔄 In Progress (3/8 milestones)

---

**End of REFACTOR.md**
