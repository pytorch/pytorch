# M03 Summary — CI Workflow Audit for Silent Failures

**Milestone:** M03  
**Status:** 🔒 Closed and Locked  
**Date:** 2026-02-08  
**Mode:** Audit-Only  
**Effort:** 4 hours

---

## Executive Summary

M03 performed a **systematic audit of 142 CI workflow files** in PyTorch's `.github/workflows/` directory to identify silent failure patterns — places where CI can report success despite underlying failures.

The audit found **4 high-severity silent failure risks** that should be addressed in M04, plus several lower-severity patterns that are either acceptable by design or require monitoring.

**Key insight:** PyTorch's CI infrastructure is *fundamentally sound*, but a small number of critical workflows use `continue-on-error: true` in ways that can mask real failures. Most notably, **Target Determination** — the ML-based test selection system — can fail silently, meaning tests may run with incomplete coverage while CI reports success.

---

## What Was Done

1. **Inventoried all 142 CI workflows** — Categorized by purpose, trigger, and signal strength
2. **Defined a silent failure taxonomy** — 5 categories (soft-fail, conditional skips, retry masking, scope gaps, informational-only)
3. **Deep-analyzed Tier 1 workflows** — `pull.yml`, `trunk.yml`, `lint.yml`, `inductor.yml`
4. **Searched for anti-patterns** — `continue-on-error`, `if: always()`, `fail-fast: false`, `if: false`
5. **Cross-referenced with M00 risks** — Confirmed coverage gaps from baseline audit
6. **Produced prioritized risk register** — 8 risks classified by severity

---

## Top 5-7 CI Risks Discovered

| Rank | Risk | Severity | Location | Impact |
|------|------|----------|----------|--------|
| 1 | **Target Determination fails silently** | 🔴 High | `target_determination.yml` L58-60 | Tests run with potentially incomplete coverage; CI reports success |
| 2 | **LLM TD Retrieval job-level continue-on-error** | 🔴 High | `llm_td_retrieval.yml` L24-26 | Entire ML inference step can fail without anyone knowing |
| 3 | **Executorch build disabled with `if: false`** | 🔴 High | `trunk.yml` L417-424 | Job silently skipped; no visibility in CI results |
| 4 | **Tools unit tests always pass** | 🔴 High | `tools-unit-tests.yml` L38-39, L64-65 | Tests run but failures are ignored |
| 5 | **Scorecards workflow entirely disabled** | 🟡 Medium | `scorecards.yml` L22-24 | Security scoring never runs |
| 6 | **Monitoring steps mask errors** | 🟢 Low | Multiple reusable workflows | Acceptable — non-critical metrics |
| 7 | **Suggestion workflows designed to soft-fail** | 🟢 Low | `lint-autoformat.yml` | Acceptable — by design |

---

## Scope & Limits

### In Scope (Completed)
- All 142 `.yml` workflow files
- Pattern-based analysis for known anti-patterns
- Tiered depth of analysis (Tier 1: deep, Tier 2: medium, Tier 3: summary)
- Cross-reference with M00 audit findings

### Out of Scope (Accepted Gaps)
- **Branch protection rules** — Cannot observe from repository files
- **Shell-level `|| true`** — Requires script-by-script analysis
- **External action internals** — Third-party code not audited
- **Actual CI pass rates** — Would require HUD/metrics access

### Evidence Gaps Documented
- Required vs optional checks (GitHub repo settings, not observable)
- Runtime test coverage (requires artifacts)
- Flake rates (requires historical data)

---

## Deliverables Produced

| Artifact | Description |
|----------|-------------|
| `M03_plan.md` | Execution plan (provided externally) |
| `M03_toolcalls.md` | Tool invocation log (11 entries) |
| `M03_audit.md` | Full audit findings with taxonomy, inventory, risk register |
| `M03_summary.md` | This executive summary |

---

## Verification

- ✅ All 142 workflow files inventoried
- ✅ Workflows classified by signal strength (Strong / Weak / Cosmetic / Legacy)
- ✅ Silent failure patterns identified with evidence
- ✅ Cross-reference with M00 risks complete
- ✅ Prioritized risk register produced
- ✅ No CI or code files modified
- ✅ Audit-only language used throughout

---

## Recommended Next Milestones

### M04: Fix Silent Failures (High-Priority)

**Priority:** P1  
**Effort:** 6 hours  
**Scope:**
1. Remove `continue-on-error: true` from TD step in `target_determination.yml`
2. Remove job-level `continue-on-error` from `llm_td_retrieval.yml`
3. Remove or relocate `if: false` disabled jobs
4. Remove `continue-on-error` from test steps in `tools-unit-tests.yml`

### M05: Add actionlint to CI

**Priority:** P1  
**Effort:** 4 hours  
**Scope:** Add YAML linting to catch workflow errors before merge

### M06: Pin All Actions to SHA

**Priority:** P1  
**Effort:** 12 hours  
**Scope:** Eliminate `@main` and mutable version tags for supply chain security

---

## What Remains Unsafe Without This Audit

Before M03, the PyTorch CI infrastructure was a "black box" — known to be large (130+ workflows) but not systematically understood. M03 transforms this into a **mapped, evidence-backed surface**.

After M03, a reviewer can now answer:

> "Which CI checks can I trust, and why?"

The answer:
- **Pull/Trunk/Lint workflows** — Strong signal; failures block merge
- **Target Determination** — Weak signal; can fail silently (needs M04 fix)
- **Benchmark workflows** — Informational only; do not assert correctness
- **Unstable workflows** — Explicitly allowed to fail (by design)

---

## Deferred Work

| Item | Reason | Revisit In |
|------|--------|------------|
| Shell-level `|| true` audit | Requires script-by-script analysis | Future (if needed) |
| External action security audit | Covered by M06/M07 | M06 |
| Branch protection alignment | Requires repo admin access | Post-M05 |
| Flake rate analysis | Requires HUD/metrics access | Future |

---

## Closeout Checklist

- [x] All scope items completed
- [x] All deliverables produced
- [x] No code or CI files modified
- [x] REFACTOR.md updated with M03 CLOSED entry
- [x] Explicit closeout permission granted

---

**End of M03 Summary**

