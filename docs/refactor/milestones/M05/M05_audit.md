# M05 Audit — CI Workflow Linting & Structural Guardrails

**Milestone:** M05  
**Date:** 2026-02-08  
**Auditor:** AI Agent (Cursor)  
**Change Class:** Verification-Only  
**Status:** Complete

---

## Executive Summary

Actionlint structural validation was executed against all 144 GitHub Actions workflow files in the PyTorch repository. **Zero structural errors were found.** This establishes a clean baseline for INV-070 (CI Structural Validity).

---

## Scope of Audit

### Files Scanned
- **Directory:** `.github/workflows/`
- **Total Files:** 144 YAML workflow files
- **Tool:** actionlint v1.7.7

### Rules Applied
| Rule | Status | Notes |
|------|--------|-------|
| YAML syntax | ✅ Enabled | Core structural validation |
| Job/step references | ✅ Enabled | needs, outputs, job IDs |
| Expression validation | ✅ Enabled | ${{ }} syntax checking |
| Action inputs | ✅ Enabled | Required/optional params |
| Deprecated syntax | ✅ Enabled | set-output, save-state, etc. |
| Shellcheck | ❌ Disabled | Per M05 locked answer (scope limitation) |
| Pyflakes | ❌ Disabled | Not in PATH |

---

## Findings Summary

### Actionlint Results

| Severity | Count | Description |
|----------|-------|-------------|
| **P0 (Critical)** | 0 | Invalid YAML, unreachable jobs, broken needs/outputs |
| **P1 (Concerning)** | 0 | Ambiguous conditions, unused outputs, suspicious matrices |
| **P2 (Informational)** | 0 | Cosmetic warnings, non-idiomatic patterns |
| **Total Errors** | **0** | All 144 workflows passed structural validation |

### Execution Details

```
actionlint v1.7.7
Collected 144 YAML files
Linting 144 files
Found 0 errors in 144 files
Execution time: ~500ms
```

---

## Analysis

### Positive Findings

1. **All workflows are syntactically valid** — No malformed YAML detected
2. **Job dependencies are correct** — All `needs:` references resolve properly
3. **Expression syntax is valid** — No broken `${{ }}` expressions
4. **No deprecated syntax detected** — Workflows use current GitHub Actions patterns
5. **No unreachable jobs** — All jobs have valid execution paths

### Observations

1. **PyTorch CI is well-maintained** — The zero-error result indicates existing code review and linting practices are effective
2. **Shellcheck was disabled** — This was intentional per M05 scope. Inline bash scripts were not validated.
3. **Pyflakes was disabled** — Python expressions in workflows were not validated.
4. **Scale** — 144 workflows is substantial; clean results at this scale are notable

### Coverage Gaps (Documented, Not Addressed)

| Gap | Description | Candidate Milestone |
|-----|-------------|---------------------|
| Shellcheck | Inline bash scripts not linted | Future (M05+ candidate) |
| Pyflakes | Python expressions not validated | Future |
| Semantic analysis | Logic correctness not verified | Out of scope for static linting |
| Action pinning | SHA pinning not enforced | M06 |

---

## Invariant Verification

### INV-070 — CI Structural Validity

**Definition:** All CI workflows must be syntactically valid and analyzable by static tooling.

**Status:** ✅ **VERIFIED**

**Evidence:**
- actionlint v1.7.7 executed against 144 workflows
- 0 errors reported
- All workflows parse correctly and have valid structure

**Mode:** Observational (non-blocking CI job added)

---

## Workflow Inventory

### By Category (from M03 audit)

| Category | Count | Actionlint Status |
|----------|-------|-------------------|
| Core CI (pull, trunk, periodic) | ~15 | ✅ Clean |
| Platform-specific (linux, mac, win, rocm, xpu) | ~40 | ✅ Clean |
| Binary builds (manywheel, libtorch) | ~20 | ✅ Clean |
| Performance/benchmarks | ~25 | ✅ Clean |
| Docker/infrastructure | ~15 | ✅ Clean |
| Automation (labels, reviews, stale) | ~10 | ✅ Clean |
| Refactor program (smoke, actionlint) | 2 | ✅ Clean |
| Other | ~17 | ✅ Clean |

---

## Deliverables Produced

| Artifact | Status |
|----------|--------|
| `.github/workflows/refactor-actionlint.yml` | ✅ Created |
| Local actionlint execution | ✅ Complete |
| Findings classification (this document) | ✅ Complete |
| CI workflow created | ✅ PR #174557 |

---

## Recommendations

### Immediate (No action required in M05)
1. **Maintain current state** — Zero errors is the target state
2. **Monitor future PRs** — The actionlint workflow will catch regressions

### Future Milestones
1. **M06:** Action pinning (SHA pinning for supply chain security)
2. **Future:** Consider shellcheck integration as separate milestone
3. **Future:** Consider SARIF output for Security tab integration

---

## Non-Goals (Honored)

- ❌ No workflow fixes applied
- ❌ No SARIF/Security tab integration
- ❌ No shellcheck validation
- ❌ No action pinning (deferred to M06)
- ❌ No enforcement (continue-on-error: true)

---

## Conclusion

M05 establishes **INV-070 (CI Structural Validity)** with a clean baseline. The actionlint workflow provides ongoing structural validation for all future workflow changes. The zero-error result confirms PyTorch's existing CI practices are structurally sound.

---

**Audit Complete:** 2026-02-08  
**Auditor:** AI Agent (Cursor)  
**Approved:** Pending M05 closeout

