# 📌 Milestone Summary — M07: Add Dependabot for GitHub Actions Updates

**Project:** PyTorch Refactoring Program  
**Phase:** Phase 1 — CI Health & Guardrails  
**Milestone:** M07 — Add Dependabot (for action updates)  
**Timeframe:** 2026-02-08  
**Status:** ✅ Complete (Pending Merge Approval)  
**Baseline:** 17f7cbf71905e13c578ea75add005949deb766c4  
**Refactor Posture:** Verification / Maintenance Automation (config-only)

---

## 1. Milestone Objective

**Why this milestone existed:**

M06 pinned all 13 external third-party GitHub Actions to immutable commit SHAs, eliminating mutable version tag references. However, pinned SHAs become stale over time as actions release security patches and improvements. M07 establishes an automated mechanism to propose updates to these pinned actions through reviewable PRs.

> **What would remain unsafe without this refactor?**  
> Pinned actions would drift from upstream releases. Security patches, bug fixes, and compatibility updates would require manual discovery and intervention. The maintenance burden for 300+ action references would scale with time.

---

## 2. Scope Definition

### Delivered

| Deliverable | Description |
|-------------|-------------|
| `github-actions` ecosystem entry | Added to existing `.github/dependabot.yml` |
| Weekly schedule | Conservative update frequency |
| PR limit of 5 | Prevents notification flood |
| Matching label style | Aligns with existing repo conventions |
| M06-B ignore rules | Honors deferred internal `@main` actions |

### Out of Scope (Honored)

- ❌ No action version upgrades by hand
- ❌ No workflow logic changes
- ❌ No pin-format changes (SHAs preserved)
- ❌ No SBOM work (M08)
- ❌ No third-party audit scripting (M09)

---

## 3. Refactor Classification

### Change Type

**Config-only** — No code changes, no workflow logic changes, no test changes.

### Observability

**No externally observable changes to:**
- PyTorch API, CLI, or library behavior
- CI workflow execution or outputs
- Build artifacts or test execution

**Post-merge observable effect:**
- Dependabot will begin proposing PRs for action updates (weekly)
- PRs will have consistent labels and commit message format

---

## 4. Work Executed

| Action | Detail |
|--------|--------|
| File modified | `.github/dependabot.yml` |
| Lines added | 20 |
| Config type | Append (existing pip config preserved) |
| New ecosystem | `github-actions` |

### Configuration Added

```yaml
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    target-branch: "main"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "[Dependabot] Update"
      include: "scope"
    labels:
      - "dependencies"
      - "open source"
      - "topic: not user facing"
      - "module: ci"
      - "refactor-program"
    ignore:
      - dependency-name: "pytorch/pytorch"
      - dependency-name: "pytorch/test-infra"
```

---

## 5. Invariants & Compatibility

### Declared Invariants

| ID | Invariant | Status |
|----|-----------|--------|
| INV-060 | CI Critical Path Integrity | ✅ Protected (no logic changes) |
| INV-070 | CI Structural Validity | ✅ Protected (YAML validated) |
| INV-080 | Action Immutability | ✅ Protected (SHAs unchanged) |
| INV-090 | Action Update Channel Exists | ✅ **Introduced** (observational) |

**INV-090 Definition:** "There is a repo-native automated mechanism that proposes updates to GitHub Actions dependencies via PRs."

### Compatibility Notes

- **Backward compatibility:** ✅ Preserved (no API changes)
- **Breaking changes:** ❌ None
- **Deprecations:** ❌ None

---

## 6. Validation & Evidence

### Proof Type A: Structural Validation (Complete)

| Evidence | Method | Result |
|----------|--------|--------|
| File exists | File system check | ✅ PASS |
| YAML syntax valid | `yaml.safe_load()` | ✅ PASS |
| Version is 2 | YAML parse | ✅ PASS |
| Two ecosystems present | YAML parse | ✅ PASS (`pip`, `github-actions`) |

### Proof Type B: Runtime Validation (Deferred)

| Evidence | Status | Tracking |
|----------|--------|----------|
| Dependabot opens action update PR | ⏳ Deferred | M07-V01 |
| Dependabot enabled in GitHub UI | ⏳ Deferred | M07-V01 |

---

## 7. Connection to M06

M07 is the **maintenance automation companion** to M06:

| M06 | M07 |
|-----|-----|
| Pinned external actions to SHA | Enables automated SHA updates |
| Eliminated mutable tags | Proposes tag→SHA updates via PR |
| 13 actions, 52 files | Monitors same 13 actions |
| One-time hardening | Ongoing maintenance automation |

Together, M06 + M07 form a complete supply-chain hardening solution:
1. **M06:** Actions are immutable (no surprise changes)
2. **M07:** Updates are reviewable (controlled change process)

---

## 8. Issues, Exceptions, and Guardrails

### Issues Encountered

**None.** Config-only change with straightforward implementation.

### Guardrails Added

- **INV-090** — Action Update Channel Exists (observational invariant)
- **M06-B ignore rules** — Prevents noise from internal `@main` actions

---

## 9. Deferred Work

| Item | Reason | Tracking |
|------|--------|----------|
| Dependabot runtime verification | Cannot observe locally | M07-V01 |
| PyTorch-owned `@main` actions | Requires upstream policy | M06-V01 (unchanged) |

**M07-V01 Exit Criteria:** Dependabot opens at least one action update PR, OR shows as enabled in GitHub Security/Insights UI.

---

## 10. Governance Outcomes

**What is now provably true that was not before:**

1. ✅ Dependabot config exists for `github-actions` ecosystem
2. ✅ Weekly update schedule established
3. ✅ PR noise controlled (limit of 5)
4. ✅ Labels match repo conventions
5. ✅ M06-B deferral honored (ignore rules)
6. ✅ INV-090 structurally established
7. ✅ Existing pip/transformers config preserved

---

## 11. Exit Criteria Evaluation

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `.github/dependabot.yml` updated | ✅ Met | File modified |
| `github-actions` ecosystem added | ✅ Met | YAML parse verified |
| Conservative schedule (weekly) | ✅ Met | Config review |
| PR limit set | ✅ Met | `open-pull-requests-limit: 5` |
| M06-B deferral honored | ✅ Met | Ignore rules present |
| Labels match existing style | ✅ Met | Config review |
| Audit document created | ✅ Met | M07_audit.md |
| Deferral documented | ✅ Met | M07-V01 registered |

---

## 12. Final Verdict

**M07 milestone objectives met.** Dependabot is now configured to propose updates to GitHub Actions dependencies on a weekly basis. Combined with M06's SHA pinning, the supply-chain posture for CI actions is now both hardened (immutable) and maintainable (automated updates).

---

## 13. Authorized Next Step

- ⏳ Merge pending approval
- ⏳ REFACTOR.md update pending
- ⏳ Program progress: 8/22 milestones (36%) upon merge
- ✅ Proceed to M08 (Dependency & SBOM Baseline)

---

## 14. Canonical References

| Artifact | Reference |
|----------|-----------|
| Branch | `m07-dependabot-actions` |
| Base commit | `17f7cbf71905e13c578ea75add005949deb766c4` |
| Plan | `docs/refactor/milestones/M07/M07_plan.md` |
| Audit | `docs/refactor/milestones/M07/M07_audit.md` |
| Toolcalls | `docs/refactor/milestones/M07/M07_toolcalls.md` |

---

**End of M07 Summary**

