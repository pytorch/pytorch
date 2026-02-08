# M07 Audit — Add Dependabot for GitHub Actions Updates

**Milestone:** M07 — Add Dependabot (for action updates)  
**Phase:** Phase 1 — CI Health  
**Change Class:** Verification / Maintenance Automation (config-only)  
**Risk:** Low  
**Date:** 2026-02-08  
**Baseline Commit:** 17f7cbf71905e13c578ea75add005949deb766c4

---

## 1. Scope Verification

### In Scope (Completed)

| Deliverable | Status |
|-------------|--------|
| Append `github-actions` ecosystem to `.github/dependabot.yml` | ✅ Complete |
| Preserve existing `pip`/`transformers` configuration | ✅ Complete |
| Configure weekly schedule for action updates | ✅ Complete |
| Set conservative PR limit (5) | ✅ Complete |
| Add ignore rules for M06-B deferred repos | ✅ Complete |
| Match existing label style | ✅ Complete |

### Out of Scope (Honored)

| Non-Goal | Verified |
|----------|----------|
| No action upgrades by hand | ✅ No manual upgrades |
| No workflow logic edits | ✅ No workflow files modified |
| No pin-format changes | ✅ Existing SHAs untouched |
| No SBOM work (M08) | ✅ Deferred |
| No third-party audit scripting (M09) | ✅ Deferred |

---

## 2. Configuration Diff

### File Modified: `.github/dependabot.yml`

**Change Type:** Append (existing content preserved)

```yaml
  # M07: Keep GitHub Actions pinned SHAs up to date (see M06 action pinning)
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
    # Ignore PyTorch-owned reusable workflows (M06-B deferral: internal @main actions)
    ignore:
      - dependency-name: "pytorch/pytorch"
      - dependency-name: "pytorch/test-infra"
```

---

## 3. Configuration Rationale

| Setting | Value | Rationale |
|---------|-------|-----------|
| `package-ecosystem` | `github-actions` | Enable updates for GitHub Actions (target of M06 pinning) |
| `directory` | `/` | Root directory contains all workflow files |
| `interval` | `weekly` | Conservative frequency; avoids PR noise while staying current |
| `target-branch` | `main` | Match existing pip config behavior |
| `open-pull-requests-limit` | `5` | Prevent PR flood; allows grouped review |
| `commit-message.prefix` | `[Dependabot] Update` | Match existing prefix style in file |
| Labels | Match existing + `refactor-program` | Align with repo conventions; one new label for tracking |
| `ignore: pytorch/pytorch` | Deferred M06-B | First-party actions at `@main` require policy change |
| `ignore: pytorch/test-infra` | Deferred M06-B | First-party actions at `@main` require policy change |

---

## 4. Invariants

### Protected Invariants

| ID | Invariant | Status |
|----|-----------|--------|
| INV-060 | CI Critical Path Integrity | ✅ Protected (no workflow logic changed) |
| INV-070 | CI Structural Validity | ✅ Protected (YAML syntax validated) |
| INV-080 | Action Immutability | ✅ Protected (pinned SHAs unchanged) |

### Introduced Invariant

| ID | Invariant | Status |
|----|-----------|--------|
| INV-090 | Action Update Channel Exists | ✅ **Introduced** (observational) |

**INV-090 Definition:** "There is a repo-native automated mechanism that proposes updates to GitHub Actions dependencies via PRs."

**Observational Status:** INV-090 is now structurally true (config exists), but runtime proof requires GitHub-side scheduling and time elapsing.

---

## 5. Evidence Constraints

### Proof Type A: Structural Validation (Completed)

| Check | Method | Result |
|-------|--------|--------|
| File exists | `ls .github/dependabot.yml` | ✅ Exists |
| YAML valid | `python yaml.safe_load()` | ✅ Valid syntax |
| Version is 2 | YAML parse | ✅ `version: 2` |
| Has 2 ecosystem entries | YAML parse | ✅ `pip`, `github-actions` |
| github-actions config present | YAML parse | ✅ All required fields |

### Proof Type B: Runtime Validation (Deferred)

| Check | Status | Deferral |
|-------|--------|----------|
| Dependabot opens at least one PR | ⏳ Deferred | M07-V01 |
| Dependabot shows as enabled in GitHub UI | ⏳ Deferred | M07-V01 |

**M07-V01:** "Dependabot runtime behavior is unobservable locally; verify post-merge via GitHub UI or PR arrival within 7 days of merge."

---

## 6. Validation Summary

| Validation | Method | Result |
|------------|--------|--------|
| YAML syntax | `python -c "import yaml; yaml.safe_load(...)"` | ✅ PASS |
| Structure check | Python YAML parse | ✅ 2 ecosystems present |
| Actionlint | Not installed locally | ⚠️ Skipped (expected) |
| Existing config preserved | Diff analysis | ✅ pip config unchanged |

**Note:** Actionlint validates workflow files, not `dependabot.yml`. The YAML syntax check is the appropriate validation for this file type.

---

## 7. Rollback Plan

**Rollback Method:** Single revert commit

```bash
git revert <M07-commit-sha>
```

This removes the `github-actions` ecosystem entry while preserving the existing `pip`/`transformers` configuration.

**Risk on Rollback:** None. Dependabot will stop proposing action updates; existing pinned SHAs remain immutable.

---

## 8. Files Changed

| File | Change Type | Lines Added | Lines Removed |
|------|-------------|-------------|---------------|
| `.github/dependabot.yml` | Modified (append) | 20 | 0 |

**Total:** 1 file, 20 lines added

---

## 9. Deferred Verification Registry

| ID | Description | Discovered | Exit Criteria |
|----|-------------|------------|---------------|
| M07-V01 | Dependabot runtime behavior unobservable locally | M07 | Dependabot opens at least one action update PR, OR shows as enabled in GitHub Security/Insights UI |

---

## 10. Audit Verdict

**M07 structural validation complete.**

- ✅ Config exists and is minimal
- ✅ Conservative settings applied
- ✅ Matches existing file conventions
- ✅ M06-B deferral honored via ignore rules
- ✅ All in-scope invariants protected
- ✅ INV-090 introduced (observational)
- ⏳ Runtime proof deferred as M07-V01

**Ready for documentation pack (summary) and REFACTOR.md update.**

---

**End of M07 Audit**

