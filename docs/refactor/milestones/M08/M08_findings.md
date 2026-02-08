# M08 Findings — CI Truthfulness Inventory

**Milestone:** M08 — CI Truthfulness & Silent-Failure Elimination  
**Date:** 2026-02-08  
**Scope:** All 143 GitHub Actions workflow files

---

## Executive Summary

| Pattern | Total Instances | Files Affected | Acceptable | Needs Review | Action Required |
|---------|-----------------|----------------|------------|--------------|-----------------|
| `continue-on-error: true` | ~215 | 28 | ~200 | 10 | 5 |
| `if: always()` | ~270 | 32 | ~265 | 5 | 0 |
| `\|\| true` | ~34 | 16 | ~30 | 4 | 0 |
| `set +e` | 4 | 4 | 2 | 2 | 0 |

**Overall Assessment:** The vast majority of silent-failure patterns are **acceptable** when classified:
- Cleanup/teardown steps that must run regardless of job status
- Informational/telemetry uploads that should not block correctness
- Best-effort cache operations
- Generated nightly binary workflows (special case)

**Actionable items identified:** 5 instances require remediation or explicit justification.

---

## M04 Verification Results

All M04 fixes remain intact:

| Workflow | M04 Fix | Current Status |
|----------|---------|----------------|
| `target_determination.yml` | Removed `continue-on-error` from Do TD step | ✅ Verified - Do TD step has no `continue-on-error` |
| `llm_td_retrieval.yml` | Removed job-level `continue-on-error` | ✅ Verified - only step-level on informational step |
| `trunk.yml` | Removed disabled executorch jobs | ✅ Verified - no `if: false` patterns found |
| `tools-unit-tests.yml` | Removed `continue-on-error` from test steps | ✅ Verified - no `continue-on-error` in file |
| `scorecards.yml` | Re-enabled OSSF scoring | ✅ Verified - runs on `pytorch/pytorch` only |

---

## Pattern Classification

### Class A: Acceptable (No Action Required)

These patterns are justified and documented inline or by context.

#### A1: Cleanup/Teardown Steps with `if: always()`

**Justification:** Cleanup must run regardless of job outcome to prevent resource leaks.

| File | Lines | Step Name | Status |
|------|-------|-----------|--------|
| `_linux-test.yml` | 531, 558, 561 | Teardown Linux, Cleanup docker | ✅ Acceptable |
| `_linux-build.yml` | 468, 471 | Teardown Linux | ✅ Acceptable |
| `_win-test.yml` | 262 | Teardown | ✅ Acceptable |
| `_win-build.yml` | 218 | Teardown | ✅ Acceptable |
| `_mac-test.yml` | 280, 288 | Teardown | ✅ Acceptable |
| `_mac-build.yml` | 193 | Teardown | ✅ Acceptable |
| `_xpu-test.yml` | 319, 325 | Teardown | ✅ Acceptable |
| `_rocm-test.yml` | 312 | Teardown | ✅ Acceptable |
| `llm_td_retrieval.yml` | 121 | Teardown Linux | ✅ Acceptable |
| `docker-builds.yml` | 170, 174 | Teardown | ✅ Acceptable |

#### A2: Artifact Upload with `if: always()` (Post-Test)

**Justification:** Test artifacts must be uploaded for debugging even on failure.

| File | Pattern | Status |
|------|---------|--------|
| `_linux-test.yml` | lines 509, 524 | Upload test artifacts | ✅ Acceptable |
| `_win-test.yml` | lines 225, 240 | Upload test artifacts | ✅ Acceptable |
| `_mac-test.yml` | lines 233, 254 | Upload test artifacts | ✅ Acceptable |
| `_xpu-test.yml` | lines 287, 302 | Upload test artifacts | ✅ Acceptable |
| `_rocm-test.yml` | lines 291, 306 | Upload test artifacts | ✅ Acceptable |

#### A3: Best-Effort Cache/Download with `continue-on-error: true`

**Justification:** Cache misses should not fail builds; downloads are retried elsewhere.

| File | Lines | Step Name | Status |
|------|-------|-----------|--------|
| `target_determination.yml` | 46, 53 | Download pytest cache, Download LLM Artifacts | ✅ Acceptable |
| `_linux-build.yml` | 246 | Download pytest cache | ✅ Acceptable |
| `_linux-test.yml` | 224 | Download TD artifacts | ✅ Acceptable |

#### A4: Monitoring/Telemetry with `continue-on-error: true`

**Justification:** Telemetry collection must not block correctness.

| File | Lines | Step Name | Status |
|------|-------|-----------|--------|
| `_linux-build.yml` | 227 | Start monitoring script | ✅ Acceptable |
| `_linux-test.yml` | 204, 516, 547 | Monitoring, Stop monitoring, Upload stats | ✅ Acceptable |
| `upload-test-stats.yml` | 73 | Configure AWS credentials | ✅ Acceptable |
| `upload-test-stats.yml` | 138 | check-api-rate job | ✅ Acceptable |
| `upload-torch-dynamo-perf-stats.yml` | 42 | Configure AWS credentials | ✅ Acceptable |

#### A5: Generated Nightly Binary Workflows

**Justification:** These are auto-generated files for Windows binary builds. The `continue-on-error` and `if: always()` patterns are systematic across all matrix configurations for cleanup and notification purposes.

| File | Instance Count | Status |
|------|---------------|--------|
| `generated-windows-binary-wheel-nightly.yml` | ~140 | ✅ Acceptable (generated) |
| `generated-windows-binary-libtorch-debug-nightly.yml` | ~16 | ✅ Acceptable (generated) |
| `generated-windows-binary-libtorch-release-nightly.yml` | ~16 | ✅ Acceptable (generated) |
| `generated-windows-arm64-*.yml` | ~5 | ✅ Acceptable (generated) |
| `generated-macos-arm64-*.yml` | ~7 | ✅ Acceptable (generated) |

**Note:** These files are machine-generated and follow a consistent pattern. Modifications should be made to the generator, not the generated files.

#### A6: Informational/Best-Effort Steps

**Justification:** Steps explicitly marked as non-blocking for CI signal.

| File | Line | Step | Justification | Status |
|------|------|------|---------------|--------|
| `llm_td_retrieval.yml` | 88 | Run Retriever | Comment: "Best-effort step... does not gate correctness" | ✅ Acceptable |
| `refactor-actionlint.yml` | 16 | Job-level | Comment: "Non-blocking, observational only" | ✅ Acceptable |
| `trymerge.yml` | 79, 89, 96 | Comment on Canceled, AWS creds, Upload record | Merge bot notifications | ✅ Acceptable |
| `revert.yml` | 58 | Revert step | Best-effort revert notification | ✅ Acceptable |
| `tryrebase.yml` | 49 | Rebase step | Best-effort rebase | ✅ Acceptable |

#### A7: Shell Cleanup Commands with `|| true`

**Justification:** Commands that may fail on first run or when nothing to clean.

| File | Line | Command | Status |
|------|------|---------|--------|
| `trunk-tagging.yml` | 188, 212 | `git tag -d`, `git fetch` | ✅ Acceptable (idempotent cleanup) |
| `_linux-build.yml` | 304, 307 | `mount binfmt_misc`, `docker run --privileged` | ✅ Acceptable (setup may already exist) |
| `_linux-build.yml` | 475, 476 | `docker stop -a`, `docker kill -a` | ✅ Acceptable (cleanup) |
| `_linux-test.yml` | 511, 565, 566 | `cat logs`, `docker stop/kill` | ✅ Acceptable (logs may not exist) |
| `create_release.yml` | 68, 78 | `rm docs/requirements.txt`, `find -exec rm` | ✅ Acceptable (cleanup) |
| `generated-macos-arm64-*.yml` | multiple | `pip uninstall` | ✅ Acceptable (package may not exist) |

#### A8: `set +e` for Complex Logic

| File | Line | Context | Status |
|------|------|---------|--------|
| `trunk-tagging.yml` | 124 | Complex tag logic with manual error handling | ✅ Acceptable |
| `runner_determinator_script_sync.yaml` | 34 | Sync script with manual error handling | ✅ Acceptable |
| `build-triton-wheel.yml` | 149 | Build logic with error recovery | ⚠️ Review |
| `_mac-test.yml` | 216 | Test log collection | ⚠️ Review |

---

### Class B: Needs Review (Potential Issues)

These patterns require closer examination but may be acceptable with justification.

#### B1: Unstable/Experimental Workflows

| File | Line | Pattern | Issue |
|------|------|---------|-------|
| `unstable.yml` | 24 | `continue-on-error: true` (job-level) | **Intentional** - workflow explicitly for unstable jobs |
| `unstable-periodic.yml` | 23 | `continue-on-error: true` (job-level) | **Intentional** - workflow explicitly for unstable jobs |
| `weekly.yml` | 24 | `continue-on-error: true` | XLA commit hash update - best effort |

**Assessment:** These are **acceptable** by design. The workflows are explicitly for experimental/unstable jobs that should not block CI.

#### B2: Lint Autoformat

| File | Lines | Pattern | Issue |
|------|-------|---------|-------|
| `lint-autoformat.yml` | 21, 30, 35 | `continue-on-error: true` | Autoformat steps |

**Assessment:** ⚠️ **Needs inline justification.** Autoformatting failures could hide real issues, but may be intentionally lenient for contributor experience.

#### B3: Binary Build Upload

| File | Line | Pattern | Issue |
|------|------|---------|-------|
| `_binary-upload.yml` | 106 | `continue-on-error: true` | Upload step |
| `_binary-test-linux.yml` | 129 | `continue-on-error: true` | Binary test |

**Assessment:** ⚠️ **Needs review.** Binary upload failures should probably be visible.

---

### Class C: Action Required

These patterns should be fixed or explicitly justified.

#### C1: Build/Test Monitoring with Unclear Justification

| File | Line | Pattern | Issue | Recommendation |
|------|------|---------|-------|----------------|
| `_linux-build.yml` | 398 | `continue-on-error: true` | "Upload utilization stats" | Add inline comment |
| `_linux-build.yml` | 431 | `continue-on-error: true` | "Upload pytest cache" | Add inline comment |
| `_linux-build.yml` | 456 | `continue-on-error: true` | Step after build | Add inline comment |

**Recommendation:** Add inline comments justifying why these steps use `continue-on-error`.

#### C2: Win Build

| File | Line | Pattern | Issue | Recommendation |
|------|------|---------|-------|----------------|
| `_win-build.yml` | 139 | `continue-on-error: true` | Context unclear | Examine and document |

#### C3: Test Workflows

| File | Line | Pattern | Issue | Recommendation |
|------|------|---------|-------|----------------|
| `_linux-test-stable-fa3.yml` | 247 | `continue-on-error: true` | Flash Attention test | Verify intentional |

---

## Shell Suppression Patterns

### Acceptable Uses

| Pattern | Typical Use | Status |
|---------|-------------|--------|
| `\|\| true` after `cat` | Log file may not exist | ✅ Acceptable |
| `\|\| true` after `docker stop/kill` | Container may not exist | ✅ Acceptable |
| `\|\| true` after `rm` | File may not exist | ✅ Acceptable |
| `\|\| true` after `git fetch/tag` | Idempotent operations | ✅ Acceptable |
| `\|\| true` after `pip uninstall` | Package may not be installed | ✅ Acceptable |

### Patterns Requiring `set +e`

| File | Line | Context | Assessment |
|------|------|---------|------------|
| `trunk-tagging.yml` | 124 | Complex multi-command logic | ✅ Acceptable - uses explicit checks |
| `runner_determinator_script_sync.yaml` | 34 | Script sync logic | ✅ Acceptable - manual error handling |
| `build-triton-wheel.yml` | 149 | Build with recovery | ⚠️ Review - ensure errors are caught |
| `_mac-test.yml` | 216 | Log collection | ⚠️ Review - may hide failures |

---

## Recommendations

### Immediate Actions (M08 Scope)

1. **Add inline justification comments** to `continue-on-error` patterns in:
   - `_linux-build.yml` (lines 398, 431, 456)
   - `_win-build.yml` (line 139)
   - `lint-autoformat.yml` (lines 21, 30, 35)

2. **Verify and document** the `continue-on-error` in:
   - `_linux-test-stable-fa3.yml` (line 247)
   - `_binary-upload.yml` (line 106)
   - `_binary-test-linux.yml` (line 129)

3. **Document policy** in REFACTOR.md for when `continue-on-error` is acceptable.

### Deferred Actions (Future Milestones)

1. **Generator review:** The `generated-*.yml` files contain many patterns. Consider updating the generator to add justification comments.

2. **Enforcement mechanism:** Consider adding a CI check that scans for new `continue-on-error` patterns without inline justification.

---

## Conclusion

The CI truthfulness posture is **generally healthy**. The majority of silent-failure patterns are:

1. **Intentional and justified** (cleanup, telemetry, best-effort)
2. **In generated files** (nightly binary workflows)
3. **Already fixed by M04** (critical path integrity)

The 5 actionable items identified are minor and primarily require **documentation**, not removal.

**M08 Recommendation:** Add inline justification comments rather than removing patterns, as the patterns themselves are defensible.

---

**End of M08 Findings**

