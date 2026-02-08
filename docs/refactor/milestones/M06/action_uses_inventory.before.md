# M06 Action Uses Inventory — BEFORE

**Generated:** 2026-02-08  
**Workflow Files Scanned:** 143  
**Total `uses:` Statements:** ~1,600+  
**Unique Action References:** 104

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Local actions (`./.github/...`) | 48 | No pinning needed (repo-internal) |
| SHA-pinned (40-char) | 21 | ✅ Already secure |
| Tag-pinned (`@v1`, `@v2`, etc.) | 13 | 🟡 Needs SHA pinning |
| Branch-pinned (`@main`) | 20 | 🔴 Highest risk, needs SHA pinning |
| **Total needing action** | **33** | |

---

## Already SHA-Pinned (Secure) — 21 Actions

These are already pinned to immutable commit SHAs. No changes needed.

| Action | SHA |
|--------|-----|
| `actions/checkout` | `11bd71901bbe5b1630ceea73d27597364c9af683` |
| `actions/download-artifact` | `65a9edc5881444af0b9093a5e628f2fe47ea3b2e` |
| `actions/download-artifact` | `95815c38cf2ff2164869cbab79da8d1f422bc89e` |
| `actions/github-script` | `60a0d83039c74a4aee543508d2ffcb1c3799cdea` |
| `actions/setup-python` | `a26af69be951a213d495a4c3e4e4022e16d87065` |
| `actions/upload-artifact` | `50769540e7f4bd5e21e526ee35c689e35e0d6874` |
| `actions/upload-artifact` | `ea165f8d65b6e75b540449e92b4886f43607fa02` |
| `aws-actions/configure-aws-credentials` | `ececac1a45f3b08a01d2dd070d28d111c5fe6722` |
| `docker/login-action` | `74a5d142397b4f367a81961eba4e8cd7edddf772` |
| `docker/setup-buildx-action` | `b5ca514318bd6ebac0fb2aedd5d36ec1b5c232a2` |
| `docker/setup-qemu-action` | `29109295f81e9208d7d86ff1c6c12d2833863392` |
| `github/codeql-action/upload-sarif` | `5f532563584d71fdef14ee64d17bafb34f751ce5` |
| `ilammy/msvc-dev-cmd` | `dd5e2fa0a7de1e7929605d9ecc020e749d9856a3` |
| `necojackarc/auto-request-review` | `e08cdffa277d50854744de3f76230260e61c67f4` |
| `nick-fields/retry` | `7152eba30c6575329ac0576536151aca5a72780e` |
| `octokit/request-action` | `05a2312de9f8207044c4c9e41fe19703986acc13` |
| `ossf/scorecard-action` | `865b4092859256271290c77adbd10a43f4779972` |
| `parkerbxyz/suggest-changes` | `a2ec1653b0c4cc8287d682f0066dba4a173cc7f3` |
| `seemethere/download-artifact-s3` | `1da556a7aa0a088e3153970611f6c432d58e80e6` |
| `seemethere/upload-artifact-s3` | `baba72d0712b404f646cebe0730933554ebce96a` |
| `softprops/action-gh-release` | `da05d552573ad5aba039eaac05058a918a7bf631` |

---

## Tag-Pinned (Needs SHA) — 13 Actions

These use version tags which are mutable. Need to resolve to SHAs.

| Action | Current Ref | Risk |
|--------|-------------|------|
| `actions/checkout` | `@v4` | 🟡 Medium |
| `actions/download-artifact` | `@v4` | 🟡 Medium |
| `actions/download-artifact` | `@v4.1.7` | 🟡 Medium |
| `actions/setup-python` | `@v5` | 🟡 Medium |
| `actions/setup-python` | `@v6` | 🟡 Medium |
| `actions/upload-artifact` | `@v4` | 🟡 Medium |
| `actions/upload-artifact` | `@v4.4.0` | 🟡 Medium |
| `anthropics/claude-code-action` | `@v1` | 🟡 Medium |
| `aws-actions/configure-aws-credentials` | `@v4` | 🟡 Medium |
| `ethanis/nitpicker` | `@v1` | 🟡 Medium |
| `ilammy/msvc-dev-cmd` | `@v1` | 🟡 Medium |
| `raven-actions/actionlint` | `@v2` | 🟡 Medium |
| `seemethere/upload-artifact-s3` | `@v5` | 🟡 Medium |

---

## Branch-Pinned (Highest Risk) — 20 Actions

These use `@main` which is always mutable. Highest supply-chain risk.

| Action | Current Ref | Risk |
|--------|-------------|------|
| `pytorch/pytorch/.github/actions/binary-docker-build` | `@main` | 🔴 High |
| `pytorch/pytorch/.github/actions/checkout-pytorch` | `@main` | 🔴 High |
| `pytorch/pytorch/.github/actions/ecr-login` | `@main` | 🔴 High |
| `pytorch/pytorch/.github/actions/setup-xpu` | `@main` | 🔴 High |
| `pytorch/pytorch/.github/workflows/_runner-determinator.yml` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/bc-lint` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/calculate-docker-image` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/check-disk-space` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/pull-docker-image` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/setup-nvidia` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/setup-python` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/setup-ssh` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/setup-uv` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/teardown-linux` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/update-commit-hash` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/update-viablestrict` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/upload-benchmark-results` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/actions/upload-claude-usage` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/workflows/linux_job_v2.yml` | `@main` | 🔴 High |
| `pytorch/test-infra/.github/workflows/validate-docker-images.yml` | `@main` | 🔴 High |

---

## Local Actions (No Pinning Needed) — 48 Actions

These reference actions within the same repository. They are already immutable at checkout time.

<details>
<summary>Click to expand local actions list</summary>

- `./.github/actions/build-external-packages`
- `./.github/actions/check-tpu`
- `./.github/actions/chown-workspace`
- `./.github/actions/download-build-artifacts`
- `./.github/actions/download-td-artifacts`
- `./.github/actions/ecr-login`
- `./.github/actions/filter-test-configs`
- `./.github/actions/get-workflow-job-id`
- `./.github/actions/pytest-cache-download`
- `./.github/actions/pytest-cache-upload`
- `./.github/actions/reuse-old-whl`
- `./.github/actions/setup-linux`
- `./.github/actions/setup-rocm`
- `./.github/actions/setup-win`
- `./.github/actions/setup-xpu`
- `./.github/actions/teardown-rocm`
- `./.github/actions/teardown-win`
- `./.github/actions/teardown-xpu`
- `./.github/actions/upload-sccache-stats`
- `./.github/actions/upload-test-artifacts`
- `./.github/actions/upload-utilization-stats`
- `./.github/workflows/_bazel-build-test.yml`
- `./.github/workflows/_binary-build-linux.yml`
- `./.github/workflows/_binary-test-linux.yml`
- `./.github/workflows/_binary-upload.yml`
- `./.github/workflows/_docs.yml`
- `./.github/workflows/_get-changed-files.yml`
- `./.github/workflows/_link_check.yml`
- `./.github/workflows/_linux-build.yml`
- `./.github/workflows/_linux-test.yml`
- `./.github/workflows/_linux-test-stable-fa3.yml`
- `./.github/workflows/_mac-build.yml`
- `./.github/workflows/_mac-test.yml`
- `./.github/workflows/_rocm-test.yml`
- `./.github/workflows/_vllm-benchmark.yml`
- `./.github/workflows/_vllm-build.yml`
- `./.github/workflows/_win-build.yml`
- `./.github/workflows/_win-test.yml`
- `./.github/workflows/_xpu-test.yml`
- `./.github/workflows/inductor-unittest.yml`
- `./.github/workflows/job-filter.yml`
- `./.github/workflows/llm_td_retrieval.yml`
- `./.github/workflows/target_determination.yml`
- `./pytorch/.github/actions/chown-workspace`
- `./pytorch/.github/actions/ecr-login`
- `./pytorch/.github/actions/filter-test-configs`
- `./pytorch/.github/actions/setup-linux`
- `./pytorch/.github/actions/test-pytorch-binary`

</details>

---

## Special Cases

### Actions with Multiple References

Some actions appear with both SHA-pinned and tag-pinned variants:

| Action | SHA-pinned refs | Tag-pinned refs |
|--------|-----------------|-----------------|
| `actions/checkout` | 1 | 1 (`@v4`) |
| `actions/download-artifact` | 2 | 2 (`@v4`, `@v4.1.7`) |
| `actions/setup-python` | 1 | 2 (`@v5`, `@v6`) |
| `actions/upload-artifact` | 2 | 2 (`@v4`, `@v4.4.0`) |
| `aws-actions/configure-aws-credentials` | 1 | 1 (`@v4`) |
| `ilammy/msvc-dev-cmd` | 1 | 1 (`@v1`) |
| `seemethere/upload-artifact-s3` | 1 | 1 (`@v5`) |

### Fork Action Reference

One action references a forked version:
- `izaitsevfb/claude-code-action@forked-pr-fix` — Branch-pinned fork, needs investigation

---

## M06 Scope Decision (Governance Checkpoint)

**Decision Date:** 2026-02-08  
**Decision:** Split M06 into two explicitly governed sub-tracks.

### M06-A: External Third-Party Actions (Proceed Now)

Pin **13 external, non-PyTorch actions** where:
- The action is owned by a third party
- The tag is versioned (`@v1`, `@v4`, etc.)
- SHA resolution is possible via GitHub Releases

**Actions in scope for M06-A:**
1. `actions/checkout@v4`
2. `actions/download-artifact@v4`
3. `actions/download-artifact@v4.1.7`
4. `actions/setup-python@v5`
5. `actions/setup-python@v6`
6. `actions/upload-artifact@v4`
7. `actions/upload-artifact@v4.4.0`
8. `anthropics/claude-code-action@v1`
9. `aws-actions/configure-aws-credentials@v4`
10. `ethanis/nitpicker@v1`
11. `ilammy/msvc-dev-cmd@v1`
12. `raven-actions/actionlint@v2`
13. `seemethere/upload-artifact-s3@v5`

### M06-B: PyTorch-Owned `@main` Actions (Explicitly Deferred)

**20 branch-pinned actions** under `pytorch/*` and `pytorch/test-infra/*` are:
- **Explicitly deferred**, not silently skipped
- Documented as an **intentional trust boundary**
- Tracked as deferral item `M06-V01`

**Rationale:**
> You cannot "pin" a moving internal repo without changing how PyTorch publishes actions.
> That requires maintainer coordination and is not a mechanical refactor.

**Security posture:**
- External actions = highest supply-chain risk → **addressed in M06-A**
- Internal PyTorch actions = within trust boundary → **deferred pending upstream policy**

---

## Next Steps

1. ✅ Inventory complete
2. ⏳ Resolve tag→SHA mappings for 13 external actions (manual browser lookup)
3. Apply pins mechanically with format: `action@<sha> # <original-ref>`
4. Verify with actionlint
5. Document M06-B deferral in M06_audit.md

---

**End of Before Inventory**

