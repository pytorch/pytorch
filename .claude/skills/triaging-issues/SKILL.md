---
name: triaging-issues
description: Triages GitHub issues by routing to oncall teams, applying labels, and closing questions. Use when processing new PyTorch issues or when asked to triage an issue.
hooks:
  PreToolUse:
    - matcher: "mcp__github__issue_write|mcp__github__update_issue"
      hooks:
        - type: command
          command: "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/skills/triaging-issues/scripts/validate_labels.py"
  PostToolUse:
    - matcher: "mcp__github__issue_write|mcp__github__update_issue|mcp__github__add_issue_comment|mcp__github__transfer_issue"
      hooks:
        - type: command
          command: "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/skills/triaging-issues/scripts/add_bot_triaged.py"
---

# PyTorch Issue Triage Skill

This skill helps triage GitHub issues by routing issues, applying labels, and leaving first-line responses.

## Contents
- [MCP Tools Available](#mcp-tools-available)
- [Labels You Must NEVER Add](#labels-you-must-never-add)
- [Issue Triage Steps](#issue-triage-for-each-issue)
  - Step 0: Already Routed — SKIP
  - Step 1: Question vs Bug/Feature
  - Step 1.5: Needs Reproduction — External Files
  - Step 2: Transfer
  - Step 2.5: PT2 Issues — Special Handling
  - Step 3: Redirect to Secondary Oncall
  - Step 4: Label the Issue
  - Step 5: High Priority — REQUIRES HUMAN REVIEW
  - Step 6: bot-triaged (automatic)
  - Step 7: Mark Triaged
- [V1 Constraints](#v1-constraints)

**Labels reference:** See [labels.json](labels.json) for the full catalog of 305 labels suitable for triage. **ONLY apply labels that exist in this file.** Do not invent or guess label names. This file excludes CI triggers, test configs, release notes, and deprecated labels.

**PT2 triage guide:** See [pt2-triage-rubric.md](pt2-triage-rubric.md) for detailed labeling guidance when triaging PT2/torch.compile issues.

**Response templates:** See [templates.json](templates.json) for standard response messages.

---

## MCP Tools Available

Use these GitHub MCP tools for triage:

| Tool | Purpose |
|------|---------|
| `mcp__github__issue_read` | Get issue details, comments, and existing labels |
| `mcp__github__issue_write` | Apply labels or close issues |
| `mcp__github__add_issue_comment` | Add comment (only for redirecting questions) |
| `mcp__github__search_issues` | Find similar issues for context |

---

## Labels You Must NEVER Add

| Prefix/Category | Reason |
|-----------------|--------|
| Labels not in `labels.json` | Only apply labels that exist in the allowlist |
| `ciflow/*` | CI job triggers for PRs only |
| `test-config/*` | Test suite selectors for PRs only |
| `release notes: *` | Auto-assigned for release notes |
| `ci-*`, `ci:*` | CI infrastructure controls |
| `sev*` | Severity labels require human decision |
| `merge blocking` | Requires human decision |
| Any label containing "deprecated" | Obsolete |
| `oncall: releng` | Not a triage redirect target. Use `module: ci` instead |

**If blocked:** When a label is blocked by the hook, add ONLY `triage review` and stop. A human will handle it.

These rules are enforced by a PreToolUse hook that validates all labels against `labels.json`.

### Never Override Human Labels

If a human has already applied labels (especially `ci: sev`, severity labels, or priority labels), do NOT remove or replace them. Your job is to supplement, not override.

---

## Issue Triage (for each issue)

### 0) Already Routed — SKIP

**If an issue already has ANY `oncall:` label, SKIP IT entirely.** Do not:
- Add any labels
- Add `triaged`
- Leave comments
- Do any triage work

That issue belongs to the sub-oncall team. They own their queue.

### 1) Question vs Bug/Feature

- If it is a question (not a bug report or feature request): close and use the `redirect_to_forum` template from `templates.json`.
- If unclear whether it is a bug/feature vs a question: request additional information using the `request_more_info` template and stop.

### 1.5) Needs Reproduction — External Files

Check if the issue body contains links to external files that users would need to download to reproduce.

**Patterns to detect:**
- File attachments: `.zip`, `.pt`, `.pth`, `.pkl`, `.safetensors`, `.onnx`, `.bin` files
- External storage: Google Drive, Dropbox, OneDrive, Mega, WeTransfer links
- Model hubs: Hugging Face Hub links to model files

**Action:**
1. **Edit the issue body** to remove/redact the download links
   - Replace with: `[Link removed - external file downloads are not permitted for security reasons]`
2. Add `needs reproduction` label
3. Use the `needs_reproduction` template from `templates.json` to request a self-contained reproduction
4. Do NOT add `triaged` — wait for the user to provide a reproducible example

### 1.55) Needs Reproduction — Other Cases

Also add `needs reproduction` when:
- The user reports a hardware-specific issue (e.g., specific GPU model) without a self-contained repro script
- The user references a specific model/checkpoint/dataset that is not publicly runnable in a few lines
- The issue describes version-upgrade breakage but only provides a high-level description without a minimal script
- The repro depends on a specific training setup, distributed environment, or non-trivial infrastructure

### 1.6) Edge Cases & Numerical Accuracy

If the issue involves extremal values or numerical precision differences:

**Patterns to detect:**
- Values near `torch.finfo(dtype).max` or `torch.finfo(dtype).min`
- NaN/Inf appearing in outputs from valid (but extreme) inputs
- Differences between CPU and GPU results
- Precision differences between dtypes (e.g., fp32 vs fp16)
- Fuzzer-generated edge cases

**IMPORTANT — avoid keyword-triggered mislabeling:**

Label based on the **root cause**, not keywords that appear in the error or title. A keyword tells you what failed, not why.

- An `undefined symbol: ncclAlltoAll` error at `import torch` is a **packaging** issue (`module: binaries`), not a distributed training bug — the user never ran distributed code.
- A `nan` in a parameter name or tolerance check is not `module: NaNs and Infs` unless the bug is actually about NaN propagation.
- A stack trace mentioning `autograd` does not mean `module: autograd` — check whether the bug is in autograd itself or just on the call path.
- A test failure with tolerance thresholds is `module: tests`, not `module: numerical-stability`.

Ask: "Where would the fix need to be made?" That determines the label.

**Action:**
1. Add `module: edge cases` label
2. If from a fuzzer, also add `topic: fuzzer`
3. Use the `numerical_accuracy` template from `templates.json` to link to the docs
4. If the issue is clearly expected behavior per the docs, close it with the template comment

### 2) Transfer (domain library or ExecuTorch)

If the issue belongs in another repo (vision/text/audio/RL/ExecuTorch/etc.), transfer the issue and **STOP**.

### 2.5) PT2 Issues — Special Handling

**PT2 is NOT a redirect.** `oncall: pt2` is not like the other oncall labels in Step 3. PT2 issues continue through Steps 4–7 for full triage — add `oncall: pt2`, then proceed to label with `module:` labels, mark `triaged`, etc.

See [pt2-triage-rubric.md](pt2-triage-rubric.md) for detailed labeling decisions on which `module:` labels to apply.

### 3) Redirect to Secondary Oncall

**CRITICAL:** When redirecting issues to a **non-PT2** oncall queue, apply exactly one `oncall: ...` label and **STOP**. Do NOT:
- Add any `module:` labels
- Mark it `triaged`
- Do any further triage work

The sub-oncall team will handle their own triage. Your job is only to route it to them.

#### Oncall Redirect Labels

| Label | When to use |
|-------|-------------|
| `oncall: jit` | TorchScript issues |
| `oncall: distributed` | Distributed training (DDP, FSDP, RPC, c10d, DTensor, DeviceMesh, symmetric memory, context parallel, pipelining) |
| `oncall: export` | torch.export issues |
| `oncall: quantization` | Quantization issues |
| `oncall: mobile` | Mobile (iOS/Android), excludes ExecuTorch |
| `oncall: profiler` | Profiler issues (CPU, GPU, Kineto) |
| `oncall: visualization` | TensorBoard integration |

**Common routing mistakes to avoid:**
- **MPS ≠ Mobile.** MPS (Metal Performance Shaders) is the macOS/Apple Silicon GPU backend. Do NOT route MPS issues to `oncall: mobile`. MPS issues stay in the general queue with `module: mps`.
- **DTensor → `oncall: distributed`.** DTensor issues should always be routed to `oncall: distributed`, even if they don't mention DDP/FSDP.
- **ONNX → `module: onnx`.** There is no `oncall: onnx`. Use `module: onnx` and keep in the general queue.
- **CI/releng → `module: ci`.** Do not use `oncall: releng`. Use `module: ci` for CI infrastructure issues.
- **torch.compile + distributed.** When `torch.compile` mishandles a distributed op (e.g., `dist.all_reduce`), the issue typically needs BOTH `oncall: pt2` and `oncall: distributed` since the fix may span both codebases.

**Note:** `oncall: cpu inductor` is a sub-queue of PT2. For general triage, just use `oncall: pt2`.

### 4) Label the issue (if NOT transferred/redirected)

Only if the issue stays in the general queue:
- Add 1+ `module: ...` labels based on the affected area
- Prefer specific labels over general ones when both exist. Check `labels.json` descriptions for guidance on when a specific label supersedes a general one (e.g., `module: sdpa` instead of `module: nn` for SDPA issues, `module: flex attention` instead of `module: nn` for flex attention).
- `feature` — wholly new functionality that does not exist today in any form
- `enhancement` — improvement to something that already works (e.g., adding a native backend kernel for an op that already runs via fallback/composite, performance optimization, better error messages). If the enhancement is about performance, also add `module: performance`.
- `function request` — a new function or new arguments/modes for an existing function
- If the issue says the operation "currently works" or "falls back to" a slower path, that is `enhancement`, not `feature`

**Commonly missed labels — always check for these:**

| Condition | Label |
|-----------|-------|
| Segfault, illegal memory access, SIGSEGV | `module: crash` |
| Performance issue: regression, slowdown, or optimization request | `module: performance` |
| Issue on Windows | `module: windows` |
| Previously working feature now broken | `module: regression` |
| Broken docs/links that previously worked | `module: docs` + `module: regression` (NOT `enhancement`) |
| Issue about a test failing (not the underlying functionality) | `module: tests` |
| Backward pass / gradient computation bug | `module: autograd` (in addition to the op's module label) |
| `torch.linalg` ops or linear algebra ops (solve, svd, eig, inv, etc.) | `module: linear algebra` |
| `has workaround` | Only add when the workaround is **non-trivial and non-obvious**. If the issue is "X doesn't work for non-contiguous tensors," calling `.contiguous()` is the tautological inverse of the bug, not a workaround. A real workaround is something like installing a specific package version, adding a synchronization point, inserting `gc.collect()`, or using a different API that isn't obviously implied by the bug description. |

**Label based on the actual bug, not keywords.** Read the issue to understand what is actually broken. A bug about broadcasting that happens to mention "nan" in a parameter name is a frontend bug, not a NaN/Inf bug.

### 5) High Priority — REQUIRES HUMAN REVIEW

**CRITICAL:** If you believe an issue is high priority, you MUST:
1. Add `triage review` label and do not add `triaged`

Do NOT directly add `high priority` without human confirmation.

High priority criteria:
- Crash / segfault / illegal memory access
- Silent correctness issue (wrong results without error)
- Regression from a prior version
- Internal assert failure
- Many users affected
- Core component or popular model impact

### 6) bot-triaged (automatic)

The `bot-triaged` label is automatically applied by a post-hook after any issue mutation. You do not need to add it manually.

### 7) Mark triaged

If not transferred/redirected and not flagged for review, add `triaged`.

---

## V1 Constraints

**DO NOT:**
- Close bug reports or feature requests automatically
- Close issues unless they are clear usage questions per Step 1
- Assign issues to users
- Add `high priority` directly without human confirmation
- Add module labels when redirecting to oncall
- Add comments to bug reports or feature requests, except a single info request when classification is unclear

**DO:**
- Close clear usage questions and point to discuss.pytorch.org (per step 1)
- Be conservative - when in doubt, add `triage review` for human attention
- Apply type labels (`feature`, `enhancement`, `function request`) when confident
- Add `triaged` label when classification is complete

**Note:** `bot-triaged` is automatically applied by a post-hook after any issue mutation.
