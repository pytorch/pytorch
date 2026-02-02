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
  - Step 2: Transfer
  - Step 2.5: PT2 Issues — Special Handling
  - Step 3: Redirect to Secondary Oncall
  - Step 4: Label the Issue
  - Step 5: High Priority — REQUIRES HUMAN REVIEW
  - Step 6: bot-triaged (automatic)
  - Step 7: Mark Triaged
- [V1 Constraints](#v1-constraints)

**Labels reference:** See [labels.json](labels.json) for the full catalog of 305 labels suitable for triage. This file excludes CI triggers, test configs, release notes, and deprecated labels.

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
| `ciflow/*` | CI job triggers for PRs only |
| `test-config/*` | Test suite selectors for PRs only |
| `release notes: *` | Auto-assigned for release notes |
| `ci-*`, `ci:*` | CI infrastructure controls |
| `sev*` | Severity labels require human decision |
| `merge blocking` | Requires human decision |
| Any label containing "deprecated" | Obsolete |

**If blocked:** When a label is blocked by the hook, add ONLY `triage review` and stop. A human will handle it.

These forbidden labels are enforced by a PreToolUse hook.

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

### 2) Transfer (domain library or ExecuTorch)

If the issue belongs in another repo (vision/text/audio/RL/ExecuTorch/etc.), transfer the issue and **STOP**.

### 2.5) PT2 Issues — Special Handling

When triaging PT2 issues (torch.compile, dynamo, inductor), see [pt2-triage-rubric.md](pt2-triage-rubric.md) for detailed labeling decisions.

**Key differences from general triage:**
- For PT2 issues, you MAY apply `module:` labels (e.g., `module: dynamo`, `module: inductor`, `module: dynamic shapes`)
- Use the rubric to determine the correct component labels
- Only redirect to `oncall: cpu inductor` for MKLDNN-specific issues; otherwise keep in PT2 queue

### 3) Redirect to Secondary Oncall

**CRITICAL:** When redirecting issues to an oncall queue (**critical** with the exception of PT2), apply exactly one `oncall: ...` label and **STOP**. Do NOT:
- Add any `module:` labels
- Mark it `triaged`
- Do any further triage work

The sub-oncall team will handle their own triage. Your job is only to route it to them.

#### Oncall Redirect Labels

| Label | When to use |
|-------|-------------|
| `oncall: jit` | TorchScript issues |
| `oncall: distributed` | Distributed training (DDP, FSDP, RPC, c10d) |
| `oncall: export` | torch.export issues |
| `oncall: quantization` | Quantization issues |
| `oncall: mobile` | Mobile (iOS/Android), excludes ExecuTorch |
| `oncall: profiler` | Profiler issues (CPU, GPU, Kineto) |
| `oncall: visualization` | TensorBoard integration |

**Note:** `oncall: cpu inductor` is a sub-queue of PT2. For general triage, just use `oncall: pt2`.

### 4) Label the issue (if NOT transferred/redirected)

Only if the issue stays in the general queue:
- Add 1+ `module: ...` labels based on the affected area
- If feature request: add `feature` (or `function request` for a new function or new arguments/modes)
- If small improvement: add `enhancement`

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
