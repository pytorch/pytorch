---
name: distributed-triage
description: Sub-triages issues in the oncall:distributed queue by assigning distributed module labels, routing to sub-oncalls, and marking triaged. Use when an issue has been routed to oncall:distributed and needs second-level triage.
---

# Distributed Issue Triage Sub-Skill

This sub-skill picks up where the PT-level triage bot leaves off. It processes issues that already have the `oncall: distributed` label and performs second-level triage: routing to a distributed sub-oncall, classifying by module, and marking triaged.

## Contents
- [MCP Tools Available](#mcp-tools-available)
- [Reference Files](#reference-files)
- [Distributed Triage Steps](#distributed-triage-steps)
  - Step 0: Already Triaged by Human
  - Step 1: Is This Actually a Distributed Issue?
  - Step 2: Route to Distributed Sub-Oncall
  - Step 3: Classify Module
  - Step 4: Type Labels
  - Step 5: High Priority — REQUIRES HUMAN REVIEW
  - Step 6: Missing Reproduction
- [Constraints](#constraints)

**Distributed labels reference:** See [distributed-labels.json](distributed-labels.json) for the labels this skill is allowed to apply. **ONLY apply labels from this file.**

**Distributed triage rubric:** See [distributed-rubric.md](distributed-rubric.md) for detailed routing guidance, module classification signals, and confidence calibration.

**Response templates:** See [templates.json](templates.json) for distributed-specific comment templates.

---

## MCP Tools Available

Use these GitHub MCP tools for triage:

| Tool | Purpose |
|------|---------|
| `mcp__github__issue_read` | Get issue details, comments, and existing labels |
| `mcp__github__issue_write` | Apply labels or close issues |
| `mcp__github__add_issue_comment` | Add comment (only for reproduction requests or mislabel flags) |
| `mcp__github__search_issues` | Find similar issues for context |

---

## Distributed Triage Steps

### 0) Already Triaged by Human?

Check if the issue already has any `module:` label listed in [distributed-labels.json](distributed-labels.json).

If the issue already has one or more of these labels:
- Add `ptd-bot-triaged` label
- **STOP** — a human already classified this issue.

*This step alone should clear a large portion of the backlog.*

### 1) Is This Actually a Distributed Issue?

Read the issue title, description, and comments. Determine whether the issue is actually related to distributed training.

**Signs it is NOT a distributed issue:**
- Single-GPU issue with no distributed code (e.g., `torch.nn` on one GPU, CUDA OOM on one device)
- Build/packaging issue (e.g., `undefined symbol: ncclAlltoAll` at `import torch` with no distributed code)
- Pure `torch.compile` issue with no distributed component
- Issue about a domain library (vision, text, audio) that happens to mention "distributed"

**If NOT a distributed issue:**
1. Add `triage review` + `ptd-bot-triaged` labels
2. Post a comment using the `not_distributed` template from [templates.json](templates.json)
3. Do **NOT** remove `oncall: distributed` — let the human oncall re-route
4. **STOP**

### 2) Route to Distributed Sub-Oncall

Apply exactly ONE sub-oncall label based on the routing rules in [distributed-rubric.md](distributed-rubric.md):

| Sub-Oncall Label | When to Apply |
|-----------------|---------------|
| `oncall: distributed parallelisms` | FSDP, DDP, DTensor, tensor parallel, context parallel, pipeline parallel. **This is the default** when unsure. |
| `oncall: distributed infra` | c10d, process groups, collectives, NCCL/Gloo/MPI backends, elastic/torchrun, RPC, stores, distributed tools, DeviceMesh, symmetric memory |
| `oncall: distributed checkpointing` | Distributed checkpoint save/load, DCP, state_dict utilities, async checkpointing |

Use the routing decision tree and edge cases in [distributed-rubric.md](distributed-rubric.md) Section 1 to determine the correct sub-oncall.

**After routing to `oncall: distributed infra` or `oncall: distributed checkpointing`:**
- Add `ptd-bot-triaged`
- **STOP** — the sub-oncall team owns further triage

**After routing to `oncall: distributed parallelisms`:**
- Continue to Step 3 for module classification

### 3) Classify Module

From the issue description, comments, code snippets, and stack traces, classify into one or more distributed modules. Consult the module classification signals in [distributed-rubric.md](distributed-rubric.md).

**Confidence-based actions:**

| Confidence | Criteria | Action |
|-----------|---------|--------|
| **HIGH or MEDIUM** | Explicit module mention, obvious API usage, or probable module based on context | Add `module:` label(s) + `ptd-bot-triaged` |
| **LOW** | Cannot determine module — vague description, no code, no stack trace | Add `triage review` + `ptd-bot-triaged` |

**Rules:**
- You can apply multiple module labels when the issue spans modules (e.g., `module: fsdp` + `module: dtensor` for FSDP2 issues that hit DTensor bugs).
- When an issue has `oncall: pt2` already applied, do NOT remove it. Add distributed module labels alongside it.
- When the module is unclear, add `triage review` + `ptd-bot-triaged` — do NOT guess a module label.

### 4) Type Labels

If the issue is not a bug report, add the appropriate type label:
- `feature` — wholly new functionality that does not exist today in any form
- `enhancement` — improvement to something that already works (e.g., performance optimization, better error messages, adding a native backend for an op that already runs via fallback)

Most distributed issues are bug reports — do not add a type label for bugs. If the issue says the operation "currently works" or "falls back to" a slower path, that is `enhancement`, not `feature`. If the enhancement is about performance, also add `module: performance`.

### 5) High Priority — REQUIRES HUMAN REVIEW

**CRITICAL:** If you believe an issue is high priority, you MUST:
1. Add `triage review` label and do NOT add `ptd-bot-triaged`

Do NOT directly add `high priority` without human confirmation.

High priority criteria for distributed issues:
- Crash / segfault / illegal memory access in distributed code
- Silent correctness issue (wrong results from collectives, incorrect gradient sync)
- Regression from a prior version (e.g., FSDP worked in 2.x, broken in 2.y)
- Hang affecting multi-node training (NCCL timeout, deadlock in collectives)
- Data corruption during distributed checkpointing
- Internal assert failure in c10d or process group code
- Many users affected or core distributed component impacted

### 6) Missing Reproduction

If the issue lacks a minimal reproduction script:

1. Add `needs reproduction` + `ptd-bot-triaged` labels
2. Post a comment using the `needs_distributed_reproduction` template from [templates.json](templates.json)

**Do NOT request reproduction when:**
- The issue already has a code snippet, script, or steps that someone could follow to reproduce
- The issue is a feature request (no repro needed)
- A multi-node script is provided (that counts as reproduction even if you can't run it locally)

---

## Constraints

**DO NOT:**
- Close issues (only the PT-level bot or humans close issues)
- Remove existing labels — only add labels
- Remove `oncall: distributed` — it stays even if the issue is mislabeled
- Remove `oncall: pt2` — if already present, keep it
- Remove `bot-triaged` — it is applied by the parent skill and must stay
- Add labels not in [distributed-labels.json](distributed-labels.json)
- Add comments to issues except when using the templates in Step 1 (mislabel) or Step 6 (reproduction)
- Assign issues to users
- Add `high priority` directly — use `triage review` and let humans decide

**DO:**
- Be conservative — when in doubt, add `triage review` for human attention
- Add `ptd-bot-triaged` whenever the bot has processed the issue, regardless of confidence. Pair with `triage review` for LOW-confidence or uncertain cases so the cron sweep won't re-pick it. (Exception: §5 high-priority flow intentionally omits `ptd-bot-triaged`.)
- Always add a sub-oncall label (Step 2) before module labels (Step 3)
- Read the full issue including comments before classifying
- Check the rubric's "Common Mislabel Traps" section before finalizing

**Note:** `bot-triaged` is automatically applied by the parent skill's post-hook after any issue mutation. You do not need to add it manually.
