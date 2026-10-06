---
name: distributed-triage
description: Sub-triages issues in the oncall:distributed queue by assigning distributed module labels, routing to sub-oncalls, and marking triaged. Use when an issue has been routed to oncall:distributed and needs second-level triage.
---

# Distributed Issue Triage Sub-Skill

This sub-skill picks up where the PT-level triage bot leaves off. It processes issues that already have the `oncall: distributed` label and performs second-level triage: routing to a distributed sub-oncall, classifying by module, and marking triaged.

## Contents
- [Tools and Output](#tools-and-output)
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

## Tools and Output

You read the issue and submit a plan; you never change the issue yourself.
`.github/workflows/distributed-triage-pi.yml` applies the plan in a separate step.

| Tool | Purpose |
|------|---------|
| `get_issue` | Issue title, body, labels, state, and author |
| `get_issue_comments` | Existing comments |
| `search_issues` | Similar or duplicate issues in the same repository |
| `read`, `grep`, `find`, `ls` | This skill's files |
| `submit_result` | The plan (required, call once as your last action) |

The plan has a `decision` (the step below that decided the outcome), the `labels` to add,
the `templates` to post, and your `reasoning`. The apply step enforces it:

- Labels are only added, never removed, and only from [distributed-labels.json](distributed-labels.json).
- An issue keeps at most one sub-oncall label: an existing one is never replaced, and a plan with two goes to `triage review`.
- `triaged` is dropped whenever the issue would also carry `triage review` or `needs reproduction`.
- `templates` are [templates.json](templates.json) keys posted verbatim; a template the bot already posted is not posted again.
- `ptd-bot-triaged` is added automatically (except for `high_priority`); do not list it.
- Nothing is ever closed.

---

## Distributed Triage Steps

### 0) Already Triaged by Human?

A human has fully classified the issue only when it has **BOTH**:
1. Any `module:` label listed in [distributed-labels.json](distributed-labels.json), AND
2. One of the sub-oncall labels: `oncall: distributed parallelisms`, `oncall: distributed infra`, or `oncall: distributed checkpointing`.

If both are present:
- `decision: "already_classified"` with `triaged` (the human classification is complete and confident)
- **STOP** — a human already classified this issue.

If only one is present (a module label without a sub-oncall, or a sub-oncall without a module label), triage is **incomplete** — proceed to Step 1. The PT-level triage bot can apply distributed module labels alongside `oncall: distributed`, but it does not pick the sub-oncall; that is your job.

*This step alone should clear a large portion of the backlog.*

### 1) Is This Actually a Distributed Issue?

Read the issue title, description, and comments. Determine whether the issue is actually related to distributed training.

**Signs it is NOT a distributed issue:**
- Single-GPU issue with no distributed code (e.g., `torch.nn` on one GPU, CUDA OOM on one device)
- Build/packaging issue (e.g., `undefined symbol: ncclAlltoAll` at `import torch` with no distributed code)
- Pure `torch.compile` issue with no distributed component
- Issue about a domain library (vision, text, audio) that happens to mention "distributed"

**If NOT a distributed issue:**
1. `decision: "not_distributed"` with `triage review` and the `not_distributed` template
2. `oncall: distributed` stays — let the human oncall re-route
3. **STOP**

### 2) Route to Distributed Sub-Oncall

Each issue carries **exactly ONE** sub-oncall label. If the issue already has one of the three sub-oncall labels (`oncall: distributed parallelisms`, `oncall: distributed infra`, or `oncall: distributed checkpointing`), keep it as-is — do NOT add a second sub-oncall, even if your own classification would have picked a different one. Use the existing sub-oncall to decide the next step (continue to Step 3 if it's `oncall: distributed parallelisms`; otherwise use `decision: "route"` with no new sub-oncall and STOP).

If no sub-oncall is present, apply exactly one based on the routing rules in [distributed-rubric.md](distributed-rubric.md):

| Sub-Oncall Label | When to Apply |
|-----------------|---------------|
| `oncall: distributed parallelisms` | FSDP, DDP, DTensor, tensor parallel, context parallel, pipeline parallel. **This is the default** when unsure. |
| `oncall: distributed infra` | c10d, process groups, collectives, NCCL/Gloo/MPI backends, elastic/torchrun, RPC, stores, distributed tools, DeviceMesh, symmetric memory |
| `oncall: distributed checkpointing` | Distributed checkpoint save/load, DCP, state_dict utilities, async checkpointing |

Use the routing decision tree and edge cases in [distributed-rubric.md](distributed-rubric.md) Section 1 to determine the correct sub-oncall.

**After routing to `oncall: distributed infra` or `oncall: distributed checkpointing`:**
- `decision: "route"` with that sub-oncall and `triaged` (the routing is a confident, complete outcome)
- **STOP** — the sub-oncall team owns further triage

**After routing to `oncall: distributed parallelisms`:**
- Continue to Step 3 for module classification

### 3) Classify Module

From the issue description, comments, code snippets, and stack traces, classify into one or more distributed modules. Consult the module classification signals in [distributed-rubric.md](distributed-rubric.md).

**Confidence-based actions:**

| Confidence | Criteria | Action |
|-----------|---------|--------|
| **HIGH or MEDIUM** | Explicit module mention, obvious API usage, or probable module based on context | `decision: "classify"` with the sub-oncall, `module:` label(s), and `triaged` |
| **LOW** | Cannot determine module — vague description, no code, no stack trace | `decision: "low_confidence"` with the sub-oncall and `triage review` (no `triaged` — punting to a human) |

**Rules:**
- You can apply multiple module labels when the issue spans modules (e.g., `module: fsdp` + `module: dtensor` for FSDP2 issues that hit DTensor bugs).
- When an issue has `oncall: pt2` already applied, do NOT remove it. Add distributed module labels alongside it.
- When the module is unclear, use `low_confidence` with `triage review` — do NOT guess a module label.

### 4) Type Labels

If the issue is not a bug report, add the appropriate type label:
- `feature` — wholly new functionality that does not exist today in any form
- `enhancement` — improvement to something that already works (e.g., performance optimization, better error messages, adding a native backend for an op that already runs via fallback)

Most distributed issues are bug reports — do not add a type label for bugs. If the issue says the operation "currently works" or "falls back to" a slower path, that is `enhancement`, not `feature`. If the enhancement is about performance, also add `module: performance`.

### 5) High Priority — REQUIRES HUMAN REVIEW

**CRITICAL:** If you believe an issue is high priority, you MUST:
1. Use `decision: "high_priority"` with `triage review` (the apply step leaves off `ptd-bot-triaged`, so the daily sweep re-surfaces it)

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

1. `decision: "needs_reproduction"` with `needs reproduction` and the `needs_distributed_reproduction` template

**Do NOT request reproduction when:**
- The issue already has a code snippet, script, or steps that someone could follow to reproduce
- The issue is a feature request (no repro needed)
- A multi-node script is provided (that counts as reproduction even if you can't run it locally)

---

## Constraints

**DO NOT:**
- Ask for `triaged` when you are NOT confident in the classification — i.e. any time the plan also adds `triage review` or `needs reproduction`, or in the §5 high-priority flow
- Ask for labels not in [distributed-labels.json](distributed-labels.json), including `high priority` — use `triage review` and let humans decide
- Ask for comments other than the templates in Step 1 (mislabel) or Step 6 (reproduction)

**DO:**
- Be conservative — when in doubt, use `triage review` for human attention
- Ask for `triaged` ONLY when you reach a confident, complete classification: a human already classified it (Step 0), a confident sub-oncall routing (Step 2), or a HIGH/MEDIUM-confidence module classification (Step 3)
- Pick the sub-oncall (Step 2) before module labels (Step 3)
- Read the full issue including comments before classifying
- Check the rubric's "Common Mislabel Traps" section before finalizing
