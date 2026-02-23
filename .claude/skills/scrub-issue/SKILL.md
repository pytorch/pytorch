---
name: scrub-issue
description: Fetch, analyze, reproduce, and minimize GitHub issue reproductions. Use when asked to check if an issue reproduces, minimize a repro, analyze a bug report, or scrub/triage an issue for reproducibility.
---

# Minimize Issue Reproduction

Fetch a GitHub issue, evaluate whether it has a reasonable repro, check if it
still reproduces, and systematically minimize the repro to the smallest possible
self-contained script.

## Tools

Assume the current environment is correct and run `python` directly. Only use
`conda run -n <env>` for version bisection (step 5a) where you need to
temporarily use a different environment. Use the Bash tool's `timeout`
parameter to enforce timeouts when running repro scripts.

- `gh issue view <NUMBER> --repo pytorch/pytorch` to fetch the issue body
- `gh issue view --comments <NUMBER> --repo pytorch/pytorch` to fetch comments
- `python <script>` to run repro scripts
- `gh issue comment <NUMBER> --repo pytorch/pytorch --body <BODY>` to comment
- `gh issue edit <NUMBER> --repo pytorch/pytorch --add-label <LABEL>` to add labels

Multiple `gh issue edit` flags can be combined in a single command (e.g.
`--add-label "bug,help wanted" --add-assignee "@me"`). Prefer batching
edits into one command to minimize API calls and reduce the chance of
auto-subscribing to notifications.

### Preserving notification subscription state

Modifying an issue (commenting, adding labels) auto-subscribes you to
notifications. Use `tools/stale_issues.py` to save and restore subscription
state:

1. **Before the first modification**, save the current state. This can be
   run in parallel with fetching the issue body/comments, but it must
   **complete** before any `gh issue edit`, `gh issue comment`, or
   `gh issue close` command is executed:
   ```
   python tools/stale_issues.py subscription save <NUMBER>
   ```
2. **After the last modification**, restore the saved state — **unless the
   issue was closed** (via `gh issue close`), in which case skip the restore
   so the user stays subscribed to follow any responses to the closure. The
   restore must be the very last GitHub API call — do not run it in the
   background or in parallel with any `gh issue edit` or `gh issue comment`
   commands:
   ```
   python tools/stale_issues.py subscription restore <NUMBER>
   ```

If either the save or restore command fails, warn the user and continue
without the save/restore mechanism.

**Important**: Never run `gh issue edit`, `gh issue comment`, `gh issue close`,
or subscription save/restore commands in the background. These must all run in
the foreground so their completion can be verified before proceeding.
If commenting or editing fails because the issue is locked, report this to the
user and skip the modification.

### Security review checklist

Before running any repro code, check for the following concerns:

- Network requests to untrusted URLs (requests, urllib, curl, wget)
- File operations outside `/tmp/`
- Shell command execution (os.system, subprocess, eval, exec) — but
  `subprocess` used to launch `torchrun` or `mp.spawn` for distributed repros
  is expected and not a concern
- Downloading or loading external files (model weights, pickled objects, data
  files) — especially `torch.load` on untrusted `.pt`/`.pth` files
- Obfuscated code (base64-encoded strings, encoded bytes, unusual escapes)
- Package installation (pip install, conda install)
- Environment variable manipulation that could affect the host system — but
  setting `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`,
  `CUDA_VISIBLE_DEVICES`, or other standard PyTorch/CUDA env vars is expected
  and not a concern

If any of these are present, explain the concern to the user and ask whether
to proceed, skip, or modify the repro to remove the risky parts. If the user
chooses to skip, still refresh the `triaged` label timestamp (remove and
re-add, or just add if not present) before reporting that the analysis is
finished.

## Steps

### 1) Fetch the issue

Fetch the issue body and comments in parallel. Identify the reported repro
script and error. If multiple repros are present, prefer the most recent one
from the original poster. If a commenter has posted a strictly shorter and
more self-contained repro that doesn't require additional context from the
issue description, prefer that one. Note which repro you selected. If the
repro code is only present in screenshots or images rather than copyable text,
stop and report this to the user.

### 2) Check if the issue is actionable

Before investing effort in reproduction, check whether the issue is actionable.
**Do not proceed to later steps** if any of the following apply:

- **Already closed or resolved** in comments. Report to the user and stop.
  Do not modify labels on closed issues.
- **Duplicate** of another issue (linked or obviously the same bug)
- **Not a bug report** (feature request, question, discussion, refactoring /
  code cleanup task). If the issue is a feature request and doesn't already
  have the `feature` label, add it via `gh issue edit`. If the issue is a
  better-engineering / refactoring task and doesn't already have the
  `better-engineering` label, add it via `gh issue edit`. If the issue
  includes a repro script that demonstrates the current behavior, apply the
  security review checklist first, then run it to verify the behavior
  persists. If the repro needed to be modernized (e.g. updated imports for
  renamed APIs), or if you verified that the behavior still persists, comment
  on the issue with the findings and updated repro. After adding any
  applicable labels (and optionally running/commenting on a repro), report
  the analysis to the user and do not proceed to later steps.
- **Tracking/meta issue** (umbrella issue tracking multiple bugs, burn-down
  lists, improvement proposals without a specific repro). If the issue doesn't
  already have the `tracker` label, add it via `gh issue edit`. Then stop and
  report to the user.
- **Requires unavailable hardware** (specific GPU models, TPUs, multi-node) with
  no path to simplify. Note: CUDA is available on the current machine, so
  single-GPU CUDA repros can be run directly.

For non-actionable issues that are old (more than ~1 year), have no assignees,
no recent progress, and are underspecified or lack concrete motivation, suggest
closing them as "not planned" (`gh issue close --reason "not planned"`) with a
comment explaining the rationale. Ask the user before closing.

If the issue is not actionable and no GitHub-visible modification was made (no
label added, no comment posted — saving subscription state does not count),
refresh the `triaged` label to update the issue's
"last updated" timestamp. A single `gh issue edit` with both `--remove-label`
and `--add-label` doesn't work because the remove and add cancel each other
out. Instead, chain both edits in a single Bash tool call:
`gh issue edit ... --remove-label triaged && gh issue edit ... --add-label triaged`.
If the issue doesn't have the `triaged` label, just add it.

Then summarize why the issue is not actionable. If a label or other update was
made, just report that the analysis is finished. Only ask the user how to
proceed if no update was made and the situation is ambiguous. Always ask the
user before closing an issue.

### 3) Analyze the repro

Evaluate whether the issue has a reasonable repro:

- Is there a code snippet that can be run?
- Are the dependencies available (CUDA, distributed, specific hardware)?
  Note: `torch.distributed` repros often don't require special hardware — they
  can be launched with `torchrun --nproc_per_node=1` or `mp.spawn` on a single
  machine.
- Is the expected error described?
- Is the repro self-contained or does it need external data/models?

If there is no repro code at all, or the issue is missing critical info (no
error message, no description of expected vs actual behavior), add the
`needs reproduction` label (if not already present) via `gh issue edit`, then
stop and report to the user. Do not attempt to write a repro from the
description without being asked.

Conversely, if the issue already has the `needs reproduction` label but does
have a valid repro, remove the label via `gh issue edit --remove-label
"needs reproduction"`.

Apply the security review checklist (see above) to the repro before running it.

If the repro requires third-party packages that are not installed (e.g.
`transformers`, `torchvision`, `numpy`), stop and ask the user how to proceed
rather than installing them yourself.

Summarize your assessment before proceeding.

### 4) Check for recent verification

If a comment from the last six months already confirms the issue still
reproduces (with a matching error and a reasonable repro), stop and ask the
user whether they want to re-verify or skip ahead to minimization.

### 5) Check if it still reproduces

Extract the repro code into a temporary file under `/tmp/` and run it with a
timeout of 120 seconds. For repros involving `torch.compile` that call compile
multiple times in the same process, add `torch._dynamo.reset()` between
invocations to reset in-memory Dynamo state. This is unnecessary for scripts
that compile once and exit.

If the script times out, consider whether a hang is the reported bug or an
unrelated issue, and report to the user.

Record the PyTorch version before running
(`python -c "import torch; print(torch.__version__)"`) for inclusion in the
report.

Check both the exit code and output to determine the result. An exit code > 128
indicates the process was killed by a signal (e.g. segfault = 139, OOM kill =
137) — this is a valid crash reproduction even without a Python traceback. If
the repro crashes with a CUDA out-of-memory error and OOM is not the reported
bug, try reducing tensor sizes before concluding it doesn't reproduce. For
correctness bugs (wrong numerical results rather than crashes), the repro
should include an assertion that fails when the bug is present. If the
original repro only prints output without asserting, add a simple assertion
based on the expected behavior described in the issue (e.g.
`assert torch.allclose(actual, expected)`) so the repro has a clear
pass/fail signal.

If the result is inconsistent across runs, run 3-5 times to assess flakiness.
For non-deterministic bugs, try setting `PYTHONHASHSEED=0` and a fixed
`torch.manual_seed` to stabilize reproduction. Report the success/failure ratio
to the user. Flaky repros are still valid bugs — note the flakiness in the
issue comment (step 8) and include the success/failure ratio.

Three possible outcomes (to determine which outcome applies, match on the
exception class and a distinctive substring of the error message — the
substring should be specific enough to identify the bug, e.g.
`RuntimeError: expected scalar type Float` rather than just `RuntimeError`):

- **Same error as reported**: the bug still reproduces. Continue to step 6.
- **No error (passes)**: the bug may have been fixed. Try to identify the
  fixing PR (see step 5a), then report to the user. The issue will be closed
  in step 8.
- **Different error**: distinguish between setup issues (missing import, renamed
  API) that can be fixed and genuinely different bugs. If the error is due to
  API changes between the reported version and the current version (renamed
  functions, moved modules, changed signatures), adapt the repro to use the
  current API while preserving the original intent. If the error is unclear,
  consider re-running with `TORCH_LOGS=+dynamo` or other relevant logging flags
  for more diagnostic output. Report genuinely different errors to the user.

### 5a) Identify when and how it was fixed

When the bug no longer reproduces, try to determine which version fixed it and
which PR introduced the fix.

**Version bisection**: Check if conda environments named `pytorch-<version>`
(e.g. `pytorch-2.6`, `pytorch-2.8`) are available (`conda env list | grep
pytorch-`). If they exist, binary-search across them to find the first version
where the bug is fixed. Run from `/tmp` in a subshell and clear `PYTHONPATH`
to avoid picking up the local source tree (which would cause `torch._C` import
errors): `(cd /tmp && PYTHONPATH= conda run -n pytorch-<version> python
/tmp/repro_....py)`. To speed up bisection, pick 2-3 evenly-spaced probe
points from the candidate range and test them in parallel each round (e.g.
if candidates are 2.2 through 2.8, test 2.4 and 2.6 simultaneously to split
the range into thirds). If no versioned conda environments are
available, skip bisection and just report that the bug no longer reproduces on
the current version.

**PR identification**: Try to find the specific PR that fixed the bug. Use
version control blame on the relevant fix code to find the changeset, then look
up the commit message for the PR number (format `(#NNNNN)`). Alternatively,
search version control history for commits touching the relevant file with a
related keyword. If this doesn't yield a clear answer quickly, just report the
version — don't spend extra time on PR identification.

### 6) Minimize the repro

Save the original working repro to `/tmp/repro_<issue_number>_original.py`
before making any changes.

First, assess whether the repro is already reasonably minimal. Only minimize if
the repro has significant unnecessary complexity (large models, unused code
paths, unnecessary dependencies, etc.). When counting complexity, don't count
irreducible boilerplate — e.g. a tensor subclass definition that only contains
the required dunder methods (`__new__`, `__init__`, `__torch_dispatch__`,
`__tensor_flatten__`, `__tensor_unflatten__`) is not reducible even if it's
20+ lines. Focus on whether the trigger code and model/setup complexity can
be meaningfully reduced. If the only possible "simplifications" are cosmetic
(inlining variables, removing `__repr__`), skip minimization.

Systematically reduce the repro by testing whether each simplification still
triggers the same error. Use a shorter timeout (30-60 seconds) during
minimization since simplified repros should run faster. Run multiple candidate
simplifications in parallel when they are independent (i.e. they modify
non-overlapping parts of the code and neither depends on what the other
removes).

**Reduction strategies (apply in roughly this order):**

1. **Remove unnecessary imports and setup** (distributed init, env vars, logging)
2. **Shrink the model** (replace large modules with minimal equivalents)
3. **Remove the class/module wrapper** if a bare function suffices
4. **Reduce tensor sizes** (large dims → small dims like 4 or 8)
5. **Remove device/dtype requirements** (try CPU and float32 first)
6. **Simplify the computation** (replace complex ops with minimal ones that still trigger the bug)
7. **Remove unnecessary control flow** (branches, loops, conditions)
8. **Try simpler backends** (e.g. `aot_eager` instead of inductor) if the bug is not backend-specific

After each round, verify the error still reproduces — same exception class and
a distinctive error message substring as described in step 5 (minor traceback
differences are fine). For correctness bugs, preserve the assertion that demonstrates the
wrong result and verify it still shows the same incorrect behavior. When merging multiple successful parallel simplifications, verify
the combined result still reproduces since independent simplifications can
interact. Stop minimizing when the repro is under ~20 lines of non-blank
non-import code, or when two consecutive rounds (where a round is one full
pass through the applicable reduction strategies) fail to simplify further.

### 7) Apply trivial fixes

If the analysis reveals a trivial fix (e.g. removing a stale `xfailIfTorchDynamo`
or `expectedFailure` annotation from a test because the underlying issue is
fixed), report the fix to the user and ask whether to apply it. Do not modify
source files without the user's approval. In step 8, mention the fix
regardless of whether it was applied or declined.

### 8) Report findings

If the repro was minimized, save it to `/tmp/repro_<issue_number>.py` so the
user can run it directly.

Present findings to the user including:
- Whether the bug still reproduces (and on what PyTorch version)
- The fixing PR, if identified (only if the bug no longer reproduces)
- The minimized repro script (only if we minimized it)
- The necessary conditions to trigger the bug
- Any trivial fix identified in step 7 (whether applied or not)
- Recommended next action (e.g. "still a valid bug", "appears fixed", "needs
  more info from reporter")

After presenting findings, always comment on the issue with the results.
Keep the comment concise — don't repeat information already on the issue.
Only include a repro if it was materially changed from the original (e.g.
minimized, modernized imports, fixed to run on current API). Only include
trigger conditions if they are new findings not already discussed in prior
comments. If the only finding is "still reproduces" or "no longer reproduces",
a short comment is sufficient.

```
This issue [still reproduces / no longer reproduces] on PyTorch <version>.

[If fixed and PR identified:]
Fixed by #NNNNN.

[If minimized or modernized — only include repro if changed from original:]
Minimized repro:

\`\`\`python
<repro script>
\`\`\`

[Only if new findings about trigger conditions:]
All of the following are necessary to trigger the bug:
- <condition 1>
- <condition 2>

(Analysis done by Claude.)
```

If the bug no longer reproduces, after commenting close the issue:
`gh issue close <NUMBER> --repo pytorch/pytorch --reason completed --comment "Closing as this was fixed in PyTorch <version>."`.
Do not ask the user before closing — fixed bugs should always be closed.

After the last GitHub modification, restore the notification subscription
state (see "Preserving notification subscription state" above) — unless the
issue was closed, in which case skip the restore.
