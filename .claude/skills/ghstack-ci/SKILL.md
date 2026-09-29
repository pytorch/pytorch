---
name: ghstack-ci
description: Manage CI for PyTorch ghstack stacks by running CI where its results are useful now and deferring other PRs with [no-ci]. Use when creating, submitting, updating, restacking, or landing a stack of PRs.
---

# CI for ghstack stacks

Keep CI usage low by running it where the results can guide the current
work. Explicitly mark deferred PRs with the `[no-ci]` title prefix, and
remove it when their results become useful. Follow explicit user CI
preferences.

Use the exact `[no-ci]` spelling described in
[Skip CI while iterating](../../../CONTRIBUTING.md#skip-ci-while-iterating).

## Choosing where to run CI

- Run CI on PRs being prepared for review or landing, or when its results
  are needed to diagnose a failure or validate a change on a CI-only platform.
- Run CI on an upper PR when it provides needed integration coverage for
  several changes together. Stack position alone is not a reason to defer it.
- Consider deferring speculative or rapidly changing PRs whose results will
  not affect the next decision, or upper PRs waiting on a known lower-stack
  fix. Keep CI enabled wherever it is needed to investigate that fix.

For stacks longer than five PRs, the bottom five are a useful starting
point when preparing to land from the bottom: consider deferring higher
PRs until their results are needed. Five is a hint to limit repeated CI
during iteration, not a quota or a cap. Use fewer or more as the work needs.
For example, in a stack A through H, CI might be useful on A through E
while F through H are still speculative. Enable G too if its integration
results are needed now; after A and B land, reassess F through H instead
of automatically enabling a fixed number of PRs.

Before GitHub writes or workflow reruns, follow
[CLAUDE.md](../../../CLAUDE.md) and [AI_POLICY.md](../../../AI_POLICY.md).
Show the exact PR numbers, old/new titles, and runs to rerun, and obtain
explicit approval for any actions not already approved. Existing approval
for those exact changes does not need to be repeated.

## Submitting or updating a stack

1. Identify the full stack in dependency order, from bottom to top, including
   commits that will become new PRs. Use commit ancestry and `Pull-Request`
   trailers, and refresh existing PR titles and states with `gh`. Exclude
   landed changes. GitHub can show a ghstack PR as closed after it lands;
   confirm landing on `main` rather than relying only on the `MERGED` state.
2. Choose PRs whose results are useful now. Prefix the other titles with
   `[no-ci] `, for example `[no-ci] Add another operator`. Preserve the rest of each title and
   avoid adding the prefix twice. Record the PR numbers and reasons for
   deferring them in the task notes.
3. Before creating new PRs with ghstack, put the prefix in the commit
   subjects for deferred PRs. The initial PR title comes from the subject,
   so setting it only after creation can let CI start.
4. For existing PRs, add or remove the prefix in both the live PR title
   and the corresponding commit subject before submitting updates.
   Ordinary `ghstack` submissions preserve GitHub titles; `ghstack -u`
   replaces them from commit subjects. Keep their prefix state consistent
   so a later `ghstack -u` cannot undo the CI decision. When rewording a
   commit, read and preserve its current `Pull-Request:` and
   `ghstack-source-id:` trailers, and follow the repository's submission rules.

Adding `[no-ci]` does not cancel CI that is already running.

## Enabling CI and advancing the stack

After updates, restacks, reorders, or lower PRs landing, refresh the stack
and reassess where CI is useful. To enable a PR, remove the leading prefix
and its separating space from both its live title and commit subject.
Preserve explicit user requests to defer CI. If a prefix's origin or
purpose is unknown in the current context, ask before removing it unless
the user has already explicitly authorized enabling CI for that PR.
Do not assume task notes from a previous session are available.

Enable CI after removing the title prefix: push an actual update with
ghstack, or rerun the latest workflows that failed at the `[no-ci]` gate
for that PR's current head. A title edit alone does not start CI.

Set `pr_number` to the selected PR and inspect its current checks:

```bash
gh pr checks "$pr_number" --repo pytorch/pytorch --json name,state,workflow,link,bucket,completedAt
```

Set `check_link` to a candidate failed gate's Actions `link`. For a link
with the path `/pytorch/pytorch/actions/runs/12345/job/67890`, the run ID
is `12345`. Extract it and inspect the run:

```bash
run_id="${check_link#*/actions/runs/}"
run_id="${run_id%%/*}"
gh pr view "$pr_number" --repo pytorch/pytorch --json headRefOid
gh run view "$run_id" --repo pytorch/pytorch --json headSha,conclusion,jobs
gh run view "$run_id" --repo pytorch/pytorch --log-failed
```

Confirm that the run's `headSha` matches the PR's `headRefOid` and its logs
show failure at the `[no-ci]` gate. Rerun each distinct selected run ID
once with `gh run rerun "$run_id" --repo pytorch/pytorch`; several checks
may belong to the same run. Avoid stale heads and unrelated failures.
Record the updated enabled and deferred PRs in the task notes when done.

## Before landing

Enable CI for every open PR included in the landing, including lower
dependencies. Remove `[no-ci]` from their live titles and commit subjects,
and wait for the required checks to pass on their current heads before
requesting the merge. The prefix intentionally fails a merge-blocking
gate, and the live PR title becomes the landed commit subject. CI on an
upper PR does not replace required checks on the lower PRs being landed.
