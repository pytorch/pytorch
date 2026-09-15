---
name: ghstack-ci
description: Manage CI for PyTorch ghstack stacks by enabling the bottom five open PRs and deferring the rest with [no-ci]. Use when creating, submitting, updating, restacking, or landing a stack of PRs.
---

# CI for ghstack stacks

Keep CI enabled for the bottom five open PRs in a stack. Add the `[no-ci]`
title prefix to every PR above them, and remove it as those PRs move into
the bottom five after lower PRs land. Explicit user CI preferences take
precedence over this default.

Use the PR-title prefix described in
[Skip CI while iterating](../../../CONTRIBUTING.md#skip-ci-while-iterating).

## Submitting or updating a stack

1. Identify the full stack in dependency order, from bottom to top, including
   commits that will become new PRs. Use commit ancestry and `Pull-Request`
   trailers, and refresh existing PR titles and states with `gh`. Count open
   PRs and planned new PRs, excluding landed changes. GitHub can show a
   ghstack PR as closed after it lands; confirm landing on `main` rather
   than relying only on the `MERGED` state.
2. Leave the bottom five eligible for CI. Prefix the titles of the sixth
   and all higher PRs with `[no-ci] `, for example
   `[no-ci] Add another operator`. Preserve the rest of each title and
   avoid adding the prefix twice. Record which PRs this policy deferred
   in the task notes so later updates can distinguish these prefixes from
   independent user requests to disable CI.
3. Before creating new PRs with ghstack, put the prefix in the commit
   subjects for the PRs above the first five. The initial PR title comes
   from the subject, so setting it only after creation can let CI start.
4. For existing PRs, edit their live titles before submitting updates.
   Ordinary `ghstack` submissions preserve GitHub titles. Before using
   `ghstack -u`, synchronize the commit subjects with the intended title
   prefixes, since that option replaces PR titles from commit subjects.
   Preserve `Pull-Request:` and `ghstack-source-id:` trailers when
   rewording commits, and follow the repository's submission rules.

Adding `[no-ci]` does not cancel CI that is already running.

## Advancing the stack

After lower PRs land, refresh the stack and recompute the bottom five
remaining open PRs. Remove the leading prefix and its separating space
from newly eligible PRs that this policy deferred. Preserve prefixes the
user independently requested. Keep the prefix on every PR still above
the first five, and apply the same calculation after a restack or reorder.
If five or fewer PRs remain, all are eligible for CI.

Enable CI after removing the title prefix: push an actual update with
ghstack, or rerun the latest workflows that failed at the `[no-ci]` gate
for that PR's current head. A title edit alone does not start CI. Use
`gh pr checks` to identify the relevant runs, then rerun each workflow
once; avoid rerunning stale heads or unrelated failures.

With the PR number, complete updated title, and relevant run ID selected:

```bash
gh pr edit "$pr_number" --repo pytorch/pytorch --title "$new_title"
gh pr checks "$pr_number" --repo pytorch/pytorch --json name,state,workflow,link,bucket,completedAt
gh run rerun "$run_id" --repo pytorch/pytorch
```

Record the updated enabled and deferred PRs in the task notes when done.

## Example

For a stack ordered A through H, with A at the bottom:

| Landed PRs | Open PRs with CI enabled | Open PRs prefixed with `[no-ci]` |
| --- | --- | --- |
| None | A, B, C, D, E | F, G, H |
| A, B | C, D, E, F, G | H |
| A, B, C | D, E, F, G, H | None |
