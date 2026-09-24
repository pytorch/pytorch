---
orphan: true
---

# PyTorch Governance | Maintainer Guide

This page describes what is expected from module maintainers when handling issues and pull requests.
The contributor side of this process, including which parts of it are still being rolled out, is described in the
[Issue and PR Workflow](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#issue-and-pr-workflow).

## Expectations of Module Maintainers

- Ensure owned modules follow the PyTorch [design principles](design.md)
- Fully triage issues and triage pull requests within 1 week of creation
- Provide guidance, reviews, as well as assistance in closing high priority issues and pull requests
- Ensure proper documentation related to newly added APIs for owned modules
- Respond within 1 week for cross module issues from other module maintainers
- Attend the issue triage meeting regularly (at least once per month when issues for your module are tagged for discussion)

## Definitions

- **Module**: a GitHub repository within the PyTorch GitHub organization, or a directory within
  [pytorch/pytorch](https://github.com/pytorch/pytorch) that is mapped to a `module: *` or `oncall: *` label.
- **Fully triaged issue**: an issue that is closed or has one of the following labels:
  - `needs reproduction`: waiting for anyone to reproduce the issue, and for a maintainer to validate the reproduction.
  - `needs research`: waiting for anyone to provide evidence that the feature is useful or the bug is valid,
    and for a maintainer to decide whether it is worth pursuing.
  - `needs design`: waiting for anyone to propose a design for the feature or the fix, and for a maintainer to validate it.
  - `actionable`: the issue has enough detail for anyone to write a good PR (otherwise it should stay `needs design`),
    and the maintainer applying the label is ok with reviewing the corresponding change.
  - `won't fix`: the feature or bug is valid, but the ROI of implementing or fixing it is too low at this time.

  A fully triaged issue is also either assigned to the person who will move it forward (which can be the maintainer
  themselves), or assigned to no-one to indicate that we are looking for community help.
- **Triaged pull request**: a pull request with the `triaged` label and one reviewer assigned per module or team it touches.
- **High priority**: a status applied to issues or pull requests typically related to, but not limited to, bugs with
  core user functionality: regressions, hard crashes and silent correctness issues. Undocumented APIs and edge cases
  are not high priority by default, but the maintainer can still make them high priority if they think it is important.
  An issue is high priority if it is high priority for any of the modules it is labeled with.

## Triaging Issues

An issue labeled `triaged` and with one of your module labels, but with none of the labels above, needs your attention.

- Only mark an issue `actionable` if you are ok with reviewing the fix.
- Also mark especially simple `actionable` issues as `good first issue`.
- `won't fix` is a final state for the issue: explain the reason in a comment when applying it.
- When a contributor removes one of these labels and comments to request re-evaluation, look at the new information
  and pick the label again. Contributors who abuse this can lose the ability to change labels, up to being banned.

## Pre-reviewing Pull Requests

Pre-review is a quick review of the direction of the PR, to ensure it is worth the author's time to finalize it.
Until the automation for it lands, a maintainer with write access adds the `in progress` label once they accept the pre-review.

- It is the responsibility of the author to provide all the information needed for a quick assessment.
- You should be able to do a pre-review in under a minute. It is always ok to reject a PR at pre-review.
- PRs from authors without write access either have a corresponding `actionable` issue or name the maintainer who
  pre-approved them. Authors with write access may send PRs without one.

To assess a PR:

- Is the PR description clear, concise and reflective of the change?
- Is this PR solving a problem that is important?
- Is the approach of the PR clear and agreeable?
- Is this PR modular and simple enough to review, or does it need a design discussion?

Based on that:

- A design discussion is needed: close the PR and move the discussion to an issue.
- The author didn't provide succinct justifications to enable a fast pre-review: close the PR or move it to draft.
- A minor discussion or clarification in the PR description is needed: move the PR to draft.

Every reviewer assigned to the PR must accept the pre-review. Only one of them needs to do the full review at the end.

## Reviewing Pull Requests

PRs labeled `ready for review` have passed the automated review, or have the `no automated review` label, and are
waiting for one of their assigned reviewers.

- If the PR requires significant changes, use "Request changes". Until this is automated, also add the `in progress`
  label back so that the PR goes through the automated review again.
- Apart from this and the pre-review acceptance above, do not add `in progress` or `ready for review` by hand.
- Once you approve the PR, the author fixes CI and merges it with `@pytorchbot merge`.

If you are listed in [`.github/merge_rules.yaml`](https://github.com/pytorch/pytorch/blob/main/.github/merge_rules.yaml),
your own PRs may also be approved by [GreenLight](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#greenlight).

## Closing Issues and Pull Requests

Always give the reason when closing an issue or a pull request and, when it applies, link to
[Why was my issue or PR closed?](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#why-was-my-issue-or-pr-closed).
For example:

- "Closing this PR because it requires a design discussion that we should continue on the issue.
  Please comment on the linked issue with a summary of the status and approach to further discuss."
- "Closing this PR as it is not linked to an issue labeled `actionable`.
  See https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#why-was-my-issue-or-pr-closed for more details."
