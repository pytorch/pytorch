# PyTorch Governance | Maintainer Guide

This page describes what is expected from module maintainers when handling issues and pull requests.
The contributor side of this process, including which parts of it are still being rolled out, is described in the
[Issue and PR Workflow](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#issue-and-pr-workflow).
Parts of this process that are not automated yet are marked with **TEMPORARY:** below.

## tl;dr: Expectations of Module Maintainers

- Ensure owned modules follow the PyTorch [design principles](design.md)
- Fully triage issues within 1 week of creation. Example triage queue for autograd module [here](https://github.com/pytorch/pytorch/issues?q=is%3Aissue%20is%3Aopen%20label%3Atriaged%20label%3A%22module%3A%20autograd%22%20-label%3A%22needs%20reproduction%22%20-label%3A%22needs%20research%22%20-label%3A%22needs%20design%22%20-label%3Aactionable%20-label%3A%22won%27t%20fix%22)
  (update the query to the specific module you're interested in).
- Ensure PRs assigned to your module or to you are moved through the review process (reviewed, or closed while further
  discussion happens on the issue). Example review queue
  [here](https://github.com/pytorch/pytorch/pulls?q=is%3Apr%20is%3Aopen%20review-requested%3A%40me%20label%3A%22ready%20for%20review%22%20-label%3A%22missing%20actionable%20issue%22).
- Provide guidance, reviews, as well as assistance in closing high priority issues and pull requests
- Ensure proper documentation related to newly added APIs for owned modules
- Respond within 1 week for cross module issues from other module maintainers
- Attend the issue triage meeting regularly (at least once per month when issues for your module are tagged for discussion)

## Definitions

- **Module**: a subset of [pytorch/pytorch](https://github.com/pytorch/pytorch) that is mapped to a `module: *` or
  `oncall: *` label, or a GitHub repository within the PyTorch GitHub organization.
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

## Module and Oncall Labels

Both `module: *` and `oncall: *` labels map to a module. Issues and pull requests with a `module: *` label are
discussed in the main triage meeting of the maintainers, while `oncall: *` labels correspond to larger areas that have
their own dedicated team running a regular triage discussion for them.

## Triaging Issues

An issue labeled `triaged` and with one of your module labels, but with none of the labels above, needs your attention.
For example, [this search](https://github.com/pytorch/pytorch/issues?q=is%3Aissue%20is%3Aopen%20label%3Atriaged%20label%3A%22module%3A%20autograd%22%20-label%3A%22needs%20reproduction%22%20-label%3A%22needs%20research%22%20-label%3A%22needs%20design%22%20-label%3Aactionable%20-label%3A%22won%27t%20fix%22)
lists the issues waiting for triage for `module: autograd`. Update the `module: autograd` filter to the module you are
working on.

- Only mark an issue `actionable` if you are ok with reviewing the fix.
- Also mark especially simple `actionable` issues as `good first issue`.
- `won't fix` is a final state for the issue: explain the reason in a comment when applying it.
- When a contributor removes one of these labels and comments to request re-evaluation, look at the new information
  and pick the label again. Contributors who abuse this can lose the ability to change labels, up to being banned.

**TEMPORARY:** `@pytorchbot` cannot remove labels yet, so contributors cannot remove a label themselves to request
re-evaluation. They comment on the issue and mention the maintainer who applied the label instead.

## Reviewing Pull Requests

When a PR is in your `ready for review` queue, you can expect that:

- The automated review ([pr-review skill](https://github.com/pytorch/pytorch/blob/main/.agents/skills/pr-review/SKILL.md)) <!-- @lint-ignore -->
  ran and did not flag anything blocking.
- All CI has finished and Dr. CI classified every failure as unrelated to the PR. Workflows waiting for a maintainer to
  approve their run do not block this, so you may still need to approve CI.
- Every comment from a maintainer has been addressed, either by a code change that fixes the specific problem or by a
  reply explaining why no change is needed.

PRs labeled `no automated review` skip these checks and move directly to `ready for review`. Adding the label
withdraws the automated review's change requests, so anyone who can label the PR can clear them; your review is
then the only check.

Do your review as usual. Once you accept the PR, the author is able to merge it.
If the PR needs a design discussion first, move it back to draft or close it, and continue the discussion on the issue.
If the PR requires significant changes, use "Request changes".
If you see patterns that shouldn't happen, update the
[pr-review skill](https://github.com/pytorch/pytorch/blob/main/.agents/skills/pr-review/SKILL.md) <!-- @lint-ignore --> so that the
automated review catches them in other PRs.

Do not add `in progress` or `ready for review` by hand.

## Closing Issues and Pull Requests

We have automation that will close PRs and issues with a comment giving a reason.
When you close an issue or a pull request yourself, for example when a PR needs a design discussion first,
always give the reason and, when it applies, link to
[Why was my issue or PR closed?](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#why-was-my-issue-or-pr-closed).
For example:

- "Closing this PR because it requires a design discussion that we should continue on the issue.
  Please comment on the linked issue with a summary of the status and approach to further discuss."
