# Auto PR Triage

Auto PR Triage processes every pull request once it is ready for review. Intake
decides whether a pull request takes no action, is marked as missing an
actionable issue, or is routed to reviewers; ownership analysis combines deterministic codepath rules with
LLM-suggested semantic owners; planning turns both into one plan, which live
mode applies. The code and its docstrings describe how each stage works. This
file keeps only what the code cannot show: goals, accepted tradeoffs, the
reasons behind some rules, and operating notes.

```mermaid
%%{init: {"htmlLabels": false, "flowchart": {"htmlLabels": false}}}%%
flowchart TD
    ready["Ready PR targeting main<br/>no prior Auto PR Triage outcome"] --> own("Ownership analysis<br/>CODEOWNERS, semantic owners, bypass intake")
    own --> intake(Intake)
    intake -- "fails policy" --> missing["Missing actionable issue<br/>“triaged”, “bot-triaged”, “missing actionable issue”"]
    intake -- "fails policy, on a workflow rerun<br/>or with a too uncertain bypass claim" --> open["Kept open<br/>no labels"]
    intake -- "fails policy, LLM failed" --> incomplete["Incomplete<br/>“bot-triage-error”"]
    intake -- "passes policy" --> plan(Planning)
    plan -- "owner other than the author,<br/>an owner or handoff reviewer assigned" --> triaged["Triaged<br/>“triaged”, “bot-triaged”, reviewers requested or already assigned"]
    plan -- "owner other than the author,<br/>no owner or handoff reviewer assigned" --> open
    plan -- "no owner other than the author,<br/>handoff reviewer engaged" --> triaged
    plan -- "no owner other than the author,<br/>no handoff reviewer" --> open
    plan -- "LLM failed, a team owner has no reviewer, or<br/>reviewer state is unreadable with an owner" --> incomplete
    open --> manual(Manual triage) --> triaged
    incomplete --> manual
    classDef state fill:#f3f4f6,stroke:#6b7280,color:#000
    classDef aiproc fill:#dbeafe,stroke:#2563eb,color:#000
    classDef humanproc fill:#fef3c7,stroke:#d97706,color:#000
    class ready,missing,open,triaged,incomplete state
    class own,intake,plan aiproc
    class manual humanproc
```

Rectangles are states. Rounded boxes are processes: blue for automated ones
and orange for human ones. When several edges apply, Incomplete wins over the
other outcomes, and Kept open wins over Missing actionable issue.

- **Runs** when a PR is opened, reopened or marked ready for review, or when
  `bot-retriage` is added.
- **Passes policy**: it closes a same-repository issue marked `actionable`, the
  author has triage+ permission, a triage+ maintainer took deliberate action on
  it, it says "Supported by @login" for a triage+ maintainer, it says "Part of
  #N" for an `actionable` issue, or a team's bypass intake criteria matched.
- **Owners** are CODEOWNERS matches plus LLM-suggested semantic owners. A team
  owner ID is resolved through `.github/auto-pr-triage/team_members.json`. A
  roster member already covering the PR (codepath owner, submitted or pending
  reviewer) satisfies it; otherwise a member is picked at random, seeded by
  the PR.
- **Handoff reviewer**: see "An assigned owner other than the author is
  enough to triage" below.

## Goals and non-goals

The design aims to:

- apply a deterministic intake policy;
- preserve codepath ownership as the review baseline while adding semantic
  ownership;
- select configured reviewers while reusing review coverage that already
  exists;
- produce an exact, inspectable plan in the read-only job, whether or not live
  mode applies it; and
- run the same code in different repositories using their checked-in
  configuration.

It does not aim to:

- judge whether a pull request is correct, useful, or ready to merge;
- eliminate human triage when no owner is found or nobody can be assigned;
- provide a transactional view of GitHub state or automatically reconcile every
  change that occurs after analysis;
- guarantee perfect semantic assignments or a perfectly even reviewer load; or
- create required labels or ownership configuration automatically.

## Design principles

- **Separate intake from ownership.** Repository policy determines whether a
  pull request proceeds to routing; ownership analysis determines who should
  review it.
- **Keep ownership additive.** Semantic ownership supplements rather than
  replaces codepath ownership.
- **Plan once, then apply.** Analyze reads pull-request, ownership, and reviewer
  state once and plans the exact effects; apply only executes a validated plan.
- **Leave uncertainty for people.** Missing information cannot mark a PR as
  missing an actionable issue, and incomplete routing is reported for human
  follow-up.
- **Keep writes out of shadow.** Shadow and live compute the same plan; shadow
  never runs a job that can write to GitHub.

## Accepted stale-result risk

Auto PR Triage intentionally does not close races by re-reading mutable GitHub
state. Intake, ownership analysis, and planning each read once, and apply
executes the plan without further reads, so separate calls can observe
different moments and later changes are ignored for that run.

This is a deliberate complexity and availability tradeoff, not a missing
security check. A run can make a stale but bounded mutation: request a
configured reviewer, or add one of the bot's own labels, such as marking a PR
as missing an actionable issue after it gained one. These effects are accepted
because maintainers can remove review requests and labels or rerun the
workflow, and the labels and artifacts make every action visible. Security
review should judge whether the mutation set stays bounded rather than expect
transactional consistency across jobs. Do not "fix" the staleness by adding
uncoordinated race-closing reads to apply.

## Security model

- `pull_request_target` runs only workflow and action code from the default
  branch, whatever the PR's base, and the PR head is fetched as data, never
  checked out or run.
- The LLM has no GitHub token and no file, shell, web, MCP, plugin, or subagent
  capability; its Bedrock session can only invoke the model.
- The write-capable job runs only in live mode and receives only the bounded
  `ActionPlan`. Rationale and diff evidence stay in the analyze job as log
  provenance and never choose reviewers, labels, or effects.
- User-controlled text never becomes a shell argument, label, or comment.
  Issue numbers and logins parsed from the PR body reach read-only API paths
  and the supporter request only after pattern validation.
- New reviewer requests come from trusted configuration, a verified supporter
  claim, or the actor who applied `actionable` to an admitting issue; the last
  two must have current triage-or-higher access.
- LLM output can keep a PR from being marked as missing an actionable issue,
  through a bypass match, but can never cause the mark. A prompt injection in the PR text can at worst keep a PR open and route it to
  a team that configured `bypass_intake_criteria`, bounded to that team's
  roster. Bypass claims need the same verified diff evidence as any owner, and
  only teams that opt in are exposed.
- Structural checks cannot prove the LLM's judgment correct: an LLM error or
  injection can omit a legitimate additional owner or suggest an unnecessary
  configured one. Live effects stay bounded to configured reviewers and the
  bot's labels; no plan can close a PR or post a comment.

## Why some rules are the way they are

- **Linked issue timelines do not count as maintainer activity.** Maintainers
  label and comment on most issues during triage, so that activity would admit
  any PR that merely mentions a triaged issue. Only the PR's own timeline is
  read.
- **Deployments are passive timeline events.** GitHub attributes a deployment
  to whoever triggered the workflow run, including this workflow's own
  environment-gated job, so it says nothing about deliberate maintainer action.
- **An assigned owner other than the author is enough to triage.** Once a
  codepath or semantic owner other than the author is found and an owner or
  handoff reviewer other than the author is assigned (requested by the run,
  or already pending or reviewing), the PR is triaged even if some concerns
  have no owner; the assigned reviewers pull in others for them, and the
  run's step summary lists them. This accepts that a path match can come from
  an incidental edit, leaving the main concern to a reviewer who has to
  redirect it. Owners alone are not enough: if GitHub never sent the
  CODEOWNERS request and no other owner or handoff reviewer is assigned, the
  PR is kept open for a human, since a triaged PR is not looked at again.
  Unrelated review activity, such as a bot's review, does not count. When no
  owner other than the author is found, the PR is triaged only if a handoff
  reviewer other than the author is engaged: a verified supporter, a
  maintainer who labeled an `actionable` issue the PR closes or is "Part of",
  or a triage+ reviewer whom another triage+ maintainer, not the author,
  requested.
- **Every pull request is triaged; no label authorizes a run.** Opening,
  reopening, or marking a pull request ready starts a run, so anyone who opens
  a pull request can start one. Each run makes at most one LLM call and its
  effects stay bounded, and a pull request with an outcome label is a no-op
  that skips the LLM. Removing a label does not cancel a run that has started.

## Codepath owners come from CODEOWNERS

GitHub requests codepath owners itself, through the repository's `CODEOWNERS`
file; Auto PR Triage never requests them. It parses the same file to tell the
LLM which changed files are already owned and, in planning, to skip a roster
pick for a team whose member is a codepath owner and to explain each codepath
owner. GitHub can legitimately fail to send a request: the file can be invalid
or over GitHub's size limit, an owner can no longer exist or lack repository
access, or GitHub's processing can be delayed. So the explanation reports what
the reviewer state showed for each codepath owner (a pending request, a review,
neither, or unreadable state), and "neither" flags a request that GitHub may
not have sent or that someone removed. For the same reason planning triages a
PR only once an owner or handoff reviewer is assigned. Unreadable reviewer
state never triages a PR with owners: the plan is incomplete, or fails outright
when there are supporters or actionable labelers to filter. Like GitHub, the
parser skips `CODEOWNERS` lines it cannot read, such as an email owner, and the
ownership input step logs each skipped line; a skipped line's owners get no
provenance.

## Operating notes

- **Prerequisites.** The default branch needs the workflow and composite action,
  root `CODEOWNERS`, the two files in `.github/auto-pr-triage/`, and the
  `bot-retriage`, `actionable`, `triaged`, `bot-triaged`, `bot-triage-error`,
  and `missing actionable issue` labels. The workflow does not create labels.
- **Shadow mode leaves no outcome label,** so a later eligible event analyzes
  the PR again. Labels such as `bot-shadow-close` and `bot-shadow-triaged` from
  earlier shadow runs are not handled state.
- **Rerunning.** Pushes and comments never trigger another run. Reopening a
  pull request or marking it ready again does, but while an outcome label is
  present, the run is a no-op. To reevaluate a PR, comment `@pytorchbot
  triage`, or remove its outcome labels (`triaged` and `bot-triaged`, with
  `missing actionable issue` if present, or `bot-triage-error`), then add (or
  remove and re-add) `bot-retriage`. Every `bot-retriage` label event starts a
  new run. Rerunning all jobs of an
  earlier run also works, but rerunning only the apply job is rejected because
  its plan may be stale. A rerun never marks a PR as missing an actionable
  issue: only the first attempt of a run can, and a rerun leaves an unadmitted
  PR for a human.
- **PRs left by the removed `routed_untriaged` decision** have no outcome
  label and nothing reruns them automatically. Add `bot-retriage` to apply
  the current rules.
- **Opting out.** A PR labeled `no automated triage` is left as is: a run that
  starts while the label is present is a no-op and the PR stays in the manual
  triage queue. Adding the label mid-run does not cancel that run. To triage
  the PR later, remove the label and follow the rerunning steps above.
- **Failures in shadow mode** are visible only as a failed workflow run and its
  artifacts. In live mode the error-reporting job adds `bot-triage-error` on a
  best-effort basis.
