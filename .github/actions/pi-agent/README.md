# pi Agent (Bedrock)

Runs one hermetic, non-interactive [pi](https://github.com/earendil-works/pi) session
against Amazon Bedrock: an alternative to `anthropics/claude-code-action` with a choice of
models (Claude and OpenAI on Bedrock) and our own TypeScript extensions. The directory is
self-contained; the same action lives in `pytorch/ciforge`.

| File | Role |
|---|---|
| `action.yml`, `run.sh` | Install pi from the lockfile, run the session, write the step summary |
| `package.json`, `package-lock.json` | pi version and integrity-pinned dependencies |
| `install-tools.sh` | SHA-256-pinned ripgrep and fd binaries; no sudo |
| `allowed-models.json` | Model IDs callers may use |
| `models.json` | Pinned definitions missing from pi's bundled model catalog |
| `claude-execution.jq` | Converts pi events to the Claude-compatible execution file |
| `extensions/tool-guard.ts` | Per-call tool allowlist; path tools confined to the working directory |
| `extensions/submit-result.ts` | Schema-validated `submit_result` tool for structured output |

## Usage

The search-tool installer currently targets Linux x86_64 runners (`ubuntu-24.04`).
It installs ripgrep 15.2.0 and fd 10.5.0 under `RUNNER_TEMP` and adds them to `PATH`,
so pi's `grep` and `find` work with `PI_OFFLINE=1`. Unsupported platforms fail at setup.

```yaml
permissions:
  contents: read
  id-token: write
steps:
  - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
    with:
      persist-credentials: false
  - uses: aws-actions/configure-aws-credentials@7474bc4690e29a8392af63c5b98e7449536d5c3a # v4
    with:
      role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_claude_code
      aws-region: us-east-1
      role-duration-seconds: 900
  - id: pi
    uses: ./.github/actions/pi-agent
    with:
      model: global.anthropic.claude-sonnet-5
      tools: read,grep,find,ls
      prompt: ${{ github.workspace }}/path/to/prompt.md # or literal text
      result-schema: ${{ github.workspace }}/path/to/schema.json # optional
      skills: ${{ github.workspace }}/.agents/skills/my-skill # optional
      artifact-name: my-job-pi
```

| claude-code-action | pi-agent |
|---|---|
| `--model` | `model` (from `allowed-models.json`) |
| `--allowedTools` | `tools`, enforced per call by `tool-guard` |
| `--json-schema` → `structured_output` | `result-schema` → `structured_output` |
| `--setting-sources ""` | always: nothing discoverable from the checkout loads |
| skills discovered from `.claude/skills` | `skills:` (explicit paths only) |
| execution file | `execution-file` output (same shape, for the review log inspector and `upload-claude-usage`), plus an artifact with `transcript.html`, `events.jsonl`, `result.json` |

The step summary shows the outcome, token and cost totals, which tools were offered, used,
and blocked, the result, and a link to the run artifact (open `transcript.html`).

With `result-schema`, pi validates the model's `submit_result` arguments against the schema
and returns errors for a retry; the model is reminded up to twice if it stops without
submitting. The step fails when there is no valid result, the model call errored (pi itself
exits 0 on provider errors), or a tool outside `tools` was offered or used. Treat
`structured_output` as untrusted: pass it through env vars, never `${{ }}` in a script.

`extensions:` and `skills:` load only the listed trusted paths. Skills live in
`.agents/skills` (`.claude/skills` links to it for Claude Code). Claude-only skill
frontmatter such as `hooks:` is ignored by pi, so enforce those rules outside the model.

## Lockdown

For a job that reads untrusted input, combine the action with:

- `step-security/harden-runner` with `egress-policy: block` and `disable-sudo: true` as the
  first step. pi needs `sts.us-east-1.amazonaws.com`, `bedrock-runtime.us-east-1.amazonaws.com`,
  and `registry.npmjs.org`, plus the GitHub hosts the job uses. The pinned search-tool
  downloads require `github.com` and `release-assets.githubusercontent.com`. `pi.dev` is used only with
  `allow-unlisted-model: "true"` when a model is missing from pi's catalog.
  `upload-artifact` works without extra entries.
- No write scope in the model's job; apply effects in a separate job without AWS access.
- No `bash` in `tools` (the default is `read,grep,find,ls`); `write` and `edit` are always blocked.

`scripts/issue_triage_pi/test_workflow_contract.py` pins these properties for issue triage.

## Models

Use an exact ID from `allowed-models.json`, including its inference-profile prefix and
version suffix: `global.` routes worldwide; `us.` stays in US regions. Bare IDs may fail
with "on-demand throughput isn't supported". Some models need a one-time AWS Marketplace
subscription by an account admin.

Before adding an ID, verify it with `pi-bedrock-smoke.yml` in `pytorch/ciforge`. That trusted
probing workflow uses `allow-unlisted-model: "true"`; normal callers keep the allowlist.
`models.json` supplies definitions missing from the pinned pi catalog for offline runs.

## Issue triage

`.github/workflows/issue-triage-pi.yml` is stage 2 of issue triage, after
`issue-triage.yml` captures the opened issue. It uses the `triaging-issues` skill with
`global.anthropic.claude-sonnet-5` at medium thinking, preserving the Claude Code triage model.
Sol remains available through the `model` dispatch input. Distributed triage uses the same default:

1. **plan** (`pi-triage-plan.yml`, shared with distributed triage; Bedrock, `issues: read`,
   egress blocked): the model reads the issue through
   read-only `gh` tools (`.github/pi/extensions/github-issue-tools.ts`: `get_issue`,
   `get_issue_comments`, `find_duplicate_issues`, fixed to the workflow repository) and submits a plan
   matching `.github/pi/schemas/issue-triage-plan.json`.
2. **apply** (`issues: write`, no AWS): `scripts/issue_triage_pi/apply_plan.py` filters the
   plan with the skill's label rules (forbidden → `triage review`; unknown and redundant
   dropped), only adds labels, posts fixed template or umbrella-reference comments the bot
   has not already posted, and adds `bot-triaged`. It closes usage questions, expected
   numerical behavior, or covered reports for catalog-enabled closing umbrellas. It then
   hands the issue to `distributed-triage-pi.yml`.

Duplicate discovery binds `TRIAGE_SOURCE_ISSUE` to the matrix issue in live and replay runs;
repository and source are not model inputs. It accepts 1-2 ordered groups of 1-4 literal ASCII
terms (up to 80 characters each, generated query at most 256): first focused, last broader
family. Put each group's subject anchor first and essential context last. A third supplied
group is rejected by the schema. The tool reserves a search slot to automatically relax the
last group, independent of nonempty results or a second model call. After trimming and removing
repeated terms in first-occurrence order, it keeps the first and last terms for at least three
distinct terms, or only the first for two. It adds no query for one distinct term or if the
relaxed query is identical to a provided query. Provided queries are otherwise unchanged.
This is a bounded query-relaxation heuristic, not guaranteed recall.

All terms are quoted in fixed `in:title,body,comments` searches: at most three searches of
10 rows each per call, followed by hydration of up to five distinct candidates ordered by
their best search rank across searches. Each query result records `origin: "provided"` or
`"broadened"` and its zero-based `source_group` input index; candidate `matched_queries`
indices refer to this query-result list, including a generated query when present.
Two calls per session bound this tool to six search and ten hydration attempts; failures count,
and each invocation has a 90-second deadline with 30-second subprocess timeouts. Other issue
readers are outside this budget. Results include open and closed evidence, 3,000-character body
prefixes, canonical links, query membership, clipping and partial-failure signals, and unhydrated
candidate IDs and titles. Search requests use GitHub's text-match media type to include matching
fragments without extra reads: at most three per hydrated candidate and one per unhydrated
candidate, each clipped to 400 characters, with fragment and count clipping flags. Fragments are
deduplicated across queries, with comment matches preferred when slots compete. Each fragment
identifies its issue/comment source, title/body field, and a constructed canonical issue or comment
link. Only known GitHub Issue title/body and IssueComment body metadata shapes are accepted;
missing text matches are valid. Numeric repository-ID match URLs rely on the already
repository-validated search row, and metadata URLs are never returned or followed.
Fragments are untrusted retrieval hints, not full comments or authenticated author/maintainer
statements; read full comments before relying on author authority, policy decisions, or quotations.
`status: "complete"` means bounded retrieval completed without coverage errors, not exhaustive
recall or full-body/comment evidence. Search rank is not confidence, and empty bounded results
do not prove absence.
The tool never decides duplicates or closes issues. Generic `search_issues` remains registered
but is not in the triage allowlist.

The offline duplicate-tool tests run against Pi installed from the action lockfile, use its real
argument validator, and replace only `gh` with controlled fixtures (no GitHub access):
`PI_TEST_RUNTIME=/absolute/path/to/node_modules/@earendil-works/pi-coding-agent node --test .github/pi/tests/github-issue-tools.mjs`.

Run it manually with `mode`: `replay` shows already-triaged issues as they were before the
triage bot acted (their labels and human comments from before its first label event) and
compares each plan with the labels added since. Source titles/bodies and duplicate-candidate
evidence are current, including candidates filed later; replay is not a frozen historical snapshot.
`dry-run` plans against the issues as they are now; `apply` writes and accepts exactly one issue per run. `replay` and `dry-run`
can process batches. Issue transfers become `triage review`. A `request_reproduction` plan
redacts download/storage links from the current issue body using a fixed policy and posts
the self-contained-reproduction template. Other issue text is preserved; the model cannot
supply replacement prose.

## Distributed triage

`.github/workflows/distributed-triage-pi.yml` runs the `distributed-triage` skill on issues
in the `oncall: distributed` queue: after `issue-triage-pi.yml` hands one off, and when the
daily sweep (`claude-distributed-triage-cron.yml`) dispatches it. It uses the same
`pi-triage-plan.yml` model job; `scripts/issue_triage_pi/apply_distributed_plan.py` applies
the plan and adds the `ptd-bot-triaged` marker that the sweep checks. A model-free `record`
job appends the result to the daily manifest. Dispatch with `mode: dry-run` to see the
effects without writing.
