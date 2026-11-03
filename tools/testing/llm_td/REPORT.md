# LLM test selection for PyTorch CI: offline evaluation

**TL;DR.** If PRs picked their tests with one Opus 5.5 call (medium effort, plus a 20% TD hedge) instead of today's target determination (TD), CI test-job spend would fall **25%, from $935k to $702k a month** ($233k saved, $43k of it GPU), and the p50 wait for pull and trunk would fall from **181 to 119 minutes**. It catches more of the breakages that landed than TD does, and is level with TD, within noise, on failing PR pushes:

- **Reverted commits caught: 83.1% vs 70.3% for TD** (n=148). Every one of these commits landed and was reverted for breaking a test; "caught" means its PR's CI would have gone red. The paired difference is +12.8 points, with a 95% bootstrap interval of +4.7 to +20.9 (29 caught only by new, 10 only by old; McNemar p=0.003).
- **Failing PR pushes with every real failure selected: 84.8% vs 84.3% for TD** (n=782 pushes, 285 PRs). Flaky failures are excluded with Dr.CI's signals. The +0.5 point difference is noise: the 95% interval is -5.8 to +6.6 points when PRs are resampled (pushes of one PR are correlated) and -2.8 to +3.8 per push (86 vs 82 discordant pushes; McNemar p=0.82).
- **The dollar and wait figures include fork PRs, which never get a selection.** Fork PRs are 30-46% of each workflow's PR test hours, so the $233k, the $190k and the wait drops are overstated by about that share. `dollars.py` now leaves fork PRs out; the figures in this report have not been rescored. See Spend.

| | Old: TD | New: Opus 5.5 medium + 20% hedge |
|---|---|---|
| Reverted commits caught (n=148) | 70.3% | **83.1%** |
| Failing PR pushes with every real failure selected (n=782) | 84.3% | **84.8%** |
| CPU / GPU test time run on a typical change | 48.7% / 43.5% | **24.0% / 22.4%** |
| Test-job spend | $935k/mo | **$702k/mo** |
| pull + trunk wait, p50 / p90 | 181 / 291 min | **119 / 206 min** |

- **The idea.** One `claude -p` call per PR push sees the title, description, diff and every candidate test file annotated with its CI minutes, and returns the files worth running.
- **Old and new.** "Old" is TD as it runs on PRs today: its top 25% of test files per config, in the workflows that have TD. "New" is the proposed production config: Opus 5.5 at medium effort, plus TD's top files up to 20% of each config's test time. Main always runs everything in both.
- **The 25% and the 20% are in different units.** TD's 25% counts files: run_test.py keeps the top quarter of TD's ranked files (`get_top_per_tests(25)`) whatever their size. TD ranks the large, often-failing files high, so that quarter of the files is 48.7% of the CPU test time on a typical change. The hedge's 20% counts seconds: TD's ranked files are added in order until they fill 20% of the config's test time, about the first two fifths of what TD runs today. The model does not filter TD's list. New runs the union of the model's picks and that hedge:

| CPU test time run on a typical change | Share of a full pass |
|---|---|
| Old: TD's top 25% of files | 48.7% |
| New, model's picks alone (including the run-all rules) | 7.6% |
| New, model's picks plus the 20% hedge | 24.0% |

The hedge adds 16.4 points rather than 20 because files the model already picked cost it nothing, and a file that does not fit the remaining budget is skipped.
- **The model calls** cost about $2.2k a month. This stack makes $190k of the saving reachable, in pull and trunk, once enforcement is the default (Rollout step 3); until then it applies only to PRs with the `ci-llm-td` label. The other $43k is in workflows where TD is off today (inductor's unit tests, periodic, slow), so they run every test on PRs and the selection is not enforced there; inductor also needs a target-determination job first. See Follow-ups under Spend.
- **The stack.** The selector, harness, run_test.py integration and a shadow-mode CI workflow are in the PRs below this one. Enforcement is opt-in per PR through the `ci-llm-td` label, on same-repo PRs only.

The sections up to Time to land compare old and new only. Other models, effort levels and hedge budgets are in [Sweep](#sweep-other-models-efforts-and-hedges), which is how the config was chosen.

## Proposed production config

| | Old | New |
|---|---|---|
| Selector | TD heuristics, top 25% per config | Opus 5.5, effort medium, no tools, plus TD's top files up to 20% of the config's test time |
| Safety rules | none | edited or added test files, and files that already failed on the PR and that TD runs, always run; changes to test infrastructure run everything; a missing or malformed answer is a selector failure, which keeps TD's split |
| Runs on | PRs in TD-enabled workflows | same-repo PRs in TD-enabled workflows; fork PRs keep TD |
| Main pushes | everything | everything |

## What the backtest measures

We took historical changes, asked each policy offline which test files it would run, and checked that against which tests really failed. A failure is a (test file, CI config) pair, for example `test_ops.py` in the `default` config. Only failures in jobs that run TD today count: the pull, trunk and ROCm workflows, minus the jobs where run_test.py turns TD off (macOS jobs, xpu and onnx builds, and jobs with `force-no-td`). Jobs elsewhere (inductor's unit tests, periodic, slow, ...) and those excluded jobs run every test under both policies, so their failures cannot tell the two apart and are left out. The two headline metrics are:

- **Reverted commits caught (n=148).** These are commits that landed on main between 2026-04-01 and 2026-10-25 and that the autorevert bot then reverted because they broke tests. Main runs every test, so we see which files started failing, and in which configs, between the landing and the revert. The truth is limited to the test files the autorevert bot cites as its signal (`source_signal_keys`); other files that failed in that window are not counted. Of 160 such reverts, 148 broke a TD job, with 193 failures in those jobs; the other 12 broke only jobs without TD. A revert is caught if the policy runs at least one of its failures, which is enough for the PR's CI to go red and block the land. This truth is not biased toward either policy, since TD is not involved in choosing what main runs.
- **Failing PR pushes with every real failure selected (n=782).** We took 316 PRs from 2026-10-01 to 2026-10-25 whose CI had a test failure, and every push (head commit) of those PRs, not just the last one. A push counts if one of its TD jobs had a test failure that is not flaky by the signals below; 782 pushes of 285 PRs do, with 1,897 failures. A push counts as caught only if the policy runs **every** one of its real failures, so the author sees the whole breakage in one round instead of finding the rest on the next push. These PRs ran under TD, so we only see failures in files TD chose to run, which favors old.

**Flaky failures** are excluded from the push truth with the signals Dr.CI uses to mark a PR failure flaky or not the PR's fault, applied with hindsight (a backtest can look after the failure, which Dr.CI cannot). A failure is flaky if any of these holds:

- the same job passed on a rerun of the same commit;
- the job is marked `unstable`;
- the test was disabled (a `DISABLED` issue) within 14 days;
- the same test failed on main, or on another author's PR, within 14 days.

Of 20,094 failing tests in TD jobs on these PRs' pushes, 25.6% are flaky by these signals. The signals overlap: 24.9% failed on another author's PR, 19.4% on main, 13.3% were disabled, 0.9% passed on a rerun and 0.1% were in unstable jobs. The 74.4% that remain are what the push metric scores. Reverts need no such filter: the autorevert bot already requires the failure to start at the reverted commit and persist.

**Test time run** is the mean, over 188 random main commits from 2026-10-01 to 2026-10-25 (two that were later reverted are excluded, see below), of the share of a full CI pass's test-seconds a policy runs, with CPU and GPU counted separately, over the configs that some TD job runs (the inductor unit-test configs are priced separately below). The dollars come from this number.

## Accuracy and test time

| | Old | New |
|---|---|---|
| **Reverted commits caught (n=148)** | 70.3% | **83.1%** |
| reverted commits with every failure selected | 65.5% | 80.4% |
| their failures caught (193) | 63.7% | 79.3% |
| **Failing PR pushes with every real failure selected (n=782)** | 84.3% | **84.8%** |
| failing PR pushes with at least one real failure selected | 90.7% | 93.7% |
| their failures caught (1,897) | 90.6% | 91.6% |
| CPU test time run on a typical change | 48.7% | 24.0% |
| GPU test time run on a typical change | 43.5% | 22.4% |
| Model call, $ per push / p50 latency | - | $0.191 / 14s |

- On reverts, new catches 29 that old misses, and old catches 10 that new misses (McNemar p=0.003; 95% interval of the difference +4.7 to +20.9 points).
- On failing pushes the two are level within noise: new selects every real failure on 86 pushes where old does not, and old on 82 where new does not (95% interval -5.8 to +6.6 points, resampling PRs).
- The intervals are a paired bootstrap (10,000 resamples) over reverts and over PRs; see Reproducing.
- Old's numbers come from a replay of today's TD that is optimistic for it; see Caveats.

## Spend and savings

30-day runner cost of test jobs, from `misc.runner_cost`. Builds, lint and other non-test jobs are not counted. Only the PR-triggered part of each workflow changes; main pushes and scheduled runs keep running everything.

**Fork PRs are counted in this table, but never get a selection.** `llm-td.yml` and the selection step in the `target-determination` job (`resolve_selection.py`) skip fork PRs, so their test jobs keep TD. Fork PRs are 36% of pull's PR test hours, 46% of trunk's, 30% of inductor's, 37% of periodic's and 38% of slow's. The New column, the savings below and the Time to land drops assume every PR is selected, so they are overstated by about that share. `dollars.py` now multiplies each workflow's PR spend by one minus its fork share (and counts model calls on same-repo PR heads only, 12.5k in the 30 days to 2026-11-03); the figures here predate that and have not been rescored.

| Workflow | Old $k/mo | New $k/mo |
|---|---|---|
| pull, CPU | 463.6 | 337.5 |
| trunk, CPU | 157.0 | 123.8 |
| trunk, GPU | 158.0 | 127.3 |
| inductor, CPU (follow-up) | 79.0 | 47.8 |
| inductor, GPU (follow-up) | 54.5 | 45.8 |
| periodic, GPU (follow-up) | 17.0 | 13.7 |
| slow, CPU (follow-up) | 6.3 | 6.0 |
| **total** | **935** | **702** |
| model calls | 0 | 2.2 |

- **Saved:** $233k a month, $191k CPU and $43k GPU (dollar figures are rounded separately, so sums can be off by $1k). The first three rows, which this stack covers, are $190k, reachable once enforcement is the default (Rollout step 3).
- **Follow-ups.** run_test.py enables TD only when `GITHUB_WORKFLOW` is pull, trunk or ROCm (`--enable-td`'s default), and the selection is only enforced where TD is on. So jobs of the inductor unit-test workflow (run on PRs by `ciflow/inductor`), periodic and slow run every test, under old and new alike. The rows above assume each follow-up is done and the shadow data shows the selection is safe there: inductor $40k, periodic and slow $4k.
  - **Periodic and slow** already have a `target-determination` job, so their test jobs get TD's ranking and the selection. Each needs adding to the `--enable-td` check in run_test.py and to `TD_ENABLED` in `target_determination.yml`, which turns TD's cut and the hedge on with it.
  - **Inductor** has no `target-determination` job (neither `inductor.yml` nor `inductor-unittest.yml`), so there is no `td_results.json`: run_test.py turns TD off without it, enforcement requires TD on, and the 20% hedge these rows assume needs TD's ranking. The download step fails on the missing TD artifact before it reaches the selection. Inductor first needs a `target-determination` job, then the same two changes.
- **Why 25% and not 50%.** New runs about half the PR test time old does, but three things dilute that. Main and scheduled runs keep running everything. The inductor model benchmarks, which PRs trigger, are not test files. And the 20% fixed job overhead is not saved (see Dollar model).
- In inductor only the run_test.py unit-test configs are selectable. The model benchmarks (huggingface, timm, torchbench) are most of its GPU spend.
- **Not in the table:** rocm-mi300 and the other ROCm workflows also download the selection, but their runners are carried at $0 in `misc.runner_cost`, so they show up only under Time to land and Capacity. The remaining callers (win, mac, s390x, tpu) cost under $1k a month together or are also carried at $0.

## Time to land

| Wait for a PR push | Old (min) | New (min) |
|---|---|---|
| pull and trunk, p50 | 181 | 119 |
| pull and trunk, p90 | 291 | 206 |
| pull and trunk, mean | 205 | 137 |
| pull, trunk and inductor, p50 | 191 | 133 |
| pull, trunk and inductor, p90 | 442 | 399 |
| `@pytorchbot merge` that starts trunk, comment to merge, p50 | 185 | ~120 |
| `@pytorchbot merge` that starts trunk, comment to merge, p90 | 269 | ~206 |

The two "pull, trunk and inductor" rows assume the inductor follow-up (a `target-determination` job for inductor, see Spend); without it inductor's part of the wait does not change. Every row assumes every PR is selected, fork PRs included, which they are not (see Spend).

Time to signal (TTS) per workflow, in minutes. The inductor row assumes the inductor follow-up:

| Workflow | Old p50 | New p50 | Old p90 | New p90 |
|---|---|---|---|---|
| pull | 70 | 46 | 95 | 78 |
| trunk | 182 | 120 | 277 | 206 |
| inductor (follow-up) | 136 | 111 | 223 | 185 |
| rocm-mi300 | 208 | 127 | 313 | 223 |

- **Merge wait.** 1,643 PR heads ran pull and `ciflow/trunk` together. On those, the p50 drops by about an hour and the p90 by 1.4 hours. Trunk is the later signal for nearly all of them in both. With `ciflow/inductor` as well (1,186 heads), the inductor benchmarks bound the wait.
- **Trunk is where the latency is.** Its long test shards are the critical path, so the test shrinkage shows up as wall-clock time.
- **Pull leans toward build-bound.** Runs that wait on a build go from 12% to 34%. The floor is the xpu build (p50 40 / p90 49 min from creation to done) and the ROCm build (p50 23 / p90 53).
- **The model matches real merges.** When the merge comment started trunk (n=337), comment to merge took p50 185 / p90 269 min, against a simulated trunk TTS of 182 / 277. When trunk had already run before the comment (n=533), the merge took p50 3 min.
- **CI is a minority of end-to-end time to land.** From the first pull run on the merged head to the merge is p50 15.8 h and p90 94.5 h, dominated by review. The saving matters most for authors iterating on red CI, where every push pays it again.

**Capacity.**
- PR test job-hours across pull, trunk, inductor and rocm-mi300 go from 675k to 393k a month. That includes inductor's unit tests, which need the inductor follow-up, and fork PRs, which are not selected. `Test` is 70-95% of test-job time on every major runner type.
- Median runner queues are already 1-3 minutes. The tails are long: a10g p90 14 min, l4 p90 11, macos-m1 p90 59, mi300 p90 56, and the 4-GPU a10g pool p50 11 / p90 135. Freed capacity should cut those tails; this is a qualitative argument, not a model.

**Method.**
- For every PR workflow run in the 30 days to 2026-11-03, each test job's `Test` step is shrunk by the ratio r of new to old test time for that job. Each job keeps its start time. Builds and non-test jobs are unchanged, shards are not rebalanced, and queueing effects are ignored.
- In pull, trunk and rocm-mi300, r is 0.493 (CPU) and 0.515 (GPU), from the typical-change test-time shares. In inductor, which has no TD, r is 0.238 / 0.327 for the unit-test configs (assuming the inductor follow-up) and 1 for the model benchmarks.
- Every PR run is shrunk, fork PRs included, although fork PRs keep TD.
- TTS is the last job's completion minus the run's creation. Only first attempts are used. Cancelled runs, runs with a cancelled job, and runs that failed before any build are excluded.
- r is one ratio per workflow and runner kind, applied to every shard. An unbalanced selection, where one shard keeps most of the selected tests, would make the real latency win smaller.

## Dollar model

- The selectable part of each test job is the Test step. Outside it, checkout, docker pull, build download and teardown take a median of 4.3 to 4.8 minutes, about 12-15% of the job. The model assumes a fixed overhead fraction of 0.2 for that.
- A job's relative cost is `0.2 + 0.8 * f`, where f is the share of test time it runs. For old, f is the replayed TD share in TD workflows and 1 elsewhere. The PR share of each workflow's test hours is then rescaled by new over old.
- **Test-time shares.** pull and trunk use the typical-change shares from the Accuracy table (old 48.7% CPU / 43.5% GPU, new 24.0% / 22.4%). periodic and slow have TD off today, so old runs everything there (relative cost 1) and new uses the same new shares; those are over the configs TD jobs run, so the periodic and slow rows are approximations ($4k together). The inductor rows use the configs of inductor-unittest.yml that no TD job runs (`inductor_core`, `inductor_cpp_wrapper`, `inductor_distributed`, `inductor_amx`, `inductor_aoti_fallback`, `inductor-halide`, `inductor-triton-cpu`), over only the 69 of 190 typical commits that match the `ciflow/inductor` autolabel globs, since those are the PRs that run it; on those, new runs 23.8% of CPU and 32.7% of GPU unit-test time. These three figures come from an earlier glob parser that let `*` cross directories, whereas `dollars.py` matches `labeler.yml` with the labeler's own semantics. trunk's own `inductor` config has TD and is priced with trunk.
- **Model cost.** $0.191 per call is the mean over the 664 revert, first-push PR and typical samples, a conservative basis since typical commits average $0.146. At about 11.4k PR pushes a month that is $2.2k a month. Fork PRs make no call; `dollars.py` now uses the same-repo count instead (see Spend). Timed-out calls record no cost and are left out of the mean.

The overhead fraction dominates the uncertainty:

| Overhead | CPU saved | GPU saved | Total saved |
|---|---|---|---|
| 0.1 | $232k | $52k | $284k/mo |
| **0.2** | **$191k** | **$43k** | **$233k/mo** |
| 0.35 | $140k | $31k | $171k/mo |
| 0.5 | $99k | $22k | $121k/mo |

**Where the GPU money is.**
- The GPU number is modest because the biggest GPU line, the inductor model benchmarks, is not selectable test files.
- The same approach can gate them: ask the model whether a change can affect compiled-model performance or accuracy at all, which is a yes/no question rather than a file list. That is step 5 of the rollout.
- Skipping shards whose selection is empty would also recover the job overhead the model assumes is fixed.

## Method

**Unit of selection.**
- The unit is a test file under `test/`. Selecting a file runs it in every config it belongs to: default, distributed, slow, inductor and so on.
- The two wrapper sweeps, `dynamo_wrapped` and `crossref`, re-run hundreds of ordinary files in a special mode. The model requests them separately; an unrequested sweep runs only the selected files.
- The model can also request whole directories (`dynamo`, `inductor`, ...) or `run_all`.

**Safety rules.** `Selection.from_model_output` and `select_tests.py` apply these on top of the model's answer:
- A test file the PR edits or adds is always selected.
- The answer becomes run-all when the change touches anything that decides how tests run. That covers run_test.py, conftest, common_utils, common_device_type, tools/testing, the test shell scripts and actions, and the CI docker images. This run-all is enforced.
- A missing or malformed answer, or any other selector error, is a selector failure: the selection is marked not enforced and CI keeps TD's own split. The harness's `final+tdN` policies score it the same way, as TD's top 25%; `model`, `final` and `final-hm` score it as run-all.
- In CI, run_test.py also always runs the files that already failed on the PR and that TD's split runs, charged first to the hedge budget. The other previous failures compete for the hedge like any TD-scored file.

**Hedge.** The hedge adds TD's highest-ranked files with a positive score until they fill 20% of the config's test time, each file charged once. A file that does not fit is skipped and filling continues. The harness applies the same fill rule, per config summed over build environments rather than per job: CI fills 20% of one job's own test list (all its shards together) from that job's test times. The harness also leaves out TD's PR-history heuristic (see Caveats).

**Cost.**
- Cost is the share of one full CI pass's test-seconds that a selection runs, from `test-times.json`, per config. Configs that no TD job runs are left out; the inductor unit-test ones are priced separately.
- Jobs are classified from `.github/workflows`: by name and config, then with CUDA and Python versions wildcarded (CI has moved to newer ones since the oldest reverts), then by config alone. A reusable workflow's jobs take their caller's TD setting, so `dynamo-unittest` jobs (called from trunk) have TD and `unit-test / ...` jobs (called from inductor) do not. Within a TD workflow, jobs run_test.py runs without TD are classified as non-TD, as in CI: `_mac-test.yml` jobs, xpu and onnx build environments, and jobs with `force-no-td`.
- `test-times.json` is merged over jobs per config, so a config that TD and non-TD jobs share, like `default` in periodic, is priced as one.
- Seconds are summed over every build environment. Environments matching `cuda|rocm|sm\d+|xpu|h100|b200|a100|gpu` count as GPU.

**Dataset** (`eval/build_dataset.py`, ClickHouse plus git). There are four kinds of sample:

| Kind | n | Window | Truth |
|---|---|---|---|
| revert | 160 | 2026-04-01 to 10-25 | Main commits the autorevert system reverted. The truth is the failures of the autorevert signal's test files (`source_signal_keys`) in main jobs between landing and the revert, with their configs. |
| head | 783 | pushes of the `pr` samples' PRs | Every push of a `pr` sample's PR with a non-flaky failure in a TD job; the truth is those failures. The diff is against the push's merge base with main. 782 have a failure among the scored candidates. |
| pr | 316 | 2026-10-01 to 10-25 | The PR's earliest failing push, with failures no other author's PR shared and main did not have within 4 days. It defines which PRs `head` covers, and backs the first-push rows in Sweep. |
| neutral | 188 | 2026-10-01 to 10-25 | Random main commits ("typical changes"), with no truth. They measure how much a selector runs on a typical change, which is what the dollars depend on. |

Reverts are the least biased sample. On PRs in TD-enabled workflows, we only ever observe failures in the files TD already chose to run, so PR truth is censored toward TD's picks. Main runs everything, so revert truth is not censored this way.

**Prompt development.**
- The prompt (`prompt.md`) was iterated on a disjoint 436-sample dev set (`eval/ids/dev.txt`).
- The 10 neutral pilot samples that were looked at while writing it were removed from the reported set, and so were two neutral commits that were later reverted (build_dataset now excludes reverted commits from neutral draws), leaving 664: 160 reverts, 316 first-push PRs and 188 typical changes (`eval/ids/eval.txt`; its neutral part is `eval/ids/neutral.txt`, and the pushes are `eval/ids/head.txt`).
- Every model ran the same prompt and the same samples, with no tools. Median input was about 30k tokens; the candidate list is about 1,100 files.

**TD baseline.**
- `eval/td_replay.py` runs today's in-repo TD heuristics on each sample's changed files and commit text, then takes the top 25% cut CI uses, per config.
- The heuristics read today's historical-stats artifacts, which already contain these failures. The replay is therefore optimistic for TD.

## Sweep: other models, efforts and hedges

This section is how the production config was chosen. Rows use the same samples and definitions as [What the backtest measures](#what-the-backtest-measures).

![reverts caught and failing pushes fully selected vs CPU test time](report_tradeoff.svg)

Each line is one model and effort level, through hedge budgets of 0, 5, 10, 15 and 20% (labeled on the Opus medium line). The red X is today's TD, and the dotted red line marks its level. The gold star is the proposed config. The dashed gray line is the Pareto frontier: no point beats a frontier point on both axes. The x axis is the CPU test time run on a typical change, which is what the dollars scale with.

**Opus 5.5 at each effort and hedge:**

| Effort / hedge | Reverts caught | Reverts, every failure | Pushes, every failure | Pushes, any failure | Typical CPU | Typical GPU |
|---|---|---|---|---|---|---|
| low, 0% | 70.9% | 68.2% | 74.7% | 84.3% | 7.6% | 7.3% |
| low, 5% | 77.7% | 73.6% | 78.1% | 90.0% | 11.5% | 11.5% |
| low, 10% | 79.1% | 73.6% | 80.7% | 91.9% | 15.6% | 15.4% |
| low, 15% | 81.8% | 75.7% | 82.9% | 92.7% | 19.6% | 18.9% |
| low, 20% | 81.8% | 77.0% | 83.9% | 93.6% | 23.9% | 22.2% |
| medium, 0% | 70.9% | 68.2% | 76.0% | 85.2% | 7.6% | 7.5% |
| medium, 5% | 79.1% | 75.7% | 79.7% | 90.8% | 11.6% | 11.7% |
| medium, 10% | 80.4% | 77.0% | 81.7% | 92.2% | 15.7% | 15.6% |
| medium, 15% | 83.1% | 79.1% | 83.9% | 93.4% | 19.7% | 19.1% |
| **medium, 20% (production)** | **83.1%** | **80.4%** | **84.8%** | **93.7%** | **24.0%** | **22.4%** |
| high, 0% | 73.0% | 70.3% | 76.9% | 85.4% | 7.8% | 7.7% |
| high, 5% | 77.7% | 75.0% | 80.2% | 90.9% | 11.7% | 11.9% |
| high, 10% | 79.1% | 75.0% | 82.1% | 92.1% | 15.7% | 15.7% |
| high, 15% | 83.1% | 77.7% | 84.0% | 93.6% | 19.8% | 19.2% |
| high, 20% | 82.4% | 77.7% | 85.0% | 93.9% | 24.0% | 22.5% |
| TD today (old), for reference | 70.3% | 65.5% | 84.3% | 90.7% | 48.7% | 43.5% |

Saved per month at the 0.2 overhead, for Opus medium: $368k with no hedge, $334k at 5%, $301k at 10%, $268k at 15% and $233k at 20%.

**Other models, 20% hedge.** $/call is the mean over the 664 revert, first-push PR and typical samples; p50 wall is the model call alone.

| Model / effort | Reverts caught | Pushes, every failure | Typical CPU | $/call | p50 wall |
|---|---|---|---|---|---|
| **Opus 5.5 medium (production)** | **83.1%** | **84.8%** | **24.0%** | **$0.191** | **14s** |
| Opus 5.5 low | 81.8% | 83.9% | 23.9% | $0.179 | 8s |
| Opus 5.5 high | 82.4% | 85.0% | 24.0% | $0.201 | 18s |
| Sonnet 5 medium | 75.7% | 83.2% | 25.0% | $0.110 | 11s |
| Sonnet 5 high | 77.7% | 84.0% | 26.2% | $0.127 | 23s |
| Haiku 4.5 low | 67.6% | 73.9% | 23.2% | $0.064 | 32s |

**What the sweep shows.**
- **Why the 20% hedge.** Without a hedge, every Opus effort ties TD on reverts (70.9-73.0% vs 70.3%) at a sixth of its test time, but selects every real failure on only 75-77% of failing pushes against TD's 84.3%. The hedge closes that gap: TD's heuristics rank tests that failed recently high, and many PR failures are in those. 20% is the smallest budget at which Opus medium is level with TD on pushes; its +0.5 points there is noise (95% interval -5.8 to +6.6), so the claim is "not worse", not "better". At 5% it is ahead on reverts (79.1%) but 4.6 points behind on pushes, which is why the earlier 5% proposal was dropped. At 15% it is 0.4 points under TD on pushes, also within noise; 20% was kept for the margin.
- **Why medium effort.** At the 20% hedge, low, medium and high are within 1.3 points of each other on both metrics, which is within run-to-run noise (below). Medium is nominally at or above TD on both and low is 0.4 points under TD on pushes, but those gaps are noise too (low against TD on pushes: -0.4 points, 95% interval -6.8 to +5.8 resampling PRs), so the sweep does not separate the efforts. Medium was picked as the middle setting; it costs 7% more per call than low and is 6s slower at p50, both negligible next to the test jobs.
- **Model size matters more than effort.** Opus at any effort beats Sonnet on reverts by 4-7 points at the 20% hedge.
  - Haiku 4.5 does not support the effort setting, so its `low` label is nominal (select_tests.py always passes `--effort`, here `low`). It is well behind on both metrics.
  - Sonnet sometimes fails to produce schema-valid output after retries (12 of 783 pushes at medium effort, 3 at high). Those are selector failures, which keep TD's split.
- **Spend goes on run-all and the hedge.**
  - Without the safety rules, Opus low alone runs about 3.2% of test time on a typical change.
  - The run-all rules fire on 10 of 190 typical changes, mostly edits to `.ci/pytorch/test.sh` and the test docker images, and account for most of the no-hedge 7.6%. The hedge adds the rest.
  - Narrowing those rules is the cheapest remaining lever. They were kept broad because 8 of the 160 reverts are docker image changes.
- **The LLM and TD miss different things.** On reverts, Opus low (no hedge) catches 24 that TD misses, and TD catches 23 that Opus misses; their union catches 128 of 148 (86%). This is why the hedge works.
- **Run-to-run variance is about 2-3 points.** A second identical Opus medium run gives 81.1% of reverts (vs 83.1%) at the same 24.0% CPU. Two Opus low runs with no hedge give 70.9% and 67.6% on reverts. The second runs were scored on the eval samples only, not the pushes, so run-to-run variance on the push metric is not measured; the bootstrap intervals above cover sampling noise only.
- **The budgets are not nested.** The hedge skips a file that does not fit and keeps filling, so a larger budget can take one large file in place of several small ones and does not always catch more (Opus high catches 83.1% of reverts at 15% and 82.4% at 20%).

**First-push PR metrics.** Earlier versions of this report scored only each PR's earliest failing push, without the flaky filter. For reference, Opus medium with the 20% hedge against TD there: PRs with any attributable failure caught (n=282) 92.6% vs 92.2%; PRs with a failure that reproduced in two build environments or on two pushes (n=176) 98.9% vs 96.6%. The push metric above replaces both: it counts every push, and it excludes flakes by Dr.CI's signals rather than by requiring a second occurrence. Requiring reproduction dropped real failures too: a one-off is often a real bug that only one environment runs, or that the author fixed in the next push.

**Revert misses.** Opus high (no hedge) misses every failure on 40 of the 148 reverts. The AI assistant that wrote this stack hand-checked the misses on an earlier version of the revert truth and scoring (39 of them); a human has not reviewed them.
- About a quarter are not plausibly caused by the reverted commit; they are label noise. Examples:
  - a torchtitan pin bump labeled with a test_testing failure;
  - a docs change labeled with libtorch_agnostic;
  - a shard propagation change labeled with test_max_autotune;
  - a `gc.collect` change labeled with a torch_np dtype test.
- The real misses are mostly generic sweeps. Examples:
  - an OpInfo edit that broke functorch/test_ops;
  - a gradcheck change that broke test_ops;
  - docker image changes, now covered by the run-all rule.
- The prompt already calls these out. They are the next target for prompt iteration; the TD-off holdout and enforced PRs that are later reverted on main will say whether they happen in practice (see Rollout).

**Agentic mode.**
- **Setup.** Sonnet 5 medium was rerun on 169 of the samples (80 PR, 60 revert, 29 neutral) with Read/Grep/Glob over a checkout of the change.
- **It made no difference.** Revert recall was identical, 30 of the subset's 56 scored reverts (53.6%) with and without tools. On the subset's 49 PRs with a reproduced failure, it dropped from 46 (93.9%) to 45 (91.8%).
- **The model rarely used the tools.** In about 63% of samples it answered without calling one.
- **It added cost and latency.** Per call it cost $0.150 vs $0.134 on PRs and $0.116 vs $0.095 on reverts, and p90 latency rose from 29s to 37s.
- **Opus without tools is better.** On the same subset, Opus 5.5 low with no tools catches all 49 PRs (100%) and 40 of the 56 reverts (71.4%).
- **Conclusion.** Spend on a bigger model, not on letting a smaller one explore. CI runs with `--tools ""`, which also removes any route for PR content to drive file reads.

## Caveats

- **The TD replay is optimistic.** It uses today's heuristic stats, which include these failures. PR truth is also censored toward TD's picks. Both favor old. Failures in jobs without TD (inductor unit tests, periodic, slow, ...) are left out because both policies run everything there.
- **No PR history in the replay.** In CI, TD's "previously failed in this PR" heuristic ranks a push's earlier failures first, which matters on later pushes. The replay leaves it out for both policies. In production new gets at least the same boost: under enforcement run_test.py always runs the files that already failed on the PR and that TD runs, on top of the selection and the hedge. How much it moves either number is not measured here.
- **The flaky signals are approximations.** Like Dr.CI, a failure is matched by test, here within 14 days on either side. A real failure that another author also hit (two PRs breaking the same test, or a stack split across authors) is counted as flaky, and a flaky test that failed only on this PR is counted as real.
- **Label noise.** About a quarter of the remaining revert misses look unrelated to the reverted change, so true revert recall is higher than reported.
- **Training-data contamination.** Reverts before mid-2026 may predate the models' knowledge cutoff. The PR samples (October 2026) do not, and they show the same ordering of models.
- **Prompt injection.**
  - A PR's description or diff can try to talk the model into selecting nothing.
  - Mitigations: enforcement is opt-in per PR through the `ci-llm-td` label, read live when the workflow run resolves the selection; the selector workflow runs only for same-repo PRs, from the default branch's copy of the workflow and code (since GitHub's 2025-12-08 change, `pull_request_target` always uses the default branch, whatever the PR's base), and reads the diff through git and the title and description from the event payload (`GITHUB_EVENT_PATH`), never running PR code; main always runs everything.
  - The worst case is the same as a TD miss today: a regression is caught on main and reverted.
- **Stacked PRs see a smaller diff in production.** The backtest diffs every change against its merge base with main. In CI, `llm-td.yml` diffs a ghstack PR against its own base, the PR below it, so the model sees only that PR's change, not the stack's. How that moves recall is not measured here.
- **Deployment noise.** The CI workflow uses the existing `bedrock` environment. Its behavior there (latency, throttling) is not measured here. The shadow selection metric will measure it.

## Rollout

1. **Shadow (this stack).**
   - `llm-td.yml`, a `pull_request_target` workflow, writes a selection for every internal PR push as the artifact `llm_td_selection_<PR number>_<head sha>`.
   - The workflow run's `target-determination` job resolves the selection once (`resolve_selection.py`) and uploads it as `llm_td_selection.json`; the test jobs' download-td-artifacts step only downloads that file from their own workflow run. The provenance check is `selector_runs` in `resolve_selection.py`: it takes the artifact named for this PR and head only from an `llm-td.yml` `pull_request_target` run for that head whose head branch is in this repository, and decides enforcement from the PR's live labels (`ci-llm-td` present, `ci-no-td` absent) in a workflow where TD is on. The artifact's own flag can only turn enforcement off, so a planted or stale artifact cannot enforce a selection on a PR without the label. A selector failure, or a missing or malformed artifact, keeps TD's split.
   - run_test.py logs `llm_td_shadow_selection` (what the selection would run, and its time) and `llm_td_shadow_failure` (each failing test, with whether the model's picks covered it, whether the hedge or the previously-failed rule did, and whether it was in TD's include list).
   - **What the shadow failure metric does not measure.** With TD on, only TD's include list runs, so it sees only failures in files TD picked: it compares the selection with TD, not with the truth. Under enforcement only the selected, hedged and previously failed files run, so every observed failure is covered by construction. Live recall comes from two other sources: the `td-holdout-unstable` workflow, which runs only the `default` and `distributed` configs of the linux-jammy-py3.11-gcc11 build and the `dynamo_wrapped` config of the clang21 build, both CPU, with TD off, and only when a PR is opened (its rows, with `td_enabled` false, see every failure in those configs), and enforced PRs that later break main, where everything runs.
   - Gate the next step on a few weeks of the holdout's recall matching the backtest's push recall restricted to the same configs, and of the selection's run time matching the typical-change shares.
2. **Opt-in enforcement.** `ci-llm-td` enforces the selection, with the 20% hedge, on that PR.
3. **Default enforcement.** Enforce on every same-repo PR, with `ci-no-td` as the opt-out. This unlocks about $190k a month and the merge-wait drop above, less the fork-PR share (see Spend). Track reverts of enforced PRs on main as the live check of revert recall.
4. **Inductor, periodic and slow.** Periodic and slow already have a `target-determination` job; add them to the `--enable-td` check in run_test.py, which turns TD on for them, and to `TD_ENABLED` in `target_determination.yml`. Inductor first needs a `target-determination` job in `inductor.yml` (without TD's ranking there is no TD to enforce over and no hedge), then the same two changes. Inductor's unit tests are the largest relative saving, about $40k a month; periodic and slow are about $4k.
5. **Benchmarks and shards.** Gate the inductor model benchmarks on a yes/no model answer, and skip shards with nothing selected.

**Future work: a persistent per-test summary.** Each call re-reads the candidate list with nothing but file names and CI minutes. A summary per test file, of what it exercises and which source areas have broken it before, generated once and refreshed as tests change, would give the model the knowledge TD's heuristics encode, and could let the hedge shrink.

## Reproducing

The reported PR and neutral samples come from 2026-10-01 to 2026-10-25 and the reverts from a wider window starting 2026-04-01, so build_dataset runs three times into the same `$DATA` (`--end` is exclusive; `head` needs the `pr` samples). The sample ids the report is scored on are checked in under `eval/ids/`. A fresh build will not match them exactly, since ClickHouse and main keep moving.

**Samples, test times and the TD replay.** `run_eval`, `td_replay`, `score` and `dollars` read the per-config test times from `$DATA/test-times.json`, and `td_replay` reads TD's historical-stats artifacts from `.additional_ci_files`. Both are what CI downloads from test-infra's `generated-stats` branch. The report's snapshot is not checked in, so a fresh download has today's stats.

```
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-10-01 --end 2026-10-26 --kinds pr neutral
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-10-01 --end 2026-10-26 --kinds head
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-04-01 --end 2026-10-26 --kinds revert
python -c "from tools.stats import import_test_stats as s; [f() for f in (s.get_test_times, s.get_test_class_times, s.get_test_file_ratings, s.get_test_class_ratings, s.get_td_heuristic_historial_edited_files_json, s.get_td_heuristic_profiling_json)]"
cp .additional_ci_files/test-times.json $DATA/
python -m tools.testing.llm_td.eval.td_replay --data $DATA
```

**Model runs**, one tag per Sweep row (results are cached per sample, so a rerun only fills in failed calls):

```
IDS=tools/testing/llm_td/eval/ids
cat $IDS/eval.txt $IDS/head.txt > all.txt
OPUS=us.anthropic.claude-opus-5-5 SONNET=us.anthropic.claude-sonnet-5 HAIKU=us.anthropic.claude-haiku-4-5-20251001-v1:0
run() { python -m tools.testing.llm_td.eval.run_eval --data $DATA --tag $1 --model $2 --effort $3 --ids ${4:-all.txt} $5; }
run v2_opus55_low $OPUS low; run v2_opus55_medium $OPUS medium; run v2_opus55_high $OPUS high
run v2_sonnet5_medium $SONNET medium; run v2_sonnet5_high $SONNET high; run v2_haiku45_low $HAIKU low
run v2_opus55_medium_r2 $OPUS medium $IDS/eval.txt; run v2_opus55_low_r2 $OPUS low $IDS/eval.txt
run v2_sonnet5_medium_agentic $SONNET medium $IDS/agentic.txt --agentic
```

**Scoring, chart and dollars.** The eval ids give the revert, first-push PR and typical-change rows, the head ids the failing-push rows, and `--all-truth` the first-push "any attributable failure" row.

```
S="python -m tools.testing.llm_td.eval.score --data $DATA"
TAGS="v2_haiku45_low v2_sonnet5_medium v2_sonnet5_high v2_opus55_low v2_opus55_medium v2_opus55_high td"
H=final,final+td5,final+td10,final+td15,final+td20
$S --ids $IDS/eval.txt --policies model,$H --json rows_eval.json $TAGS v2_opus55_medium_r2 v2_opus55_low_r2
$S --ids $IDS/head.txt --policies $H --json rows_head.json $TAGS
$S --ids $IDS/eval.txt --all-truth --policies final+td20 v2_opus55_medium td
$S --ids $IDS/agentic.txt --policies model,final v2_sonnet5_medium v2_sonnet5_medium_agentic v2_opus55_low
jq -s add rows_eval.json rows_head.json > rows_chart.json
python -m tools.testing.llm_td.eval.chart rows_chart.json tools/testing/llm_td/report_tradeoff.svg
# The overhead table, then the saving at each hedge budget (Sweep).
for o in 0.1 0.2 0.35 0.5; do python -m tools.testing.llm_td.eval.dollars --data $DATA --overhead $o v2_opus55_medium final+td20; done
for p in final final+td5 final+td10 final+td15; do python -m tools.testing.llm_td.eval.dollars --data $DATA v2_opus55_medium $p; done
```

The intervals in the TL;DR are a paired bootstrap over `rows_chart.json` (10,000 resamples, seed 0) of the per-sample new-minus-old outcome for `v2_opus55_medium final+td20` against `td td25`: reverts resampled per commit, failing pushes per PR (the PR number is the second `_`-separated field of a head id) and, for reference, per push. The p-values are an exact two-sided McNemar test on the discordant counts.

**Dollar constants.** `dollars.WORKFLOWS` holds, per workflow and runner kind, the 30-day test-job cost, the PR share of its test hours, the fork share of those PR hours, and whether TD runs there; `PR_PUSHES_PER_MONTH_K` is the same-repo PR head count. All come from ClickHouse, over the 30 days to the query date:

```sql
-- Test-job cost ($k): misc.runner_cost
SELECT workflow_name, gpu, round(sum(cost) / 1000, 1) k FROM misc.runner_cost
WHERE group_repo ILIKE '%pytorch/pytorch%' AND date >= today() - 30 AND job_name ILIKE '%test%'
  AND workflow_name IN ('pull', 'trunk', 'inductor', 'periodic', 'slow')
GROUP BY workflow_name, gpu;

-- PR share of test hours: jobs not on main or trunk/ branches
SELECT workflow_name, round(sumIf(h, pr) / sum(h), 2) pr_share FROM (
  SELECT workflow_name, NOT (head_branch = 'main' OR startsWith(head_branch, 'trunk/')) pr,
    dateDiff('second', started_at, completed_at) / 3600 h
  FROM default.workflow_job FINAL
  WHERE repository_full_name = 'pytorch/pytorch' AND started_at >= now() - INTERVAL 30 DAY
    AND completed_at > started_at AND name LIKE '%/ test (%'
    AND workflow_name IN ('pull', 'trunk', 'inductor', 'periodic', 'slow'))
GROUP BY workflow_name;

-- Inductor's unit-test share (0.83 of CPU and 0.38 of GPU test hours): the rest are model benchmarks
SELECT arrayExists(l -> match(l, 'gpu|nvidia|a10g|l4|t4|h100|b200|a100|rocm'), labels) gpu,
  match(extract(name, '/ test \\(([^,]+),'), 'huggingface|timm|torchbench|_perf') bench,
  round(sum(dateDiff('second', started_at, completed_at)) / 3600) h
FROM default.workflow_job FINAL
WHERE repository_full_name = 'pytorch/pytorch' AND started_at >= now() - INTERVAL 30 DAY
  AND completed_at > started_at AND workflow_name = 'inductor' AND name LIKE '%/ test (%'
GROUP BY gpu, bench;

-- Fork share of PR test hours: pull runs by their head repo, ciflow/<label>/<PR> pushes by that PR's
WITH runs AS (
  SELECT id, name wf, event, head_repository.full_name = 'pytorch/pytorch' run_same,
    toInt64OrZero(extract(head_branch, '^ciflow/[^/]+/([0-9]+)$')) pr
  FROM default.workflow_run FINAL
  WHERE created_at >= now() - INTERVAL 31 DAY AND repository.full_name = 'pytorch/pytorch'
    AND ((name = 'pull' AND event = 'pull_request')
      OR (name IN ('trunk', 'inductor', 'periodic', 'slow') AND head_branch LIKE 'ciflow/%'))),
prs AS (
  SELECT number, any(head.repo.full_name) = 'pytorch/pytorch' same FROM default.pull_request FINAL
  WHERE number IN (SELECT pr FROM runs WHERE pr > 0) GROUP BY number)
SELECT r.wf, if(r.event = 'pull_request', r.run_same, p.same) same,
  round(sum(dateDiff('second', j.started_at, j.completed_at)) / 3600) h
FROM default.workflow_job j FINAL JOIN runs r ON j.run_id = r.id LEFT JOIN prs p ON p.number = r.pr
WHERE j.repository_full_name = 'pytorch/pytorch' AND j.started_at >= now() - INTERVAL 30 DAY
  AND j.completed_at > j.started_at AND j.name LIKE '%/ test (%'
GROUP BY r.wf, same;

-- Same-repo PR heads with a pull run
SELECT head_repository.full_name = 'pytorch/pytorch' same, uniqExact(head_sha) heads
FROM default.workflow_run FINAL
WHERE created_at >= now() - INTERVAL 30 DAY AND repository.full_name = 'pytorch/pytorch'
  AND name = 'pull' AND event = 'pull_request'
GROUP BY same;
```

**Time to land.** The replay in Time to land's Method was run from ClickHouse with three queries over the 30 days to 2026-11-03, and its simulation script is not part of this stack:
- `default.workflow_run` for pull, trunk, inductor and rocm-mi300: event, head branch and sha, creation time, attempt, conclusion and PR number. PR runs are pull's `pull_request` runs and the others' `ciflow/` pushes.
- `default.workflow_job`, first attempts only: per run, the cancelled and failed job counts, the last build and non-test job completion, and per test job its completion time, `Test` step duration, config and whether its runner is a GPU one.
- `default.merges` joined with `default.issue_comment` and the main `pull` run of the merge commit: comment, merge and first PR run times, for the merge rows.

The simulation shrinks each test job's `Test` step by r, keeps its start time, recomputes each run's TTS as its last job's completion minus its creation, and takes a PR head's wait as the latest TTS of its pull and trunk runs (and inductor's, for those rows). r is new over old test time on a typical change: 24.0/48.7 = 0.493 CPU and 22.4/43.5 = 0.515 GPU where TD runs, and the new shares (0.238 / 0.327) for inductor's unit-test configs, against 1 today.
