# LLM test selection for PyTorch CI: offline evaluation

**TL;DR.** If PRs picked their tests with one Opus 5.5 call (medium effort, plus a 20% TD hedge) instead of today's target determination (TD), CI test-job spend would fall **25%, from $935k to $702k a month** ($233k saved, $43k of it GPU), and the p50 wait for pull and trunk would fall from **181 to 119 minutes**. It does this without missing more breakages than TD on either of the two metrics below:

- **Reverted commits caught: 83.1% vs 70.3% for TD** (n=148). Every one of these commits landed and was reverted for breaking a test; "caught" means its PR's CI would have gone red.
- **Failing PR pushes with every real failure selected: 84.8% vs 84.3% for TD** (n=782 pushes, 285 PRs). Flaky failures are excluded with Dr.CI's signals.

| | Old: TD | New: Opus 5.5 medium + 20% hedge |
|---|---|---|
| Reverted commits caught (n=148) | 70.3% | **83.1%** |
| Failing PR pushes with every real failure selected (n=782) | 84.3% | **84.8%** |
| CPU / GPU test time run on a typical change | 48.7% / 43.5% | **24.0% / 22.4%** |
| Test-job spend | $935k/mo | **$702k/mo** |
| pull + trunk wait, p50 / p90 | 181 / 291 min | **119 / 206 min** |

- **The idea.** One `claude -p` call per PR push sees the title, description, diff and every candidate test file annotated with its CI minutes, and returns the files worth running.
- **Old and new.** "Old" is TD as it runs on PRs today: its top 25% of test files per config, in the workflows that have TD. "New" is the proposed production config: Opus 5.5 at medium effort, plus TD's top files up to 20% of each config's test time. Main always runs everything in both.
- **The model calls** cost about $2.2k a month. This stack reaches $190k of the saving, in pull and trunk. The other $43k needs follow-ups: the inductor workflow has to download the selection, and periodic and slow have to enable TD at all.
- **The stack.** The selector, harness, run_test.py integration and a shadow-mode CI workflow are in the PRs below this one. Enforcement is opt-in per PR through the `ci-llm-td` label.

The sections up to Time to land compare old and new only. Other models, effort levels and hedge budgets are in [Sweep](#sweep-other-models-efforts-and-hedges), which is how the config was chosen.

## Proposed production config

| | Old | New |
|---|---|---|
| Selector | TD heuristics, top 25% per config | Opus 5.5, effort medium, no tools, plus TD's top files up to 20% of the config's test time |
| Safety rules | none | edited or added test files always run; malformed output or changes to test infrastructure run everything |
| Runs on | PRs in TD-enabled workflows | the same |
| Main pushes | everything | everything |

## What the backtest measures

We took historical changes, asked each policy offline which test files it would run, and checked that against which tests really failed. A failure is a (test file, CI config) pair, for example `test_ops.py` in the `default` config. Only failures in jobs that run TD today count: the pull, trunk and ROCm workflows. Jobs elsewhere (inductor's unit tests, periodic, slow, mac-mps, ...) run every test under both policies, so their failures cannot tell the two apart and are left out. The two headline metrics are:

- **Reverted commits caught (n=148).** These are commits that landed on main between 2026-04-01 and 2026-10-25 and that the autorevert bot then reverted because they broke tests. Main runs every test, so we know exactly which files started failing, and in which configs, between the landing and the revert. Of 160 such reverts, 148 broke a TD job, with 193 failures in those jobs; the other 12 broke only jobs without TD. A revert is caught if the policy runs at least one of its failures, which is enough for the PR's CI to go red and block the land. This truth is not biased toward either policy, since TD is not involved in choosing what main runs.
- **Failing PR pushes with every real failure selected (n=782).** We took 316 PRs from 2026-10-01 to 2026-10-25 whose CI had a test failure, and every push (head commit) of those PRs, not just the last one. A push counts if one of its TD jobs had a test failure that is not flaky by the signals below; 782 pushes of 285 PRs do, with 1,897 failures. A push counts as caught only if the policy runs **every** one of its real failures, so the author sees the whole breakage in one round instead of finding the rest on the next push. These PRs ran under TD, so we only see failures in files TD chose to run, which favors old.

**Flaky failures** are excluded from the push truth with the signals Dr.CI uses to mark a PR failure flaky or not the PR's fault, applied with hindsight (a backtest can look after the failure, which Dr.CI cannot). A failure is flaky if any of these holds:

- the same job passed on a rerun of the same commit;
- the job is marked `unstable`;
- the test was disabled (a `DISABLED` issue) within 14 days;
- the same test failed on main, or on another author's PR, within 14 days.

Of 20,094 failing tests in TD jobs on these PRs' pushes, 25.6% are flaky by these signals. The signals overlap: 24.9% failed on another author's PR, 19.4% on main, 13.3% were disabled, 0.9% passed on a rerun and 0.1% were in unstable jobs. The 74.4% that remain are what the push metric scores. Reverts need no such filter: the autorevert bot already requires the failure to start at the reverted commit and persist.

**Test time run** is the mean, over 190 random main commits from 2026-10-01 to 2026-10-25 with no known breakage, of the share of a full CI pass's test-seconds a policy runs, with CPU and GPU counted separately, over the configs that some TD job runs (the inductor unit-test configs are priced separately below). The dollars come from this number.

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

- On reverts, new catches 29 that old misses, and old catches 10 that new misses.
- On failing pushes the two are level: new selects every real failure on 86 pushes where old does not, and old on 82 where new does not.
- Old's numbers come from a replay of today's TD that is optimistic for it; see Caveats.

## Spend and savings

30-day runner cost of test jobs, from `misc.runner_cost`. Builds, lint and other non-test jobs are not counted. Only the PR-triggered part of each workflow changes; main pushes and scheduled runs keep running everything.

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

- **Saved:** $233k a month, $191k CPU and $43k GPU. The first three rows, which this stack covers, are $190k.
- **Follow-ups.** Inductor ($40k) needs `inductor.yml` to download the selection. Periodic and slow ($4k) run every test on PRs today because run_test.py turns TD off outside pull, trunk and ROCm, and the selection is only enforced where TD is on; those rows assume TD is extended to them.
- **Why 25% and not 50%.** New runs about half the PR test time old does, but three things dilute that. Main and scheduled runs keep running everything. The inductor model benchmarks, which PRs trigger, are not test files. And the 20% fixed job overhead is not saved (see Dollar model).
- In inductor only the run_test.py unit-test configs are selectable. The model benchmarks (huggingface, timm, torchbench) are most of its GPU spend.
- **Not in the table:** rocm-mi300 and the other ROCm workflows also download the selection, but their runners are carried at $0 in `misc.runner_cost`, so they show up only under Time to land and Capacity. The remaining callers (win, mac, s390x, xpu) cost under $1k a month together or are also carried at $0.

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

Time to signal (TTS) per workflow, in minutes:

| Workflow | Old p50 | New p50 | Old p90 | New p90 |
|---|---|---|---|---|
| pull | 70 | 46 | 95 | 78 |
| trunk | 182 | 120 | 277 | 206 |
| inductor | 136 | 111 | 223 | 185 |
| rocm-mi300 | 208 | 127 | 313 | 223 |

- **Merge wait.** 1,643 PR heads ran pull and `ciflow/trunk` together. On those, the p50 drops by about an hour and the p90 by 1.4 hours. Trunk is the later signal for nearly all of them in both. With `ciflow/inductor` as well (1,186 heads), the inductor benchmarks bound the wait.
- **Trunk is where the latency is.** Its long test shards are the critical path, so the test shrinkage shows up as wall-clock time.
- **Pull leans toward build-bound.** Runs that wait on a build go from 12% to 34%. The floor is the xpu build (p50 40 / p90 49 min from creation to done) and the ROCm build (p50 23 / p90 53).
- **The model matches real merges.** When the merge comment started trunk (n=337), comment to merge took p50 185 / p90 269 min, against a simulated trunk TTS of 182 / 277. When trunk had already run before the comment (n=533), the merge took p50 3 min.
- **CI is a minority of end-to-end time to land.** From the first pull run on the merged head to the merge is p50 15.8 h and p90 94.5 h, dominated by review. The saving matters most for authors iterating on red CI, where every push pays it again.

**Capacity.**
- PR test job-hours across pull, trunk, inductor and rocm-mi300 go from 675k to 393k a month. `Test` is 70-95% of test-job time on every major runner type.
- Median runner queues are already 1-3 minutes. The tails are long: a10g p90 14 min, l4 p90 11, macos-m1 p90 59, mi300 p90 56, and the 4-GPU a10g pool p50 11 / p90 135. Freed capacity should cut those tails; this is a qualitative argument, not a model.

**Method.**
- For every PR workflow run in the 30 days to 2026-11-03, each test job's `Test` step is shrunk by the ratio r of new to old test time for that job. Each job keeps its start time. Builds and non-test jobs are unchanged, shards are not rebalanced, and queueing effects are ignored.
- In pull, trunk and rocm-mi300, r is 0.493 (CPU) and 0.515 (GPU), from the typical-change test-time shares. In inductor, which has no TD, r is 0.238 / 0.327 for the unit-test configs and 1 for the model benchmarks.
- TTS is the last job's completion minus the run's creation. Only first attempts are used. Cancelled runs, runs with a cancelled job, and runs that failed before any build are excluded.
- r is one ratio per workflow and runner kind, applied to every shard. An unbalanced selection, where one shard keeps most of the selected tests, would make the real latency win smaller.

## Dollar model

- The selectable part of each test job is the Test step. Outside it, checkout, docker pull, build download and teardown take a median of 4.3 to 4.8 minutes, about 12-15% of the job. The model assumes a fixed overhead fraction of 0.2 for that.
- A job's relative cost is `0.2 + 0.8 * f`, where f is the share of test time it runs. For old, f is the replayed TD share in TD workflows and 1 elsewhere. The PR share of each workflow's test hours is then rescaled by new over old.
- **Test-time shares.** pull, trunk, periodic and slow use the typical-change shares from the Accuracy table (old 48.7% CPU / 43.5% GPU, new 24.0% / 22.4%), which are over the configs TD jobs run, so the periodic and slow rows are approximations ($4k together). The inductor rows use the configs of inductor-unittest.yml that no TD job runs (`inductor_core`, `inductor_cpp_wrapper`, `inductor_distributed`, `inductor_amx`, `inductor_aoti_fallback`, `inductor-halide`, `inductor-triton-cpu`), over only the 69 of 190 typical commits that match the `ciflow/inductor` autolabel globs, since those are the PRs that run it; on those, new runs 23.8% of CPU and 32.7% of GPU unit-test time. trunk's own `inductor` config has TD and is priced with trunk.
- **Model cost.** $0.191 per call is the mean over the 666 revert, first-push PR and typical samples, a conservative basis since typical commits average $0.146. At about 11.4k PR pushes a month that is $2.2k a month.

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

**Safety rules.** `Selection.from_model_output` applies these on top of the model's answer:
- A test file the PR edits or adds is always selected.
- The answer becomes run-all when it is missing or malformed, or when the change touches anything that decides how tests run. That covers run_test.py, conftest, common_utils, common_device_type, tools/testing, the test shell scripts and actions, and the CI docker images.

**Hedge.** The hedge adds TD's highest-ranked files with a positive score until they fill 20% of the config's test time (all shards together), each file charged once. A file that does not fit is skipped and filling continues. The harness replays exactly what run_test.py does.

**Cost.**
- Cost is the share of one full CI pass's test-seconds that a selection runs, from `test-times.json`, per config. Configs that no TD job runs are left out; the inductor unit-test ones are priced separately.
- Jobs are classified from `.github/workflows`: by name and config, then with CUDA and Python versions wildcarded (CI has moved to newer ones since the oldest reverts), then by config alone. A reusable workflow's jobs take their caller's TD setting, so `dynamo-unittest` jobs (called from trunk) have TD and `unit-test / ...` jobs (called from inductor) do not.
- `test-times.json` is merged over jobs per config, so a config that TD and non-TD jobs share, like `default` in periodic, is priced as one.
- Seconds are summed over every build environment. Environments matching `cuda|rocm|sm\d+|xpu|h100|b200|a100|gpu` count as GPU.

**Dataset** (`eval/build_dataset.py`, ClickHouse plus git). There are four kinds of sample:

| Kind | n | Window | Truth |
|---|---|---|---|
| revert | 160 | 2026-04-01 to 10-25 | Main commits the autorevert system reverted. The truth is the failures that appeared between landing and the revert, with their configs. |
| head | 783 | pushes of the `pr` samples' PRs | Every push of a `pr` sample's PR with a non-flaky failure in a TD job; the truth is those failures. The diff is against the push's merge base with main. 782 have a failure among the scored candidates. |
| pr | 316 | 2026-10-01 to 10-25 | The PR's earliest failing push, with failures no other author's PR shared and main did not have within 4 days. It defines which PRs `head` covers, and backs the first-push rows in Sweep. |
| neutral | 190 | 2026-10-01 to 10-25 | Random main commits ("typical changes"), with no truth. They measure how much a selector runs on a typical change, which is what the dollars depend on. |

Reverts are the least biased sample. On PRs in TD-enabled workflows, we only ever observe failures in the files TD already chose to run, so PR truth is censored toward TD's picks. Main runs everything, so revert truth is not censored this way.

**Prompt development.**
- The prompt (`prompt.md`) was iterated on a disjoint 436-sample dev set (`eval/ids/dev.txt`).
- The 10 neutral pilot samples that were looked at while writing it were removed from the reported set, leaving 666 (`eval/ids/eval.txt`; its neutral part is `eval/ids/neutral.txt`, and the pushes are `eval/ids/head.txt`). One of the 190 neutral commits was later reverted; build_dataset now excludes reverted commits from neutral draws.
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

**Other models, 20% hedge.** $/call is the mean over the 666 revert, first-push PR and typical samples; p50 wall is the model call alone.

| Model / effort | Reverts caught | Pushes, every failure | Typical CPU | $/call | p50 wall |
|---|---|---|---|---|---|
| **Opus 5.5 medium (production)** | **83.1%** | **84.8%** | **24.0%** | **$0.191** | **14s** |
| Opus 5.5 low | 81.8% | 83.9% | 23.9% | $0.179 | 8s |
| Opus 5.5 high | 82.4% | 85.0% | 24.0% | $0.201 | 18s |
| Sonnet 5 medium | 75.7% | 83.2% | 25.0% | $0.110 | 11s |
| Sonnet 5 high | 77.7% | (running) | 26.2% | $0.127 | 23s |
| Haiku 4.5 low | 67.6% | (running) | 23.2% | $0.064 | 32s |

The Sonnet 5 high and Haiku 4.5 low runs over the 783 failing pushes were still in progress when this was written; they are absent from the chart's right panel.

**What the sweep shows.**
- **Why the 20% hedge.** Without a hedge, every Opus effort ties TD on reverts (70.9-73.0% vs 70.3%) at a sixth of its test time, but selects every real failure on only 75-77% of failing pushes against TD's 84.3%. The hedge closes that gap: TD's heuristics rank tests that failed recently high, and many PR failures are in those. 20% is the smallest budget at which Opus medium is at or above TD on both metrics. At 5% it is ahead on reverts (79.1%) but 4.6 points behind on pushes, which is why the earlier 5% proposal was dropped.
- **Why medium effort.** At the 20% hedge, low, medium and high are within 1.3 points of each other on both metrics, which is within run-to-run noise (below). Medium is at or above TD on both; low is 0.4 points under TD on pushes. Medium costs 7% more per call than low and is 6s slower at p50, both negligible next to the test jobs.
- **Model size matters more than effort.** Opus at any effort beats Sonnet on reverts by 5-7 points at the 20% hedge, and by 9-15 points without one.
  - Haiku 4.5 does not support the effort setting. It is not good enough on reverts.
  - Sonnet sometimes fails to produce schema-valid output after retries (12 of 783 pushes at medium effort). Those fall back to run-all, which is safe but costs.
- **Spend goes on run-all and the hedge.**
  - Without the safety rules, Opus low alone runs about 3.2% of test time on a typical change.
  - The run-all rules fire on 10 of 190 typical changes, mostly edits to `.ci/pytorch/test.sh` and the test docker images, and account for most of the no-hedge 7.6%. The hedge adds the rest.
  - Narrowing those rules is the cheapest remaining lever. They were kept broad because 8 of the 160 reverts are docker image changes.
- **The LLM and TD miss different things.** On reverts, Opus low (no hedge) catches 24 that TD misses, and TD catches 23 that Opus misses; their union catches 128 of 148 (86%). This is why the hedge works.
- **Run-to-run variance is about 2-3 points.** A second identical Opus medium run gives 81.1% of reverts (vs 83.1%) at the same 24.0% CPU. Two Opus low runs with no hedge give 70.9% and 67.6% on reverts.
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
- The prompt already calls these out. They are the next target for prompt iteration, and the shadow metrics will say whether they happen in practice.

**Agentic mode.**
- **Setup.** Sonnet 5 medium was rerun on 169 of the samples (80 PR, 60 revert, 29 neutral) with Read/Grep/Glob over a checkout of the change.
- **It made no difference.** Revert recall was identical, 30 of the subset's 56 scored reverts (53.6%) with and without tools. On the subset's 49 PRs with a reproduced failure, it dropped from 46 (93.9%) to 45 (91.8%).
- **The model rarely used the tools.** In about 63% of samples it answered without calling one.
- **It added cost and latency.** Per call it cost $0.150 vs $0.134 on PRs and $0.116 vs $0.095 on reverts, and p90 latency rose from 29s to 37s.
- **Opus without tools is better.** On the same subset, Opus 5.5 low with no tools catches all 49 PRs (100%) and 40 of the 56 reverts (71.4%).
- **Conclusion.** Spend on a bigger model, not on letting a smaller one explore. CI runs with `--tools ""`, which also removes any route for PR content to drive file reads.

## Caveats

- **The TD replay is optimistic.** It uses today's heuristic stats, which include these failures. PR truth is also censored toward TD's picks. Both favor old. Failures in jobs without TD (inductor unit tests, periodic, slow, ...) are left out because both policies run everything there.
- **No PR history in the replay.** In CI, TD's "previously failed in this PR" heuristic ranks a push's earlier failures first, which matters on later pushes. The replay leaves it out for both policies. In production the hedge draws from the same TD ranking, so new gets the same boost; how much it moves either number is not measured here.
- **The flaky signals are approximations.** Like Dr.CI, a failure is matched by test, here within 14 days on either side. A real failure that another author also hit (two PRs breaking the same test, or a stack split across authors) is counted as flaky, and a flaky test that failed only on this PR is counted as real.
- **Label noise.** About a quarter of the remaining revert misses look unrelated to the reverted change, so true revert recall is higher than reported.
- **Training-data contamination.** Reverts before mid-2026 may predate the models' knowledge cutoff. The PR samples (October 2026) do not, and they show the same ordering of models.
- **Prompt injection.**
  - A PR's description or diff can try to talk the model into selecting nothing.
  - Mitigations: enforcement is opt-in per PR through the `ci-llm-td` label; the selector workflow runs only for pytorch/pytorch branches, from main's copy of the workflow and code, and reads the PR only through git; main always runs everything.
  - The worst case is the same as a TD miss today: a regression is caught on main and reverted.
- **Deployment noise.** The CI workflow uses the existing `bedrock` environment. Its behavior there (latency, throttling) is not measured here. The shadow metrics are what will measure it.

## Rollout

1. **Shadow (this stack).**
   - `llm-td.yml`, a `pull_request_target` workflow, writes a selection for every internal PR push as the artifact `llm_td_selection_<PR number>_<head sha>`.
   - The test jobs' download-td-artifacts step accepts that artifact only from an `llm-td.yml` `pull_request_target` run for the same PR and head in this repository, so another workflow or PR cannot plant one.
   - run_test.py logs `llm_td_shadow_selection` and `llm_td_shadow_failure`. The second metric is a live recall measurement: every failing test, with whether the selection (hedge included) covered it.
   - Gate the next step on a few weeks of it matching the numbers above.
2. **Opt-in enforcement.** `ci-llm-td` enforces the selection, with the 20% hedge, on that PR.
3. **Default enforcement.** Enforce on every internal PR, keeping the label as an opt-out (or `ci-no-td`). This unlocks about $190k a month and the merge-wait drop above.
4. **Inductor, periodic and slow.** Have `inductor.yml` download the selection, and let run_test.py enforce it in periodic and slow, where TD is off today. Inductor's unit tests are the largest relative saving, about $40k a month; periodic and slow are about $4k.
5. **Benchmarks and shards.** Gate the inductor model benchmarks on a yes/no model answer, and skip shards with nothing selected.

**Future work: a persistent per-test summary.** Each call re-reads the candidate list with nothing but file names and CI minutes. A summary per test file, of what it exercises and which source areas have broken it before, generated once and refreshed as tests change, would give the model the knowledge TD's heuristics encode, and could let the hedge shrink.

## Reproducing

The reported PR and neutral samples come from 2026-10-01 to 2026-10-25 and the reverts from a wider window starting 2026-04-01, so build_dataset runs three times into the same `$DATA` (`--end` is exclusive; `head` needs the `pr` samples). The sample ids the report is scored on are checked in under `eval/ids/`. A fresh build will not match them exactly, since ClickHouse and main keep moving.

```
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-10-01 --end 2026-10-26 --kinds pr neutral
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-10-01 --end 2026-10-26 --kinds head
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-04-01 --end 2026-10-26 --kinds revert
IDS=tools/testing/llm_td/eval/ids
cat $IDS/eval.txt $IDS/head.txt > all.txt
python -m tools.testing.llm_td.eval.run_eval --data $DATA --tag v2_opus55_medium --model us.anthropic.claude-opus-5-5 --effort medium --ids all.txt
python -m tools.testing.llm_td.eval.td_replay --data $DATA
python -m tools.testing.llm_td.eval.score --data $DATA --ids all.txt --all-truth --policies final,final+td5,final+td10,final+td15,final+td20 --json rows.json v2_opus55_medium td
python -m tools.testing.llm_td.eval.dollars --data $DATA --overhead 0.2 v2_opus55_medium final+td20
python -m tools.testing.llm_td.eval.chart rows.json tools/testing/llm_td/report_tradeoff.svg
```
