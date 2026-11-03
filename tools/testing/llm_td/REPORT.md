# LLM test selection for PyTorch CI: offline evaluation

**Summary.**
- **The idea.** One `claude -p` call per PR sees the title, description, diff and every candidate test file annotated with its CI minutes, and returns the files worth running.
- **The proposal.** Replace TD's top-25% cut on PRs with **Opus 5.5 at low effort plus a 5% TD hedge**. Below, "old" is TD as it runs today and "new" is this config. Main always runs everything in both.
- **Accuracy.** On 666 held-out historical changes, new catches 97.8% of PRs with a reproduced failure (old 97.3%) and 76.9% of the commits autorevert later reverted (old 70.6%).
- **Test time.** New runs 11% of CPU and 12% of GPU test time on a typical change. Old runs 47% and 43%.
- **Spend.** The test jobs of the affected workflows cost **$935k a month** today. New brings that to about **$604k**, a saving of **$331k a month ($4.0M a year): $271k CPU and $60k GPU**. Model calls add about $2k. This stack reaches $284k of the saving; the inductor workflow needs a follow-up.
- **Time to land.** The wait for pull and trunk to finish on a push drops from **181 to 96 minutes at p50** and from 291 to 179 at p90.
- **The stack.** The selector, harness, run_test.py integration and a shadow-mode CI workflow come in the PRs below this one. Enforcement is opt-in per PR through the `ci-llm-td` label.

The sections up to Time to land compare old and new only. Other models, effort levels and hedge budgets are in [Sweep](#sweep-other-models-efforts-and-hedges), which is how the config was chosen.

## Proposed production config

| | Old | New |
|---|---|---|
| Selector | TD heuristics, top 25% per config | Opus 5.5, effort low, no tools, plus TD's top files up to 5% of the config's test time |
| Safety rules | none | edited or added test files always run; malformed output or changes to test infrastructure run everything |
| Runs on | PRs in TD-enabled workflows | the same |
| Main pushes | everything | everything |

## Accuracy and test time

"Caught" is the share of changes with at least one failure where the policy caught at least one; "pairs" is recall over (file, config) failures. Test-time shares are means over 190 random main commits, as a share of a full CI pass across all configs.

| | Old | New |
|---|---|---|
| PRs with a reproduced failure caught (n=185) | 97.3% | 97.8% |
| their (file, config) failures caught | 92.5% | 97.3% |
| Reverted commits caught (n=160) | 70.6% | 76.9% |
| their (file, config) failures caught | 62.9% | 72.2% |
| PRs with any attributable failure caught (n=315) | 92.1% | 87.0% |
| CPU test time run on a typical change | 46.9% | 11.3% |
| GPU test time run on a typical change | 43.4% | 11.5% |
| Model call, $ per push / p50 latency | - | $0.178 / 8s |

- New is ahead on everything except unreproduced PR failures. Those are dominated by flaky tests, which TD's historical failure-rate heuristics predict well, and running a known-flaky test on a PR is not a win.
- One of the 316 PR samples has a failure with no test file attached, so 315 have something to catch.
- Old's numbers are a replay of today's TD and are optimistic for it; see Caveats.

## Spend and savings

30-day runner cost of test jobs, from `misc.runner_cost`. Builds, lint and other non-test jobs are not counted. Only the PR-triggered part of each workflow changes; main pushes and scheduled runs keep running everything.

| Workflow | Old $k/mo | New $k/mo |
|---|---|---|
| pull, CPU | 463.6 | 277.6 |
| trunk, CPU | 157.0 | 108.0 |
| trunk, GPU | 158.0 | 112.9 |
| periodic, GPU | 17.0 | 13.3 |
| slow, CPU | 6.3 | 6.0 |
| inductor, CPU (follow-up) | 79.0 | 43.0 |
| inductor, GPU (follow-up) | 54.5 | 43.3 |
| **total** | **935** | **604** |
| model calls | 0 | 2.0 |

- **Saved:** $331k a month, $271k CPU and $60k GPU. The first five rows, which this stack covers, are $284k; inductor is $47k.
- That is 35% of test-job spend, against a 70% cut in PR test time, because $288k of the spend is main and scheduled runs.
- In inductor only the run_test.py unit-test configs are selectable. The model benchmarks (huggingface, timm, torchbench) are most of its GPU spend and are not test files.
- **Not in the table:** rocm-mi300 and the other ROCm workflows also download the selection, but their runners are carried at $0 in `misc.runner_cost`, so they show up only under Time to land and Capacity. The remaining callers (win, mac, s390x, xpu) cost under $1k a month together or are also carried at $0.

## Time to land

| Wait for a PR push | Old (min) | New (min) |
|---|---|---|
| pull and trunk, p50 | 181 | 96 |
| pull and trunk, p90 | 291 | 179 |
| pull and trunk, mean | 205 | 113 |
| pull, trunk and inductor, p50 | 191 | 120 |
| pull, trunk and inductor, p90 | 442 | 388 |
| `@pytorchbot merge` that starts trunk, comment to merge, p50 | 185 | ~93 |
| `@pytorchbot merge` that starts trunk, comment to merge, p90 | 269 | ~181 |

Time to signal (TTS) per workflow, in minutes:

| Workflow | PR runs | Old p50 | New p50 | Old p90 | New p90 |
|---|---|---|---|---|---|
| pull | 6,400 | 70 | 43 | 95 | 77 |
| trunk | 2,981 | 182 | 93 | 277 | 181 |
| inductor | 3,719 | 136 | 111 | 223 | 179 |
| rocm-mi300 | 455 | 208 | 84 | 313 | 188 |

- **Merge wait.** 1,643 PR heads ran pull and `ciflow/trunk` together. On those, new saves about 1.4 hours per push at p50, and the p90 drops by 1.9 hours. Trunk is the later signal for 100% of them in old and 98% in new. With `ciflow/inductor` as well (1,186 heads), the inductor benchmarks bound the wait.
- **Trunk is where the latency is.** Its long test shards are the critical path, so all of the test shrinkage shows up as wall-clock time.
- **Pull becomes build-bound.** Runs that wait on a build go from 12% to 55%. The floor is the xpu build (p50 40 / p90 49 min from creation to done) and the ROCm build (p50 23 / p90 53). Faster builds are the next lever for pull.
- **The model matches real merges.** When the merge comment started trunk (n=337), comment to merge took p50 185 / p90 269 min, against a simulated trunk TTS of 182 / 277. When trunk had already run before the comment (n=533), the merge took p50 3 min.
- **CI is a minority of end-to-end time to land.** From the first pull run on the merged head to the merge is p50 15.8 h and p90 94.5 h, dominated by review. The saving matters most for authors iterating on red CI, where every push pays it again.

**Capacity.**
- PR test job-hours across pull, trunk, inductor and rocm-mi300 go from 675k to 267k a month. `Test` is 70-95% of test-job time on every major runner type.
- Median runner queues are already 1-3 minutes. The tails are long: a10g p90 14 min, l4 p90 11, macos-m1 p90 59, mi300 p90 56, and the 4-GPU a10g pool p50 11 / p90 135. Freed capacity should cut those tails; this is a qualitative argument, not a model.

**Method.**
- For every PR workflow run in the 30 days to 2026-11-03, each test job's `Test` step is shrunk by the ratio r of new to old test time for that job. Each job keeps its start time. Builds and non-test jobs are unchanged, shards are not rebalanced, and queueing effects are ignored.
- In pull, trunk and rocm-mi300, r is 0.24 (CPU) and 0.275 (GPU), from the non-inductor test-time shares in Dollar model below. In inductor, which has no TD, r is 0.12 / 0.13 for the unit-test configs and 1 for the model benchmarks.
- TTS is the last job's completion minus the run's creation. Only first attempts are used. Cancelled runs, runs with a cancelled job, and runs that failed before any build are excluded.
- Only about 30% of PR pull runs survive the filters; most of the rest are superseded pushes that were cancelled. About 2.8k pull and 2.3k trunk runs failed in runner-determinator before any build, which looks like an infra issue worth a separate look.
- r is one ratio per workflow and runner kind, applied to every shard. An unbalanced selection, where one shard keeps most of the selected tests, would make the real latency win smaller.

## Dollar model

- The selectable part of each test job is the Test step. Outside it, checkout, docker pull, build download and teardown take a median of 4.3 to 4.8 minutes, about 12-15% of the job. The model assumes a fixed overhead fraction of 0.2 for that.
- A job's relative cost is `0.2 + 0.8 * f`, where f is the share of test time it runs. For old, f is the replayed TD share in TD workflows and 1 elsewhere. The PR share of each workflow's test hours is then rescaled by new over old.
- **Test-time shares.** pull, trunk, periodic and slow use the shares over the non-inductor configs: old 47.2% CPU / 41.8% GPU, new 11.4% / 11.5%. These differ slightly from the Accuracy table, which counts every config, because the inductor configs are priced separately. The inductor rows use only the 69 of 190 neutral commits that match the `ciflow/inductor` autolabel globs, since those are the PRs that run it; on those, new runs 12.0% of CPU and 13.0% of GPU unit-test time.
- **Model cost.** $0.178 per call is the mean over all 666 samples, a conservative basis since neutral commits average $0.139. At about 11.4k PR pushes a month that is $2.0k a month. The same basis is used for every policy in Sweep.

The overhead fraction dominates the uncertainty:

| Overhead | CPU saved | GPU saved | Total saved |
|---|---|---|---|
| 0.1 | $332k | $74k | $406k/mo |
| **0.2** | **$271k** | **$60k** | **$331k/mo** |
| 0.35 | $197k | $44k | $241k/mo |
| 0.5 | $138k | $31k | $169k/mo |

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

**Hedge.** The 5% hedge adds TD's highest-ranked files with a positive score until they fill 5% of the config's test time (all shards together), each file charged once. A file that does not fit is skipped and filling continues. The harness replays exactly what run_test.py does.

**Cost.**
- Cost is the share of one full CI pass's test-seconds that a selection runs, from `test-times.json`, per config.
- Seconds are summed over every build environment. Environments matching `cuda|rocm|sm\d+|xpu|h100|b200|a100|gpu` count as GPU.

**Dataset** (`eval/build_dataset.py`, ClickHouse plus git). There are three kinds of sample:

| Kind | n | Window | Truth |
|---|---|---|---|
| PR | 316 | 2026-10-01 .. 10-25 | Test files that failed on the PR head and that we can blame on the PR: no other author's PR failed the same test, and main did not fail it within 4 days. "Strong" truth keeps only failures that reproduced on a second run or a later head; 185 PRs have one. |
| revert | 160 | 2026-04-01 .. 10-24 | Main commits the autorevert system reverted. The truth is the failures that appeared between landing and the revert, with their configs. |
| neutral | 190 | recent main | Random main commits, with no truth. They measure how much a selector runs on a typical change, which is what the dollars depend on. |

Reverts are the most informative sample. On PRs in TD-enabled workflows, we only ever observe failures in the files TD already chose to run, so PR truth is censored toward TD's picks. Main runs everything, so revert truth is not censored this way.

**Prompt development.**
- The prompt (`prompt.md`) was iterated on a disjoint 436-sample dev set (`eval/ids/dev.txt`).
- The 10 neutral pilot samples that were looked at while writing it were removed from the reported set, leaving 666 (`eval/ids/eval.txt`; its neutral part is `eval/ids/neutral.txt`). One of the 190 neutral commits was later reverted; build_dataset now excludes reverted commits from neutral draws.
- Every model ran the same prompt and the same samples, with no tools. Median input was about 30k tokens; the candidate list is about 1,100 files.

**TD baseline.**
- `eval/td_replay.py` runs today's in-repo TD heuristics on each sample's changed files and commit text, then takes the top 25% cut CI uses, per config.
- The heuristics read today's historical-stats artifacts, which already contain these failures. The replay is therefore optimistic for TD.

## Sweep: other models, efforts and hedges

This section is how the production config was chosen. Rows use the same samples and definitions as Accuracy and test time; $/call is the mean over all 666 samples; p50 wall is the model call alone.

![caught vs GPU share](report_tradeoff.svg)

Each point is one model and effort level. Filled points are the selection as the model returns it, after the safety rules. Hollow points add the 5% TD hedge. The red square is the replayed TD. The x axis is the GPU share of the sample's own CI pass, which is higher than on a typical change because failing changes are larger. The PR panel uses the 185 PRs with strong truth.

**Models and effort levels, no hedge:**

| Model / effort | PR caught | PR pairs | Revert caught | Revert pairs | Neutral CPU | Neutral GPU | $/call | p50 wall |
|---|---|---|---|---|---|---|---|---|
| Haiku 4.5 low | 84.9% | 75.9% | 45.0% | 42.6% | 6.9% | 6.8% | $0.064 | 32s |
| Haiku 4.5 high | 86.5% | 77.6% | 45.6% | 43.5% | 6.9% | 6.8% | $0.064 | 32s |
| Sonnet 5 low | 95.7% | 91.8% | 54.4% | 50.6% | 8.7% | 8.1% | $0.102 | 5s |
| Sonnet 5 medium | 95.1% | 91.8% | 57.5% | 52.7% | 8.6% | 8.2% | $0.110 | 11s |
| Sonnet 5 high | 95.1% | 92.8% | 63.7% | 60.8% | 10.1% | 9.4% | $0.127 | 23s |
| Opus 5.5 low | 97.3% | 95.3% | 70.6% | 65.8% | 7.5% | 7.3% | $0.178 | 8s |
| Opus 5.5 low (rerun) | 98.4% | 96.0% | 67.5% | 62.0% | 7.8% | 7.5% | $0.179 | 8s |
| Opus 5.5 medium | 98.4% | 96.5% | 70.0% | 66.2% | 7.6% | 7.5% | $0.192 | 14s |
| Opus 5.5 high | 98.4% | 97.0% | 72.5% | 68.4% | 7.7% | 7.7% | $0.201 | 18s |

**With the 5% hedge:**

| Model / effort | PR caught | PR pairs | Revert caught | Revert pairs | Neutral CPU | Neutral GPU | Saved $k/mo | Model $k/mo |
|---|---|---|---|---|---|---|---|---|
| **Opus 5.5 low (production)** | **97.8%** | **97.3%** | **76.9%** | **72.2%** | **11.3%** | **11.5%** | **331** | **2.0** |
| Opus 5.5 high | 98.4% | 97.3% | 77.5% | 73.4% | 11.5% | 11.9% | 329 | 2.3 |
| Sonnet 5 medium | 97.8% | 96.3% | 69.4% | 65.8% | 12.5% | 12.5% | 322 | 1.3 |
| Haiku 4.5 low | 90.8% | 84.1% | 60.0% | 57.0% | 10.8% | 11.0% | 337 | 0.7 |

Opus 5.5 low without the hedge would save $366k but catches 6.3 points fewer reverts. Haiku saves more only because it runs less, and catches 17 points fewer reverts than production.

**Hedge budgets, Opus 5.5 low:**

| Hedge | Revert caught | Neutral CPU |
|---|---|---|
| 0% | 70.6% | 7.5% |
| **5% (production)** | **76.9%** | **11.3%** |
| 10% | 78.1% | 15.4% |
| 15% | 82.5% | 19.3% |
| 20% | 81.2% | 23.4% |

The budgets are not nested: the hedge skips a file that does not fit and keeps filling, so a larger budget can take one large file in place of several small ones and lose a catch. That is why 20% is below 15%.

**What the sweep shows.**
- **Model size matters more than effort.** Opus at low effort beats Sonnet at high effort on reverts (70.6% vs 63.7%) and is about 3x faster (8s vs 23s p50).
  - Effort helps Opus by about two points on reverts, which is within run-to-run noise.
  - Haiku 4.5 does not support the effort setting, so its two runs are effectively replicates. It is not good enough on reverts.
  - Sonnet sometimes fails to produce schema-valid output after retries: 12 of 666 samples at low effort. Those fall back to run-all, which is safe but costs.
- **Spend goes almost entirely on run-all.**
  - Without the safety rules, Opus low alone runs about 3.2% of test time on a neutral change.
  - The run-all rules fire on 10 of 190 neutral commits, mostly edits to `.ci/pytorch/test.sh` and the test docker images, and account for most of the rest of the 7.5%.
  - Narrowing those rules is the cheapest remaining lever. They were kept broad because 8 of the 160 reverts are docker image changes.
- **The LLM and TD miss different things.** On reverts, Opus low catches 25 that TD misses, and TD catches 25 that Opus misses; their union catches 138 of 160 (86%). Running all of TD's top 25% on top of the model would cost what TD costs now, which is why the hedge is a small budget. 5% is the recommended start because many of the misses it leaves are label noise.
- **Run-to-run variance is about 3 points.** Two identical Opus low runs give 70.6% and 67.5% on reverts, so differences of 2 or 3 points between neighbors are not significant.

**Revert misses.** Opus high (no hedge) misses every failure on 44 of the 160 reverts. The AI assistant that wrote this stack hand-checked the misses on an earlier version of the revert truth (39 of them); a human has not reviewed them.
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
- **It made no difference.** Revert recall was identical at 55.0% caught, with and without tools. PR recall dropped from 94.1% to 92.2%.
- **The model rarely used the tools.** In about 63% of samples it answered without calling one.
- **It added cost and latency.** Per call it cost $0.150 vs $0.134 on PRs and $0.116 vs $0.095 on reverts, and p90 latency rose from 29s to 37s.
- **Opus without tools is better.** On the same subset, Opus 5.5 low with no tools reaches 100% PR and 71.7% revert caught.
- **Conclusion.** Spend on a bigger model, not on letting a smaller one explore. CI runs with `--tools ""`, which also removes any route for PR content to drive file reads.

## Caveats

- **The TD replay is optimistic.** It uses today's heuristic stats, which include these failures. PR truth is also censored toward TD's picks. Both favor old, so new's relative standing is if anything understated.
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
2. **Opt-in enforcement.** `ci-llm-td` enforces the selection, with the 5% hedge, on that PR.
3. **Default enforcement.** Enforce on every internal PR, keeping the label as an opt-out (or `ci-no-td`). This unlocks about $284k a month and the merge-wait drop above.
4. **Inductor.** Have `inductor.yml` download the selection. It has no TD today, so its unit tests are the largest relative saving: about $47k a month.
5. **Benchmarks and shards.** Gate the inductor model benchmarks on a yes/no model answer, and skip shards with nothing selected.

## Reproducing

The reported PR and neutral samples come from a 2026-10-01 .. 10-26 window and the reverts from a wider 2026-04-01 one, so build_dataset runs twice into the same `$DATA`. The sample ids the report is scored on are checked in under `eval/ids/`. A fresh build will not match them exactly, since ClickHouse and main keep moving; the numbers above were scored from `samples_v2/` and `td_v2/`, the outputs of rerunning build_dataset and td_replay after the revert-truth fix.

```
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-10-01 --end 2026-10-26 --kinds pr neutral
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-04-01 --end 2026-10-26 --kinds revert
IDS=tools/testing/llm_td/eval/ids
python -m tools.testing.llm_td.eval.run_eval --data $DATA --tag v2_opus55_low --model us.anthropic.claude-opus-5-5 --effort low --ids $IDS/eval.txt
python -m tools.testing.llm_td.eval.td_replay --data $DATA
S="python -m tools.testing.llm_td.eval.score --data $DATA --ids $IDS/eval.txt"
$S --policies model,final,final+td5 --json rows.json v2_opus55_low td
$S --policies final+td5,final+td10,final+td15,final+td20 v2_opus55_low
$S --all-truth --policies final,final+td5 v2_opus55_low td
python -m tools.testing.llm_td.eval.dollars --data $DATA --overhead 0.2 v2_opus55_low final+td5
python -m tools.testing.llm_td.eval.chart rows.json tools/testing/llm_td/report_tradeoff.svg
```
