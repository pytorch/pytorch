# LLM test selection for PyTorch CI: offline evaluation

**Summary.**
- **The idea.** One `claude -p` call per PR sees the title, description, diff and every candidate test file annotated with its CI minutes, and returns the files worth running.
- **Accuracy.** On 666 held-out historical changes, Opus 5.5 at low effort with a small TD hedge catches 97.8% of PRs with a reproduced failure and 76.9% of the commits autorevert later reverted. TD today catches 97.3% and 70.6%.
- **Test time.** It runs 11% of CPU and 12% of GPU test time on a typical change, against 47% and 43% for TD's top 25% cut.
- **Spend today.** The test jobs of the workflows run_test.py drives cost about **$935k a month**. $647k of that is PR-triggered test time a selector can cut.
- **Savings.** The dollar model puts the saving at about **$331k a month ($4.0M a year): $271k CPU and $60k GPU**, or 35% of the $935k. Model calls cost about $2k a month. This stack alone reaches about $285k of it; the inductor workflow needs a follow-up.
- **Time to land.** The wait for pull and trunk to finish on a push drops from **181 to 96 minutes at p50 and from 291 to 179 at p90**.
- **The stack.** The selector, harness, run_test.py integration and a shadow-mode CI workflow come in the PRs below this one. Enforcement is opt-in per PR through the `ci-llm-td` label. Main always runs everything.

![caught vs GPU share](report_tradeoff.svg)

Each point is one model and effort level. Filled points are the selection as the model returns it, after the safety rules. Hollow points add the 5% TD hedge. The red square is the replayed TD at its CI cut. The x axis is the GPU share of the sample's own CI pass, which is higher than on a typical change because failing changes are larger. The PR panel uses the 185 PRs with strong truth.

## Current spend and savings

All figures are 30-day runner cost of test jobs, from ClickHouse job durations times runner price. Builds, lint and other non-test jobs are not counted. The recommended policy is Opus 5.5 low + 5% TD hedge, with a job overhead of 0.2 (see Dollars).

| Workflow | Runner | Test-job spend $k/mo | PR-triggered, selectable $k/mo | Saved $k/mo | Saved, share of spend | In this stack |
|---|---|---|---|---|---|---|
| pull | CPU | 463.6 | 375.5 | 186.0 | 40% | yes |
| trunk | CPU | 157.0 | 98.9 | 49.0 | 31% | yes |
| trunk | GPU | 158.0 | 99.5 | 45.1 | 29% | yes |
| inductor | CPU | 79.0 | 51.1 | 36.0 | 46% | follow-up |
| inductor | GPU | 54.5 | 16.2 | 11.2 | 21% | follow-up |
| periodic | GPU | 17.0 | 5.3 | 3.7 | 22% | yes |
| slow | CPU | 6.3 | 0.4 | 0.3 | 5% | yes |
| **total** | | **935** | **647** | **331 (271 CPU, 60 GPU)** | **35%** | **285 now, 47 later** |

- The selectable column is the PR share of each workflow's test hours. For inductor it is only the run_test.py unit-test configs. The model benchmarks (huggingface, timm, torchbench) are most of inductor's GPU spend and are not test files.
- Main pushes and scheduled runs keep running everything, which is why the saving is 35% of spend rather than the 70% cut in PR test time.
- The model calls add about $2.0k a month, so the net saving is about $329k.

## Time to land

**Method.**
- For every PR workflow run in the 30 days to 2026-11-03, each test job's `Test` step is shrunk by the ratio r of LLM to today's test time for that job. Each job keeps its start time. Builds and non-test jobs are unchanged, shards are not rebalanced, and queueing effects are ignored.
- In pull, trunk and rocm-mi300, r is 0.24 (CPU) and 0.275 (GPU): the LLM share of test time over replayed TD's share on neutral commits. In inductor, which has no TD, r is 0.12 / 0.13 for the unit-test configs and 1 for the model benchmarks.
- Time to signal (TTS) is the last job's completion minus the run's creation. Only first attempts are used. Cancelled runs, runs with a cancelled job, and runs that failed before any build are excluded.

**Merge wait.**
- For the 1,643 PR heads where pull and `ciflow/trunk` started together, the wait for both goes from **181 to 96 minutes at p50 and from 291 to 179 at p90** (mean 205 to 113).
- That is about **1.4 hours saved per push at p50 and 2.5 hours at p90**.
- Trunk is the later signal for 100% of these heads today and 98% after.
- For the 1,186 heads that also ran `ciflow/inductor`, the wait for all three goes from 191 to 120 min at p50 and from 442 to 388 at p90. There the inductor benchmarks bound it.

| Workflow | PR runs | TTS p50 old -> new (min) | TTS p90 old -> new (min) | Mean saved (min) | Runs bound by a build, old -> new |
|---|---|---|---|---|---|
| pull | 6,400 | 70 -> 43 | 95 -> 77 | 22 | 12% -> 55% |
| trunk | 2,981 | 182 -> 93 | 277 -> 181 | 92 | 0.1% -> 0.8% |
| inductor | 3,719 | 136 -> 111 | 223 -> 179 | 34 | 0% -> 0.1% |
| rocm-mi300 | 455 | 208 -> 84 | 313 -> 188 | 112 | 0% -> 0% |

- **Trunk is where the latency is.** Its long test shards are the critical path, so all of the test shrinkage shows up as wall-clock time and its p50 roughly halves.
- **Pull becomes build-bound.** After the change, 55% of pull runs wait on a build. The floor is the xpu build (p50 40 / p90 49 min from creation to done) and the ROCm build (p50 23 / p90 53). Faster builds are the next lever for pull.
- **The model matches real merges.** When `@pytorchbot merge` itself started trunk (n=337), comment to merge took p50 185 / p90 269 min, against the simulated trunk TTS of 182 / 277. Those merges would drop to about 93 / 181 min. When trunk had already run before the comment (n=533), the merge took p50 3 min, because the CI wait was paid earlier, at push time.
- **CI is a minority of end-to-end time to land.** From the first pull run on the merged head to the merge is p50 15.8 h and p90 94.5 h, dominated by review. The 1.4 h is a real but minority share of that. It matters most for authors iterating on red CI, where every push pays it again.

**Capacity.**
- The change frees about 60% of PR test runner-hours: 408k of 675k job-hours a month across pull, trunk, inductor and rocm-mi300.
- `Test` is 70-95% of test-job time on every major runner type.
- Median runner queues are already 1-3 minutes, so a typical run gains little there. The tails are long, though: a10g p90 14 min, l4 p90 11, macos-m1 p90 59, mi300 p90 56, and the 4-GPU a10g pool p50 11 / p90 135. Freed capacity should cut those tails. This is a qualitative argument, not a model.

**Caveats.**
- Only about 30% of PR pull runs survive the filters. Most of the rest are superseded pushes that were cancelled.
- About 2.8k pull and 2.3k trunk runs failed in runner-determinator before any build. That looks like an infra issue worth a separate look.
- r is one ratio per workflow and runner kind, applied to every shard. An unbalanced selection, where one shard keeps most of the selected tests, would make the real latency win smaller. Rebalancing shards later trades latency for even more capacity.

## Method

**Unit of selection.**
- The unit is a test file under `test/`. Selecting a file runs it in every config it belongs to: default, distributed, slow, inductor and so on.
- The two wrapper sweeps, `dynamo_wrapped` and `crossref`, re-run hundreds of ordinary files in a special mode. The model requests them separately; an unrequested sweep runs only the selected files.
- The model can also request whole directories (`dynamo`, `inductor`, ...) or `run_all`.

**Safety rules (the "final" policy).** `Selection.from_model_output` applies these on top of the model's answer:
- A test file the PR edits or adds is always selected.
- The answer becomes run-all when it is missing or malformed, or when the change touches anything that decides how tests run. That covers run_test.py, conftest, common_utils, common_device_type, tools/testing, the test shell scripts and actions, and the CI docker images.

**Hedge.** `final+td5` adds TD's highest-ranked files with a positive score until they fill 5% of the config's test time (all shards together), each file charged once. The harness replays exactly what run_test.py does.

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
- The prompt (`prompt.md`) was iterated on a disjoint 436-sample dev set.
- The 10 pilot samples that were looked at while writing it were removed from the reported set, leaving 666.
- Every model ran the same prompt and the same samples, with no tools. Median input was about 30k tokens; the candidate list is about 1,100 files.

**TD baseline.**
- `eval/td_replay.py` runs today's in-repo TD heuristics on each sample's changed files and commit text, then takes the top 25% cut CI uses, per config.
- The heuristics read today's historical-stats artifacts, which already contain these failures. The replay is therefore optimistic for TD.

## Results

"Caught" means the share of changes with at least one failure where the policy caught at least one; "pairs" means recall over (file, config) failures. CPU/GPU are mean shares of a full pass on the neutral commits. PR rows use strong truth (n=185). $/call is the mean over all 666 samples; p50 wall is the model call alone.

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
| **Opus 5.5 low + 5% TD hedge** | **97.8%** | 97.3% | **76.9%** | 72.2% | **11.3%** | **11.5%** | $0.178 | 8s |
| Opus 5.5 high + 5% TD hedge | 98.4% | 97.3% | 77.5% | 73.4% | 11.5% | 11.9% | $0.201 | 18s |
| Sonnet 5 medium + 5% TD hedge | 97.8% | 96.3% | 69.4% | 65.8% | 12.5% | 12.5% | $0.110 | 11s |
| TD top 25% (replayed) | 97.3% | 92.5% | 70.6% | 62.9% | 46.9% | 43.4% | - | - |

**What the table shows.**

- **Model size matters more than effort.** Opus at low effort beats Sonnet at high effort on reverts (70.6% vs 63.7%) at a similar latency.
  - Effort helps Opus by about two points on reverts, which is within run-to-run noise.
  - Haiku 4.5 does not support the effort setting, so its two runs are effectively replicates. It is not good enough on reverts.
  - Sonnet sometimes fails to produce schema-valid output after retries: 12 of 666 samples at low effort. Those fall back to run-all, which is safe but costs.
- **Spend goes almost entirely on run-all.**
  - Without the safety rules, the model alone runs about 3.2% of test time on a neutral change.
  - The run-all rules fire on 10 of 190 neutral commits, mostly edits to `.ci/pytorch/test.sh` and the test docker images, and account for most of the rest of the 7.5%.
  - Narrowing those rules is the cheapest remaining lever. They were kept broad because 8 of the 160 reverts are docker image changes.
- **The LLM and TD miss different things.**
  - On reverts, Opus low catches 25 that TD misses, and TD catches 25 that Opus misses. Their union catches 138 of 160 (86%).
  - Running all of TD's top 25% on top of the model would cost what TD costs now. Instead, run_test.py adds TD's highest-ranked files that fit in 5% of the config's test time (`LLM_TD_HEDGE`).
  - On reverts that buys 6.3 points of recall (70.6% to 76.9%) for about 4 points of CPU and GPU test time on a neutral change.
  - Larger budgets help less per point of cost: 10% reaches 78.1% at 15.4% CPU, 15% reaches 82.5% at 19.3%, and 20% reaches 81.2% at 23.4%. 5% is the recommended start because many of the misses it leaves are label noise; see below.
- **Run-to-run variance is about 3 points.**
  - Two identical Opus low runs give 70.6% and 67.5% on reverts.
  - The same is true of every row in the table, so differences of 2 or 3 points between neighbors are not significant.
- **Results with all PR truth.**
  - Counting every attributable PR failure, not only reproduced ones (n=315 failing PRs): Opus low + hedge catches 87.0%, Opus high + hedge 88.3%, and TD 92.1%.
  - The unreproduced failures are dominated by flaky tests. TD has historical failure-rate heuristics that are good at predicting those, and running a known-flaky test on a PR is not a win.

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

## Agentic mode

- **Setup.** Sonnet 5 medium was rerun on 169 of the samples (80 PR, 60 revert, 29 neutral) with Read/Grep/Glob over a checkout of the change.
- **It made no difference.** Revert recall was identical at 55.0% caught, with and without tools. PR recall dropped from 94.1% to 92.2%.
- **The model rarely used the tools.** In about 63% of samples it answered without calling one.
- **It added cost and latency.** Per call it cost $0.150 vs $0.134 on PRs and $0.116 vs $0.095 on reverts, and p90 latency rose from 29s to 37s.
- **Opus without tools is better.** On the same subset, Opus 5.5 low with no tools reaches 100% PR and 71.7% revert caught.
- **Conclusion.** Spend on a bigger model, not on letting a smaller one explore. CI runs with `--tools ""`, which also removes any route for PR content to drive file reads.

## Dollars

**Sources.**
- The 30-day runner cost of each workflow's test jobs comes from ClickHouse job durations times runner price. The PR share is the PR share of test hours in the same data.
- The selectable part of each job is the Test step. Outside it, checkout, docker pull, build download and teardown take a median of 4.3 to 4.8 minutes, about 12-15% of the job. The model assumes a fixed overhead fraction of 0.2 for that.

**Old and new relative cost.**
- For TD workflows, today's relative cost is `0.2 + 0.8 * f_td`, with the replayed TD share on neutral commits (47.2% CPU, 41.8% GPU), so 0.58 and 0.53.
- For workflows without TD, today's relative cost is 1.
- The new relative cost is `0.2 + 0.8 * f_llm`, about 0.29 with f_llm = 11.4% CPU and 11.5% GPU.

**Inductor.**
- Only the inductor unit-test configs are selectable: 83% of its CPU test spend and 38% of its GPU test spend.
- The inductor numbers use only the 69 of 190 neutral commits that match the `ciflow/inductor` autolabel globs, since those are the PRs that run it. On those, the LLM runs 12.0% of CPU and 13.0% of GPU unit-test time.

**Sensitivity.** The overhead fraction dominates the uncertainty:

| Overhead | CPU saved | GPU saved | Total |
|---|---|---|---|
| 0.1 | $332k | $74k | $406k/mo |
| 0.2 | $271k | $60k | $331k/mo |
| 0.35 | $197k | $44k | $241k/mo |
| 0.5 | $138k | $31k | $169k/mo |

Other policies at overhead 0.2:

| Policy | CPU saved | GPU saved | Model cost |
|---|---|---|---|
| Opus 5.5 low + 5% hedge (recommended) | $271k | $60k | $2.0k/mo |
| Opus 5.5 low, no hedge | $299k | $67k | $2.0k/mo |
| Opus 5.5 high + 5% hedge | $270k | $59k | $2.3k/mo |
| Sonnet 5 medium + 5% hedge | $263k | $59k | $1.3k/mo |
| Haiku 4.5 low + 5% hedge | $276k | $61k | $0.7k/mo |

The no-hedge and Haiku rows save more only because they run less; they miss 6 and 17 more points of reverts respectively.

**Model cost.** Opus low averages $0.139 on neutral calls. At about 11.4k PR pushes a month to pull, that is about $2.0k a month.

**Where the GPU money is.**
- The GPU number is modest because the biggest GPU line, the inductor model benchmarks, is not selectable test files.
- The same approach can gate them: ask the model whether a change can affect compiled-model performance or accuracy at all, which is a yes/no question rather than a file list. That is step 5 of the rollout.
- Skipping shards whose selection is empty would also recover the job overhead the model above assumes is fixed.

## Caveats

- **The TD replay is optimistic.** It uses today's heuristic stats, which include these failures. PR truth is also censored toward TD's picks. Both favor TD, so the LLM's relative standing is if anything understated.
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
   - run_test.py logs `llm_td_shadow_selection` and `llm_td_shadow_failure`. The second metric is a live recall measurement: every failing test, with whether the selection covered it.
   - Gate the next step on a few weeks of it matching the numbers above.
2. **Opt-in enforcement.** `ci-llm-td` enforces the selection, with the 5% hedge, on that PR.
3. **Default enforcement.** Enforce on every internal PR, keeping the label as an opt-out (or `ci-no-td`). This unlocks about $285k a month and the merge-wait drop above.
4. **Inductor.** Have `inductor.yml` download the selection. It has no TD today, so its unit tests are the largest relative saving: about $47k a month.
5. **Benchmarks and shards.** Gate the inductor model benchmarks on a yes/no model answer, and skip shards with nothing selected.

## Reproducing

The reported PR and neutral samples come from a 2026-10-01 .. 10-26 window and the reverts from a wider 2026-04-01 one; build_dataset takes one window per run, so run it twice into the same `$DATA`. The numbers above were scored from `samples_v2/` and `td_v2/`, the outputs of rerunning build_dataset and td_replay after the revert-truth fix; a fresh run writes the default `samples/` and `td/`.

```
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-10-01 --end 2026-10-26
python -m tools.testing.llm_td.eval.run_eval --data $DATA --tag v2_opus55_low --model us.anthropic.claude-opus-5-5 --effort low --ids $DATA/ids_eval.txt
python -m tools.testing.llm_td.eval.td_replay --data $DATA
S="python -m tools.testing.llm_td.eval.score --data $DATA --ids $DATA/ids_eval_clean.txt"
$S --policies model,final,final+td5 --json rows.json v2_opus55_low td
$S --policies final+td5,final+td10,final+td15,final+td20 v2_opus55_low
$S --all-truth --policies final,final+td5 v2_opus55_low td
python -m tools.testing.llm_td.eval.dollars --data $DATA --overhead 0.2 v2_opus55_low final+td5
python -m tools.testing.llm_td.eval.chart rows.json tools/testing/llm_td/report_tradeoff.svg
```
