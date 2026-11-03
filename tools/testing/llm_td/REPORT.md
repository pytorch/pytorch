# LLM test selection for PyTorch CI: offline evaluation

**Summary.**
- **The idea.** One `claude -p` call per PR sees the title, description, diff and every candidate test file annotated with its CI minutes, and returns the files worth running.
- **Accuracy.** On 666 held-out historical changes, Opus 5.5 at low effort with a small TD hedge catches 97.3% of PRs with a reproduced failure and 75.0% of the commits autorevert later reverted. TD today catches 97.3% and 70.6%.
- **Test time.** It runs 11% of CPU and 12% of GPU test time on a typical change, against 45% and 44% for TD's top 25% cut.
- **Savings.** At the 30-day runner spend of the workflows run_test.py drives, the dollar model puts this at about **$320k a month: $261k CPU and $59k GPU**. Model calls cost about $2k a month.
- **Range.** Depending on how much of each job is setup rather than tests, the estimate ranges from $160k to $400k a month.
- **The stack.** The selector, harness, run_test.py integration and a shadow-mode CI job come in the PRs below this one. Enforcement is opt-in per PR through the `ci-llm-td` label. Main always runs everything.

![caught vs GPU share](report_tradeoff.svg)

Each point is one model and effort level. Filled points are the selection as the model returns it, after the safety rules. Hollow points add the 5% TD hedge. The red square is the replayed TD at its CI cut. The x axis is the GPU share of the sample's own CI pass, which is higher than on a typical change because failing changes are larger.

## Method

**Unit of selection.**
- The unit is a test file under `test/`. Selecting a file runs it in every config it belongs to: default, distributed, slow, inductor and so on.
- The two wrapper sweeps, `dynamo_wrapped` and `crossref`, re-run hundreds of ordinary files in a special mode. The model requests them separately; an unrequested sweep runs only the selected files.
- The model can also request whole directories (`dynamo`, `inductor`, ...) or `run_all`.

**Safety rules (the "final" policy).** `Selection.from_model_output` applies these on top of the model's answer:
- A test file the PR edits or adds is always selected.
- The answer becomes run-all when it is missing or malformed, or when the change touches anything that decides how tests run. That covers run_test.py, conftest, common_utils, common_device_type, tools/testing, the test shell scripts and actions, and the CI docker images.

**Cost.**
- Cost is the share of one full CI pass's test-seconds that a selection runs, from `test-times.json`.
- Seconds are summed over every build environment. Environments matching `cuda|rocm|xpu|sm*|h100|b200|a100|gpu` count as GPU.

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
- `eval/td_replay.py` runs today's in-repo TD heuristics on each sample's changed files and commit text, then takes the top 25% cut CI uses.
- The heuristics read today's historical-stats artifacts, which already contain these failures. The replay is therefore optimistic for TD.

## Results

Caught means the share of changes with at least one failure where the policy caught at least one failure; "pairs" means recall over (file, config) failures. CPU/GPU are mean shares of a full pass. PR rows use strong truth.

| Model / effort | PR caught | PR pairs | Revert caught | Revert pairs | Neutral CPU | Neutral GPU | $/call | p50 wall |
|---|---|---|---|---|---|---|---|---|
| Haiku 4.5 low | 84.9% | 75.9% | 45.0% | 34.8% | 6.9% | 6.8% | $0.064 | 32s |
| Haiku 4.5 high | 86.5% | 77.6% | 46.2% | 35.9% | 6.9% | 6.8% | $0.064 | 32s |
| Sonnet 5 low | 95.7% | 91.8% | 54.4% | 43.2% | 8.7% | 8.1% | $0.102 | 5s |
| Sonnet 5 medium | 95.1% | 91.8% | 58.1% | 43.5% | 8.6% | 8.2% | $0.110 | 11s |
| Sonnet 5 high | 95.1% | 92.8% | 63.1% | 51.6% | 10.1% | 9.4% | $0.127 | 23s |
| Opus 5.5 low | 97.3% | 95.3% | 70.6% | 55.1% | 7.5% | 7.3% | $0.179 | 8s |
| Opus 5.5 low (rerun) | 98.4% | 96.0% | 67.5% | 52.2% | 7.8% | 7.5% | $0.178 | 8s |
| Opus 5.5 medium | 98.4% | 96.5% | 70.0% | 55.1% | 7.6% | 7.5% | $0.192 | 14s |
| Opus 5.5 high | 98.4% | 97.0% | 72.5% | 58.0% | 7.7% | 7.7% | $0.201 | 18s |
| **Opus 5.5 low + 5% TD hedge** | **97.3%** | 95.8% | **75.0%** | 58.8% | **10.9%** | **11.7%** | $0.179 | 8s |
| Opus 5.5 high + 5% TD hedge | 98.4% | 97.3% | 76.2% | 60.9% | 11.2% | 12.2% | $0.201 | 18s |
| TD top 25% (replayed) | 97.3% | 97.5% | 70.6% | 58.6% | 45.0% | 44.3% | - | - |

**What the table shows.**

- **Model size matters more than effort.** Opus at low effort beats Sonnet at high effort on reverts (70.6% vs 63.1%) at a similar cost, and is faster.
  - Effort helps Opus by about two points on reverts, which is within run-to-run noise.
  - Haiku 4.5 does not support the effort setting, so its two runs are effectively replicates. It is not good enough on reverts.
  - Sonnet sometimes fails to produce schema-valid output after retries: 12 of 666 samples at low effort. Those fall back to run-all, which is safe but costs.
- **Spend goes almost entirely on run-all.**
  - Without the safety rules, the model alone runs about 3.4% of test time on a neutral change.
  - The run-all rules fire on 10 of 190 neutral commits, mostly edits to `.ci/pytorch/test.sh` and the test docker images, and account for the rest of the 7.5%.
  - Narrowing those rules is the cheapest remaining lever. They were kept broad because 8 of the 160 reverts are docker image changes.
- **The LLM and TD miss different things.**
  - On reverts, Opus low catches 28 that TD misses, and TD catches 28 that Opus misses. Their union catches 88%.
  - Running all of TD's top 25% on top of the model would cost what TD costs now. Instead, run_test.py adds TD's highest-ranked files that fit in 5% of the shard's test time (`LLM_TD_HEDGE`).
  - On reverts that buys 4.4 points of recall (70.6% to 75.0%) for 3.4 points of CPU and 4.4 points of GPU test time on a neutral change.
  - Larger budgets keep paying: 10% reaches 76.9%, 15% reaches 77.5% and 20% reaches 80.0%. 5% is the recommended start because the misses it leaves are mostly not real; see below.
- **Run-to-run variance is about 3 points.**
  - Two identical Opus low runs disagree on 9 of 160 reverts: 70.6% vs 67.5%.
  - The same is true of every row in the table, so differences of 2 or 3 points between neighbors are not significant.
- **Results with all PR truth.**
  - Counting every attributable PR failure, not only reproduced ones: Opus low + hedge catches 85.1%, Opus high + hedge 86.7%, and TD 89.5%.
  - The unreproduced failures are dominated by flaky tests. TD has historical failure-rate heuristics that are good at predicting those, and running a known-flaky test on a PR is not a win.

**Revert misses.** I hand-checked the 39 revert samples that Opus high missed.
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
- **It made no difference.** Revert recall was identical: 55.0% caught and 48.3% caught-all, with and without tools. PR recall dropped from 94.1% to 92.2%, and the selection was 1 to 2 points smaller.
- **The model rarely used the tools.** In about 63% of samples it answered without calling one; 2 turns is the structured answer alone.
- **It added cost and latency.** It cost 13% more per call ($0.126 vs $0.112) and raised p90 latency from 29s to 37s.
- **Opus without tools is better.** On the same subset, Opus 5.5 low with no tools reaches 100% PR and 71.7% revert caught.
- **Conclusion.** Spend on a bigger model, not on letting a smaller one explore. CI runs with `--tools ""`, which also removes any route for PR content to drive file reads.

## Dollars

**Sources.**
- The 30-day runner cost of each workflow comes from ClickHouse job durations times runner price. The PR share of that cost is taken from the same data.
- The selectable part of each job is the Test step. Outside it, checkout, docker pull, build download and teardown take a median of 4.3 to 4.8 minutes, about 12-15% of the job. The model assumes a fixed overhead fraction of 0.2 for that.

**Old and new relative cost.**
- For TD workflows, today's relative cost is `0.2 + 0.8 * f_td`, with the replayed TD share on neutral commits.
- For workflows without TD, today's relative cost is 1.
- The new relative cost is `0.2 + 0.8 * f_llm`.

**Inductor.**
- Only the inductor unit-test configs are selectable. The model benchmarks (huggingface, timm, torchbench) are 52.5k of the 85k GPU hours in that workflow, and run_test.py does not drive them.
- The inductor numbers use only the 69 of 190 neutral commits that match the `ciflow/inductor` autolabel globs, since those are the PRs that run it.

Recommended policy (Opus 5.5 low + 5% hedge), overhead 0.2:

| Workflow | Runner | PR cost $k/mo | TD today | LLM | Saved $k/mo |
|---|---|---|---|---|---|
| pull | CPU | 375.5 | 0.55 | 0.29 | 178.2 |
| trunk | CPU | 98.9 | 0.55 | 0.29 | 46.9 |
| trunk | GPU | 99.5 | 0.53 | 0.29 | 44.8 |
| inductor unit tests | CPU | 51.1 | 1.00 | 0.31 | 35.5 |
| inductor unit tests | GPU | 16.2 | 1.00 | 0.32 | 11.0 |
| periodic | GPU | 5.3 | 1.00 | 0.29 | 3.7 |
| slow | CPU | 0.4 | 1.00 | 0.29 | 0.3 |
| **total** | | **647** | | | **320 (261 CPU, 59 GPU)** |

**Sensitivity.** The overhead fraction dominates the uncertainty:

| Overhead | CPU saved | GPU saved | Total |
|---|---|---|---|
| 0.1 | $323k | $73k | $396k/mo |
| 0.2 | $261k | $59k | $320k/mo |
| 0.35 | $187k | $43k | $230k/mo |
| 0.5 | $130k | $30k | $160k/mo |

Two more bounds:
- Using the selection shares of the failing PR samples (22.6% CPU, 20.0% GPU), which are much larger changes than typical, gives $224k CPU and $54k GPU.
- Without the hedge, Opus low saves $354k, but misses more reverts.

**Model cost.**
- Opus low averages $0.179 per call over all samples and $0.139 on neutral ones. At about 11.4k PR pushes a month to pull, that is about $2.1k a month.
- Sonnet 5 medium would be $1.3k, but with the hedge it saves about $9k a month less and its revert recall is 9 points lower.

**Where the GPU money is.**
- The GPU number is modest because the biggest GPU line, the inductor model benchmarks, is not selectable test files.
- The same approach can gate them: ask the model whether a change can affect compiled-model performance or accuracy at all, which is a yes/no question rather than a file list. That is phase 3 below.
- Skipping shards whose selection is empty would also recover the job overhead the model above assumes is fixed.

## Caveats

- **The TD replay is optimistic.** It uses today's heuristic stats, which include these failures. PR truth is also censored toward TD's picks. Both favor TD, so the LLM's relative standing is if anything understated.
- **Label noise.** About a quarter of the remaining revert misses look unrelated to the reverted change, so true revert recall is higher than reported.
- **Training-data contamination.** Reverts before mid-2026 may predate the models' knowledge cutoff. The PR samples (October 2026) do not, and they show the same ordering of models.
- **Prompt injection.**
  - A PR's description or diff can try to talk the model into selecting nothing.
  - Mitigations: enforcement is opt-in per PR through the `ci-llm-td` label; the selector job runs only on pytorch/pytorch branches, from a trusted main checkout, and reads the PR only through git; main always runs everything.
  - The worst case is the same as a TD miss today: a regression is caught on main and reverted.
- **Deployment noise.** The CI job uses the existing `bedrock` environment. Its behavior there (latency, throttling) is not measured here. The shadow metrics are what will measure it.

## Rollout

1. **Shadow (this stack).**
   - The `llm-test-selection` job writes a selection for every internal PR, and run_test.py logs `llm_td_shadow_selection` and `llm_td_shadow_failure`.
   - The second metric is a live recall measurement: every failing test, with whether the selection covered it.
   - Gate the next step on a few weeks of it matching the numbers above.
2. **Opt-in enforcement.** `ci-llm-td` enforces the selection, with the 5% hedge, on that PR.
3. **Default enforcement.** Enforce on every internal PR, keeping the label as an opt-out (or `ci-no-td`).
4. **Inductor.** Wire `target_determination.yml` into `inductor.yml`. It has no TD today, so its unit tests are the largest relative saving.
5. **Benchmarks and shards.** Gate the inductor model benchmarks on a yes/no model answer, and skip shards with nothing selected.

## Reproducing

```
python -m tools.testing.llm_td.eval.build_dataset --data $DATA --start 2026-09-15 --end 2026-11-01
python -m tools.testing.llm_td.eval.run_eval --data $DATA --tag v2_opus55_low --model us.anthropic.claude-opus-5-5 --effort low --ids $DATA/ids_eval.txt
python -m tools.testing.llm_td.eval.td_replay --data $DATA
python -m tools.testing.llm_td.eval.score --data $DATA --ids $DATA/ids_eval_clean.txt --policies model,final,final+td5 v2_opus55_low td
```
