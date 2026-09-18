---
name: test-cost
description: Generate an HTML report of CI compute used by pytorch/pytorch test files over the last 14 UTC days - machine-hours per test file and per owner (the "# Owner(s)" header), split by hardware class (CPU x86/arm64, Windows, macOS, T4, A10G, L4, A100, H100, B200, ROCm, XPU, TPU, s390x) and by GPU sub-class (model and GPUs per runner, e.g. MI300 x4, A10G x1) with GPU-hours - from the CI ClickHouse workflow_job and all_test_runs tables with a GitHub Actions API spot-check, uploaded to Pixelcloud. Use when asked which tests, files, owners or modules burn the most CI time, about test cost by hardware or owner, GPU test spend, or for a test-cost report.
---

# Test cost

One script measures the wall-clock machine-hours of pytorch/pytorch `test (...)` and legacy `test-osdc (...)` jobs, attributes successful jobs to their test files and owners, and writes one self-contained HTML page. Run the script; do not hand-write ClickHouse SQL for this question, and never retype or round its numbers by hand.

## What it measures

- Test-job hours: `completed_at - started_at` of every completed `... / test (...)` or `... / test-osdc (...)` job in `default.workflow_job` (the archived GitHub Actions webhook payloads), deduplicated by job id and bucketed by completion day in UTC. All conclusions count, including failure, cancellation and timeout. Queue time is excluded; a rerun is a separate job. The trigger (main, pr, scheduled, other) comes from `default.workflow_run`.
- Attributed hours: each successful job's wall time is split across the files it ran in proportion to their per-test seconds in `tests.all_test_runs`, using the latest ingestion snapshot of each source report and preserving repeated testcases and separate rerun reports. Non-successful jobs remain entirely unattributed because interrupted files may never upload a report; this deliberately also excludes failed jobs with complete reports. Attributed hours never exceed test-job hours. File job counts and raw test seconds cover only successful jobs; job IDs are deduplicated after resolving invoking-file aliases. The unattributed remainder also includes jobs without per-test results (XPU, s390x, TSan, torchtitan, most perf jobs), upload lag and ingestion gaps. Success is a conservative eligibility rule, not proof that every report arrived. Check coverage and data-quality warnings rather than assuming a fixed attributable share.
- Owner: the first label of the file's `# Owner(s): [...]` header in the current checkout with the `module: ` / `oncall: ` prefix removed (`test/dynamo/test_deviceguard.py` -> `dynamo`). `unknown` is the literal `module: unknown` label (test_ops.py and friends), `no-header` marks files without the header, and `unmapped` marks invoking names with no file in the checkout (C++ gtest launchers such as `test_libtorch`, or files that only exist on a PR branch).
- Hardware class: derived from the runner label (`mt-l-x86aavx2-29-113-a10g` -> A10G). Accelerator classes are further split into sub-classes, the GPU model plus GPUs per runner parsed from the label (`linux.rocm.gpu.mi300.4` -> ROCm, `MI300 x4`; `mt-l-x86iavx512-45-172-t4-4` -> T4, `T4 x4`; labels without a count are single-GPU, except legacy EC2 sizes such as `g5.48xlarge` that map to their GPU count), and GPU-hours multiply runner hours by GPUs per runner. Hours are reported per class and sub-class and are not converted to dollars: an H100 hour costs far more than a CPU hour, and donated hardware (ROCm, B200, XPU, TPU, s390x) has no price in the CI cost tables. Do not add USD figures.
- GitHub cross-check: a deterministic sample of jobs (default 20) is fetched from the GitHub Actions API with `gh api` and compared on runner label and wall seconds; the result is in the report and in the stdout summary.

## Requirements

Run from the repo root with `python3` (standard library only). ClickHouse is reached through [.claude/skills/ci-metrics/gcx-wrapper.sh](../ci-metrics/gcx-wrapper.sh), so `gh` must be authenticated (`gh auth login --hostname github.com --git-protocol ssh --web`) and `curl` must be on PATH. `px` is optional; when it is installed the page is uploaded to Pixelcloud.

## Run

```bash
python3 .claude/skills/test-cost/scripts/test_cost.py --self-test   # offline checks, a few seconds
python3 .claude/skills/test-cost/scripts/test_cost.py               # last 14 full UTC days, uploads
python3 .claude/skills/test-cost/scripts/test_cost.py --days 7 --end 2026-09-15 --no-upload
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--days N` | 14 | number of full UTC days |
| `--end YYYY-MM-DD` | today (UTC) | exclusive end; pin it to make a report reproducible |
| `--out FILE` | `agent_space/test-cost/<start>_<last>.html` | output file |
| `--verify-sample N` | 20 | jobs cross-checked against the GitHub API; 0 disables |
| `--no-upload` | | skip the Pixelcloud upload |
| `--no-cache` | | refetch every day and write nothing to the cache |

A cold run issues two ClickHouse queries per day, plus one sample query when `--verify-sample` is on (2-3 min per day, so roughly 30-40 min for 14 days). Days at least 3 days old are cached under `agent_space/test-cost/cache/<query hash>/`, so later runs take a few minutes because only the youngest days are refetched; delete the cache directory to force a refresh. Progress goes to stderr. The stdout summary ends with `report: <path>` and `uploaded: <url | skipped (...) | failed (...)>`.

## Summarize the result

Answer from the stdout summary and point at the report for the full tables (each table row and chart mark is on its own line, so `grep 'test_ops.py' <report>.html` returns that file's rows instead of the whole page). Cover, in this order:

1. The window, the total test-job hours and the attributed share (coverage). Explain that non-successful jobs and jobs without per-test results remain unattributed; upload lag or ingestion gaps can also affect coverage.
2. The top owners with hours and share, calling out accelerator hours (NVIDIA classes, ROCm, XPU, TPU) explicitly because the totals are not price-weighted. Give the sub-class split from the stdout summary (runner hours per GPU model and count under each accelerator class) and its total GPU-hours line; per-sub-class GPU-hours are in the report's By sub-class table, and the Owners by sub-class table covers any owner the user asks about. In the page, the Owners and Files tables carry a Mix column (a mini stacked bar of the row's hardware-family split in the owner chart's colours) that sorts by accelerator share, and the masthead sparkline shows test-job hours per day. In the three stacked charts, clicking a legend entry re-sorts the bars by that series (and re-selects the top rows for it) with the series drawn first in every bar; the Total pill restores the default.
3. The top files, and the per-owner file list for any owner the user asked about.
4. Anything notable in unmapped, no-header or the `unknown` hardware class, plus any warnings the script printed.
5. The GitHub cross-check result and the exact command that produced the report (both are also in the report's Methodology section).
6. The Pixelcloud URL from the `uploaded:` line, alone on the final line of the reply. If that line is not a URL, give the local report path instead and say why the upload did not happen.

## Rules

- Only this script produces the numbers. For a new slice or dimension, change the script and its self-test so the result stays reproducible; do not answer with ad-hoc SQL and never edit a generated page.
- If a number looks wrong, rerun with `--no-cache` or a larger `--verify-sample` and report what you see.
- Reports and the cache live in `agent_space/` (git-ignored) and are never committed. The skill files are tracked; commit them only when asked.

## Maintenance

- `scripts/cost_data.py` holds the classifier (`HW_PATTERNS`: first match wins, GPU tokens before OS and CPU tokens, and an accelerator guard that sends unrecognised GPU fleets to `unknown` instead of CPU; `MODEL_PATTERNS` and the GPU-count regexes turn an accelerator label into its sub-class), the invoking-file aliases and the owner parser. `--self-test` in `scripts/test_cost.py` pins every runner label seen in CI to its class, sub-class and GPU count; add a label there when a new fleet appears. The report's runner-label table shows the mapping in use and stderr warns when `unknown` exceeds 1% of hours.
- The SQL lives in `scripts/test_cost.py`; the cache directory is keyed by a hash of it, so changing a query invalidates old caches automatically. `scripts/cost_html.py` only renders.
- Run `python3 .claude/skills/test-cost/scripts/test_cost_tests.py` in the repository's test environment for regression checks. Set `TEST_COST_CLICKHOUSE=1` to also execute the SQL against synthetic fixtures in ClickHouse; these checks are read-only.
