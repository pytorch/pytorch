# Test Target Determination (TTD) — Research & Improvement Roadmap

## Executive Summary

PyTorch runs 12+ hours of tests per PR. The existing TTD system uses 10
heuristics to score and reorder tests, running only the top 25%. Analysis of
13,920 commits over the past year identified several high-ROI improvements that
could reduce PR test time toward a 30-minute target.

**Key findings:**
- 80% of commits touch ≤5 files — most PRs have narrow blast radius
- Inductor/Dynamo account for 33% of all changes and were previously conflated in TTD
- Only 18% of commits touch C++ files, but these have the worst TTD coverage
- CI-only changes (10% of commits) get random test selection, not intelligent skipping
- `native_functions.yaml` changes (251/year) trigger broad test runs despite being highly parseable

---

## Current Architecture

### Heuristic System

Location: `tools/testing/target_determination/`

10 heuristics score tests from -1 to 1, scores are summed, and the top 25% run:

| # | Heuristic | Max Score | How It Works |
|---|-----------|-----------|-------------|
| 1 | PreviouslyFailedInPR | 1.0 | Tests that failed in prior CI runs of this PR |
| 2 | EditedByPR | 1.0 | Test files directly modified in the PR |
| 3 | MentionedInPR | 1.0 | Tests named in PR body/commits/linked issues |
| 4 | HistoricalClassFailureCorrelation | 0.25 | Test classes that historically fail when these files change (**trial_mode**) |
| 5 | CorrelatedWithHistoricalFailures | 0.25 | Test files that historically fail when these files change |
| 6 | HistoricalEditedFiles | 0.25 | Tests co-edited with changed files in past commits |
| 7 | Profiling | 0.25 | Python code coverage data mapping source → test files |
| 8 | LLM | 0.25 | CodeLlama-7b-Python RAG-based retrieval |
| 9 | Filepath | 0.25 | Keyword matching between changed file paths and test names |
| 10 | PublicBindings | 1.0 | Changes to `torch/` trigger `test_public_bindings` |

Binary heuristics (1-3, 10) dominate at score 1.0. Correlation-based heuristics
(4-9) are all capped at 0.25, meaning they only affect ordering among tests not
directly edited or previously failed.

### Data Pipeline

```
ClickHouse (CI execution history)
    → test-infra daily cron jobs (update_test_file_ratings.yml, update-test-times.yml)
    → generated-stats branch (JSON files)
    → PyTorch CI downloads at test time
    → tools/testing/target_determination/determinator.py runs heuristics
    → td_results.json uploaded to S3
    → test/run_test.py reads results, runs top 25%
```

Key data files on `generated-stats` branch:
- `test-times.json` — Per-file test durations
- `test-class-times.json` — Per-class test durations
- `file_test_rating.json` — Source file → test file failure correlations
- `file_test_class_rating.json` — Source file → test class failure correlations
- `td_heuristic_historical_edited_files.json` — Co-change correlations
- `td_heuristic_profiling.json` — Python coverage-based correlations

### Legacy System (Disabled)

`tools/testing/modulefinder_determinator.py` used Python's `modulefinder` to
trace imports from test files. It was disabled (`run_test.py:1467`) because
`import torch` eagerly loads ~30 subpackages, making every test appear to depend
on everything.

---

## Test Suite Analysis

- **13,920 commits/year** (~38/day)
- **72,478 file changes** across those commits
- **~1,327 test files** across distributed (304), dynamo (158), inductor (156)
- **Average 5.2 files per commit** — 80% touch ≤5 files

### Commit Classification

| Category | Commits/Year | % |
|----------|-------------|---|
| Source + Test together | 5,549 | 40% |
| Source only (no tests) | 4,882 | 35% |
| CI only | 1,347 | 10% |
| Test only | 1,191 | 9% |
| Tools only | 176 | 1% |
| Other | 774 | 6% |

### Language Distribution

| Language | File Changes | % |
|----------|-------------|---|
| Python | 38,964 | 54% |
| C++ | 7,491 | 10% |
| C++ Headers | 4,558 | 6% |
| CUDA | 1,002 | 1.4% |
| HIP/ROCm | 1,863 | 2.6% |
| Other | 18,600 | 25% |

### Top Changed Source Areas

| Area | File Changes/Year |
|------|------------------|
| torch/_inductor | 7,286 |
| torch/_dynamo | 4,437 |
| torch/distributed + torch/csrc/distributed | 4,567 |
| .github + .ci (CI config) | 4,328 |
| aten/native (CPU + CUDA kernels) | 2,427 |
| torch/csrc (other C++) | 2,471 |

---

## Known Gaps

### 1. Hub Imports (`import torch`)

`torch/__init__.py` (~2,994 lines) eagerly imports ~30 subpackages. Only 4 are
lazy (`_dynamo`, `_inductor`, `_export`, `onnx`). This means import-based
dependency analysis (modulefinder) sees every test depending on everything.

**Impact**: The modulefinder_determinator was disabled entirely because of this.

**Potential fix**: Making more subpackages lazy (especially `distributed` with
2,132 commits/year) would help, but only if paired with a new differential
import tracker. The current heuristic-based system doesn't use import analysis.

### 2. C++ Tracing

~4,710 C++ files compile into two monolithic libraries (`torch_cpu.so`,
`torch_python.so`). The current TTD classifies C++ changes with only basic
keyword matching. The profiling heuristic is explicitly Python-only.

**Impact**: ~18% of commits touch C++. C++ changes to `c10/core/` or
`aten/native/` get no meaningful test scoping.

### 3. Heuristic Weight Tuning

All correlation-based heuristics are hard-coded to max 0.25. Weights are not
learned from data. The trial_mode infrastructure exists for testing new
heuristics but isn't used for weight optimization.

---

## Improvement Roadmap (ROI-Ranked)

### P0: Immediate Impact (Hours of work)

#### Fix Inductor/Dynamo/Export Synonym Conflation
- **File**: `filepath.py:34`
- **Problem**: `"inductor": ["dynamo", "export"]` treats all three as synonyms
- **Impact**: 33% of commits (4,600/year) over-select tests
- **Fix**: Split into independent keywords
- **Status**: Implemented

#### Filter CI/Docs Keywords from Filepath Heuristic
- **Problem**: CI path components ("github", "workflows") extracted as keywords
  but never match test names; CI-only PRs get random 25% selection
- **Impact**: 10% of commits (1,347/year)
- **Fix**: Add CI/docs path components to `not_keyword` list
- **Status**: Implemented

### P1: Short-term (Days of work)

#### Move Multi-Dtype Comprehensive Tests to Nightly
- **Problem**: Comprehensive decomp/inductor tests across float16/32/64 take
  10-20+ min each
- **Impact**: ~185 GPU-minutes per PR saved
- **Fix**: Add `@slowTest` to multi-dtype comprehensive tests, keep float32 in
  per-PR CI
- **Risk**: Low — float32 catches ~80% of dtype regressions

#### C++ c10d Zone Mapping
- **Problem**: c10d changes (932/year) fan out to 248 distributed tests
- **Impact**: Could narrow to 3-5 tests per change
- **Fix**: Static mapping: ProcessGroupNCCL.cpp → test_c10d_nccl.py, etc.
- **Effort**: 1-2 days for static mapping table

### P2: Medium-term (1-2 weeks each)

#### native_functions.yaml Op-Level Diff Parsing
- **Problem**: 265 changes/year, 74% are dispatch-add (easiest to narrow)
- **Impact**: 80-95% reduction in OpInfo test cases per dispatch-add commit
- **Fix**: Parse YAML diffs to extract changed op names → map to OpInfo tests
- **Effort**: Leverage existing `torchgen` parser

#### Learn Heuristic Weights via ML
- **Problem**: Hard-coded equal weights, no feature interactions
- **Impact**: 5-15% recall improvement at same 25% budget
- **Fix**: Train logistic regression/XGBoost on existing heuristic outputs
- **Effort**: trial_mode infrastructure already exists

### P3: Strategic (Weeks)

#### Full ML Prediction Model
- 10-20% improvement over current ensemble (based on Google PTS results)
- Requires training pipeline using ClickHouse data

#### Hub Import Lazy Loading + Differential Import Tracker
- Near-zero TTD benefit without building a new import tracker
- `distributed` (2,132 commits/year) is the biggest lazy candidate

---

## Key Files

| File | Purpose |
|------|---------|
| `tools/testing/target_determination/determinator.py` | Main TD orchestrator |
| `tools/testing/target_determination/heuristics/__init__.py` | Heuristic registry |
| `tools/testing/target_determination/heuristics/interface.py` | TestPrioritizations, AggregatedHeuristics |
| `tools/testing/target_determination/heuristics/filepath.py` | Keyword-based path matching |
| `tools/testing/target_determination/heuristics/utils.py` | Shared utilities |
| `tools/testing/do_target_determination_for_s3.py` | CI entry point |
| `tools/testing/test_selections.py` | Test sharding logic |
| `tools/testing/discover_tests.py` | Test discovery |
| `tools/testing/modulefinder_determinator.py` | Legacy import-based TD (DISABLED) |
| `test/run_test.py` | Main test runner with TD integration |
| `aten/src/ATen/native/native_functions.yaml` | Operator definitions (2,040 dispatch entries) |
| `torch/testing/_internal/common_methods_invocations.py` | OpInfo database (721 entries) |
