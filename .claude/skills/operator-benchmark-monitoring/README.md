# Operator Benchmark Monitoring Skill

Monitors PyTorch operator microbenchmark daily runs for performance regressions and alerts the team via GitHub issues.

## Overview

This skill automatically monitors the daily scheduled runs of `operator_microbenchmark` workflow in `pytorch/pytorch`, detects performance regressions, and creates issues to alert `@jainapurva` and the team.

## How It Works

```
Daily at 08:00 UTC (after operator_microbenchmark runs at 06:00 UTC)
    ↓
[claude-operator-benchmark-monitoring.yml] triggers
    ↓
Job 1: monitor (Claude analysis, READ-ONLY)
    ├─ Fetch recent operator_microbenchmark runs from pytorch/pytorch
    ├─ Download benchmark result artifacts
    ├─ Load baseline data from CSV files in pytorch/pytorch
    ├─ Compare performance: current vs baseline
    ├─ Detect regressions (>20% = high, >30% = critical)
    ├─ Check existing perf-regression issues
    └─ Produce /tmp/benchmark-regression-actions.json
    ↓
Job 2: apply-actions (WRITE permissions)
    ├─ Parse actions JSON
    ├─ Create new issues for new regressions
    ├─ Update existing issues if severity changed
    ├─ Close issues if regressions resolved
    └─ Always assign to @jainapurva
```

## Regression Thresholds

| Change | Severity | Action |
|--------|----------|--------|
| >30% slower | **Critical** | Create issue immediately |
| 20-30% slower | **High** | Create issue |
| 10-20% slower | **Medium** | Create issue if confirmed across 2+ runs |
| 5-10% slower | **Low** | Note for monitoring |
| <5% | - | Noise, ignore |

## Context-Aware Analysis

**Operator importance:**
- Core operators (matmul, conv, linear, attention): Stricter thresholds
  - Critical if >20% slower
- Common operators (activations, normalizations): Standard thresholds
- Specialized operators (quantized, sparse): Higher tolerance

**Benchmark duration:**
- Micro-benchmarks (<100μs): 15% threshold (more noise)
- Mid-range (100μs-10ms): 10% threshold
- Macro-benchmarks (>10ms): 5% threshold (stable)

## Issue Format

Issues created in `pytorch/pytorch`:

**Title:** `[Operator Benchmark] Performance regression in <operator>`

**Labels:**
- `perf-regression` (always)
- `operator-benchmark` (always)
- Category label: `module: linear algebra`, `module: conv`, etc.
- Priority: `high priority` (if critical)

**Assignees:** `jainapurva` (always)

**Body (summary):**
- Brief description
- Severity level
- Affected devices
- Baseline vs current performance
- Percentage change
- First detected date

**First Comment (details):**
- Full regression table
- Timeline
- Potential causes (recent commits)
- Impact assessment
- Recommendations
- Link to dashboard

## Files

```
.claude/skills/operator-benchmark-monitoring/
├── SKILL.md              # Main skill instructions
└── README.md             # This file

.claude/hooks/operator-benchmark-monitoring/
├── restrict-write.sh          # Pre-hook: restrict Write tool
├── validate-post-write.sh     # Post-hook: validate JSON structure
└── validate-on-stop.sh        # Stop hook: final validation

.github/workflows/
└── claude-operator-benchmark-monitoring.yml  # Workflow (daily + manual)
```

## Testing

### Manual Trigger

```bash
gh workflow run claude-operator-benchmark-monitoring.yml \
  --repo pytorch/pytorch
```

### Check Recent Runs

```bash
gh run list --workflow claude-operator-benchmark-monitoring.yml \
  --repo pytorch/pytorch \
  --limit 5
```

### View Logs

```bash
gh run view RUN_ID --log --repo pytorch/pytorch
```

## Disabling

To disable the daily monitoring:

```bash
gh workflow disable claude-operator-benchmark-monitoring.yml \
  --repo pytorch/pytorch
```

Or comment out the `schedule` trigger in the workflow file.

## Security Model

**Read-only Claude job:**
- GitHub token for pytorch/pytorch is read-only
- AWS credentials are read-only (can download artifacts from S3)
- Can only write to `/tmp/benchmark-regression-actions.json`

**Separate write job:**
- Validates actions JSON
- Creates/updates issues in pytorch/pytorch
- Always assigns to @jainapurva

**Validation hooks:**
- Ensure output conforms to schema
- Block writes to unauthorized files
- Validate repo is pytorch/pytorch

## Future Enhancements

- [ ] Send Gchat notifications (in addition to GitHub issues)
- [ ] Automatic bisection to find first-bad commit
- [ ] Integration with performance dashboard
- [ ] Trend analysis (gradual degradation over time)
- [ ] Automatic revert PR creation for critical regressions

## References

- **Operator microbenchmark workflow:** `pytorch/pytorch/.github/workflows/operator_microbenchmark.yml`
- **Baseline data:** `pytorch/pytorch/benchmarks/operator_benchmark/*_expected_*.csv`
- **Dashboard:** https://hud.pytorch.org/benchmark/v3/dashboard/pytorch_operator_microbenchmark
- **Pattern:** Based on `.claude/skills/treehugger/`
