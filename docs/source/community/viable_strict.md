---
orphan: true
---

# viable/strict CI Jobs
## Background
The [*viable/strict*](https://github.com/pytorch/pytorch/tree/viable/strict) branch ensures all essential PyTorch tests are passing on it. This provides a stable base for contributions and greatly increases confidence that test failures are newly introduced. *viable/strict* should closely track *main*, providing a topline read on trunk health.

All merged commits on *main* are candidates for becoming the new *viable/strict* head. Tests that must pass for this promotion to succeed are part of the *viable/strict*-blocking set. These tests must be high signal and minimally flaky. No subset of these tests should dominate advancement failures.

## Blocking Set Requirements
The *viable/strict*-blocking set is drawn from the [*trunk*](https://github.com/pytorch/pytorch/blob/main/.github/workflows/trunk.yml), [*pull*](https://github.com/pytorch/pytorch/blob/main/.github/workflows/pull.yml), [*lint*](https://github.com/pytorch/pytorch/blob/main/.github/workflows/lint.yml), and [*docs-build*](https://github.com/pytorch/pytorch/blob/main/.github/workflows/docs-build.yml) workflows. Jobs that are part of these workflows should meet the following requirements:

Let **Failures %** (given a job, over an N-day window) be defined as the percent of commits where this job was red, over the decided (non-pending) *main* commits in the provided window.
1. A job should only be added to the *viable/strict*-blocking set if it meets **Failures % less than 10% over the last 7 days**, viewable on the HUD [*reliability* page](https://hud.pytorch.org/reliability), or computed manually on [the trunk dashboard](https://hud.pytorch.org/hud/pytorch/pytorch/main).
    - Latency expectations:
        - *pull runtime*: **Jobs added to the frequently-run *pull* workflow should not cause *pull* P50 runtime to exceed 75 minutes.** Current *pull* runtime can be viewed on the HUD [*metrics: p50 pull TTS* panel](https://hud.pytorch.org/metrics). For a specific PR, the Gantt chart view at *https://hud.pytorch.org/pytorch/pytorch/commit/<pr_commit_hash>* can be used to analyze the runtime effect of newly added jobs.
        - *trunk runtime*: **Jobs added to the *trunk* workflow should not cause *trunk* P50 runtime to exceed 270 minutes; added test jobs should not exceed 200 minutes in P50 individual runtime.** Current *trunk* runtime can be viewed on the HUD [*metrics: p50 pull, trunk, docs-build TTS* panel](https://hud.pytorch.org/metrics). The Gantt chart view can also be used to examine trunk shard runtimes.
    - The job owner is responsible for ensuring these requirements are met.
2. *viable/strict*-blocking jobs that exceed **Failures % greater than 15% over the last 7 days should be removed from the viable/strict-blocking set**, outside of sustained breakages across the *viable/strict*-blocking set. Qualifying events include GitHub SEVs and *viable/strict* lag greater than 24 hours, typically coinciding with systemic outages. Jobs should also not be demoted due to transient runner shortages that affect queue times.
    - Latency expectations:
        - *pull runtime*: **Jobs that cause the *pull* workflow P50 runtime to exceed 75 minutes** should be removed from the viable/strict-blocking set. This regression will be actioned upon if consistent for more than 7 days.
        - *trunk runtime*: **Jobs that cause the *trunk* workflow P50 runtime to exceed 270 minutes, or individual test jobs that exceed 240 minutes in P50 individual runtime,** should be removed from the viable/strict-blocking set. This regression will be actioned upon if consistent for more than 7 days.
    - The PyTorch Dev Infra team will assist with cordoning regressing jobs.

### Unstable Jobs and Introducing New Jobs
Jobs can be designated *unstable* as a staging step to determine reliability prior to (re)inclusion in the *viable/strict*-blocking set. New jobs targeting inclusion in the *viable/strict*-blocking set should be introduced in the [*unstable.yml*](https://github.com/pytorch/pytorch/blob/main/.github/workflows/unstable.yml) workflow and monitored. *Viable/strict*-blocking jobs can also be relocated to this workflow if more involved fixes are necessary.

A job can be marked *unstable* by opening a GitHub issue titled *UNSTABLE &lt;job name&gt;*, e.g. *UNSTABLE pull / linux-example / test (example)*. While the issue is open, the job will continue to run at the current cadence, but will not block *viable/strict* promotion. This requires write-access to the PyTorch repo, or the issue will be auto-closed.

### Core Jobs, Ineligible for Unstable Designation
The following jobs are not eligible for being marked as unstable given the above guidance. At least one lint job, one CPU default test job, and one CUDA default test job must always be present. Absence of signals from the following jobs should block *viable/strict* entirely.
- Lint / lintrunner-noclang-all / lint
- pull / linux-jammy-py3.10-gcc11 / test (default)
- trunk / linux-jammy-cuda13.0-py3.10-gcc11 / test (default)
