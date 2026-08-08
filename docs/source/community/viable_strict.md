---
orphan: true
---

# viable/strict CI jobs
## Background
The [*viable/strict*](https://github.com/pytorch/pytorch/tree/viable/strict) branch ensures all essential PyTorch tests are passing on it. This provides a stable base for contributions and greatly increases confidence that test failures are newly introduced. *viable/strict* should closely track *main*, providing a topline read on trunk health.

All merged commits on main are candidates for becoming the new *viable/strict* head. Tests that must pass for this promotion to succeed are part of the *viable/strict*-blocking set. These tests must be high signal and minimally flaky. No subset of these tests should dominate advancement failures.

## Blocking Set Requirements
The *viable/strict*-blocking set is encompassed by `trunk`, `pull`, `lint`, and `docs-build`. Jobs that are part of these workflows should meet the following requirements:

Let **Failures %** (given a job, over an N-day window) be defined as the percent of commits where this job was red, over the decided main commits in the provided window.
1. A job should only be added to the *viable/strict*-blocking set if it meets **Failures % less than 10% over the last 7 days**. Furthermore, jobs added to the frequently-run `pull` workflow should not cause `pull` runtime to exceed 90 minutes. The job owner is responsible for ensuring these requirements are met.
2. *viable/strict*-blocking jobs that exceed **Failures % greater than 15% over the last 7 days should be removed from the viable/strict-blocking set**, outside of sustained breakages across the *viable/strict*-blocking set. This will be maintained by the PyTorch Dev Infra team.

Job health can be viewed at HUD [/reliability](https://hud.pytorch.org/reliability) or computed manually on [the trunk dashboard](https://hud.pytorch.org/hud/pytorch/pytorch/main).

### Core Jobs, Ineligible for Unstable Designation
The following jobs are not eligible for being marked as unstable given the above guidance. At least one CPU default and one CUDA default test job must always be present. Absence of signals from the following jobs should block *viable/strict* entirely.
- pull / linux-jammy-py3.10-gcc11 / test (default)
- trunk / linux-jammy-cuda13.0-py3.10-gcc11 / test (default)

### Testing New Jobs
New jobs should be introduced and in the [*unstable.yml*](https://github.com/pytorch/pytorch/blob/main/.github/workflows/unstable.yml) workflow and monitored.

### Marking Jobs as Unstable
The UNSTABLE GitHub Issue template can be used to mark a job as unstable. The job will continue to run at the current cadence, but will not block *viable/strict* promotion. Alternatively, the job can be relocated to *unstable.yml* if more involved fixes are necessary.
