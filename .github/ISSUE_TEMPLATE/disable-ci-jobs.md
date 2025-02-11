---
name: Disable CI jobs (PyTorch Dev Infra only)
about: Use this template to disable CI jobs
title: "DISABLED [WORKFLOW_NAME] / [PLATFORM_NAME] / [JOB_NAME]"
labels: "module: ci"
---

> For example, DISABLED pull / win-vs2019-cpu-py3 / test (default). Once
> created, the job will be disabled within 15 minutes. You can check the
> list of disabled jobs at https://ossci-metrics.s3.amazonaws.com/disabled-jobs.json

> If you need to get this out ASAP instead of waiting for 15 minutes,
> you can manually trigger the workflow at https://github.com/pytorch/test-infra/actions/workflows/update_disabled_tests.yml
> once the issue is created to update the above JSON list right away.

> Noted: you need to have write access to PyTorch repo to disable CI
> jobs. The issue will be rejected otherwise.

## Reason
*Provide a reason why this is needed and when this can be resolved*.
