---
name: ci-metrics
description: Query PyTorch CI, GitHub Actions, HUD, Grafana, and infrastructure metrics. Use when users ask about CI duration, job failures, queue times, workflow trends, runner health, dashboard data, or PyTorch infrastructure metrics.
---

# PyTorch CI Metrics

PyTorch CI and infrastructure metrics are exposed through Grafana. Use [.claude/skills/ci-metrics/gcx-wrapper.sh](gcx-wrapper.sh) for all Grafana access; it configures the PyTorch Grafana server, context, and authentication. Only users with write permission to the repo have access to Grafana. The authentication only provides read only access.

## Requirements

The wrapper needs these tools on PATH:
  * `gh` - fetches the Grafana token and must be authenticated; if not, run `gh auth login --hostname github.com --git-protocol ssh --web`.
  * `curl` - downloads `gcx` and fetches the token from HUD.

On first use the wrapper downloads a pinned, checksum-verified `gcx` binary into a private cache (`~/.cache/pytorch-ci-metrics/`) and authenticates automatically. Nothing is installed on your PATH. If a tool is missing or `gh` is not authenticated, it exits with an error describing what to fix.

## Datasources

Get the list of datasources available:

```
.claude/skills/ci-metrics/gcx-wrapper.sh datasources list
```

The data contains metrics for many repos owned by the PyTorch repo. When possible, restrict queries to just the `pytorch/pytorch` repository.

## CI and Test Run Data

CI and test run data are stored in `grafana-clickhouse-datasource`. List all the available tables:

```
.claude/skills/ci-metrics/gcx-wrapper.sh datasources clickhouse list-tables
```

Important dataset:
  * GitHub webhook data
    * Database: default
      * Note: the default database also contains other non-webhook related tables.
    * Learn more about the event and payload: https://docs.github.com/en/webhooks/webhook-events-and-payloads
  * Tests
    * Database: tests
    * tests.all_test_runs - contains every test run. This is an extremely large table, so be considerate with filtering and timing.
    * Do not use tests.test_run_s3 as it contains partial data only.

To get additional guidance on common queries, clone https://github.com/pytorch/test-infra into a temporary directory and read the `torchci` folder.

### Example Queries

Within the pytorch/pytorch repo on main, list the top most failing workflow jobs in the last 2 weeks:
```
.claude/skills/ci-metrics/gcx-wrapper.sh datasources clickhouse query "
  SELECT name, count(DISTINCT id) AS failures
  FROM default.workflow_job
  WHERE conclusion = 'failure'
    AND completed_at >= now() - INTERVAL 2 WEEK
    AND repository_full_name = 'pytorch/pytorch'
    AND head_branch = 'main'
  GROUP BY name ORDER BY failures DESC LIMIT 10"
```

For a test file, how many times was it run in the last week? How many times did it pass or fail?
```
.claude/skills/ci-metrics/gcx-wrapper.sh datasources clickhouse query "
  SELECT
    file,
    classname,
    name,
    count() AS runs,
    countIf(failure_count = 0 AND error_count = 0 AND skipped_count = 0) AS successful,
    countIf(failure_count > 0 OR error_count > 0) AS fails,
    countIf(skipped_count > 0) AS skipped
  FROM tests.all_test_runs
  WHERE time_inserted >= now() - INTERVAL 7 DAY
    AND file = 'lazy/test_ts_opinfo.py'
  GROUP BY file, classname, name
  ORDER BY runs DESC"
```

## CI Infrastructure

CI infrastructure metrics are stored in `grafanacloud-pytorchci-prom`. To get a better understanding of the data, clone these repositories in a temporary directory:
  * https://github.com/pytorch/ci-infra - In `/osdc` contains the code for OSDC, the infra running PyTorch's CI. Read it to understand how metrics are exported and what metrics are available. Read the docs in /osdc/docs to understand the scope and project setup.
  * https://github.com/jeanschmidt/actions-runner-controller - To understand how actions-runner-controller exposes data.

### Example Queries

Which runner types have the deepest queue right now (jobs assigned but not yet running)?
```
.claude/skills/ci-metrics/gcx-wrapper.sh datasources prometheus query -d grafanacloud-prom 'topk(10, clamp_min(sum by (name) (gha_assigned_jobs) - sum by (name) (gha_running_jobs), 0))'
```

How many jobs were running per cluster over the last 6 hours, sampled every 30 minutes? Use `--since`/`--step` (or `--from`/`--to`) for a range query:
```
.claude/skills/ci-metrics/gcx-wrapper.sh datasources prometheus query -d grafanacloud-prom 'sum by (cluster) (gha_running_jobs)' --since 6h --step 30m
```
