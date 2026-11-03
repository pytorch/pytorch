---
name: ci-metrics
description: Query PyTorch CI, GitHub Actions, HUD, Grafana, and infrastructure metrics. Use when users ask about CI duration, job failures, queue times, workflow trends, runner health, dashboard data, or PyTorch infrastructure metrics.
---

# Requirements

PyTorch CI data lives in HUD (`hud.pytorch.org`) and Grafana (`pytorchci.grafana.net`). Both are read only and gated by GitHub identity: the user must have write access to `pytorch/pytorch` or be on HUD's allow list.

This skill needs these tools on PATH:
  * `gh` — must be authenticated; if not, run `gh auth login --hostname github.com --git-protocol ssh --web`. Its token authenticates the Tests API and is exchanged for the Grafana token.
  * `curl` — calls the Tests API, downloads `gcx`, and fetches the Grafana token from HUD.

Use [.agents/skills/ci-metrics/gcx-wrapper.sh](gcx-wrapper.sh) for all Grafana access. On first use it downloads a pinned, checksum-verified `gcx` binary into a private cache (`~/.cache/pytorch-ci-metrics/`) and authenticates with the user's `gh` session; nothing is installed on PATH. If a tool is missing or `gh` is not authenticated, it exits with an error describing what to fix.

# Datasources

Get the list of datasources available:

```
.agents/skills/ci-metrics/gcx-wrapper.sh datasources list
```

Pass the `uid` from this list to `-d`. [gcx 1.7.1 interprets this value directly as a UID](https://github.com/grafana/gcx/blob/v1.7.1/internal/datasources/query/resolve.go#L17-L25); it does not resolve display names. A display name therefore requests the wrong UID and can produce a misleading `datasources:read` permission error. The two that matter:
  * `ceczcsck1b20wb` (name `grafana-clickhouse-datasource`) — CI and test data in ClickHouse.
  * `grafanacloud-prom` is the UID (name `grafanacloud-pytorchci-prom`) — CI infrastructure metrics in Prometheus.

The data contains metrics for many repos in the PyTorch organization. When possible, restrict queries to just the `pytorch/pytorch` repository.

# CI and Tests

PyTorch CI saves test metrics. To access this data prefer this order:
  * Tests API — standard queries about tests, their health, runs and owners. The OpenAPI schema at `https://hud.pytorch.org/api/authed/openapi.json` is the reference for parameters, defaults and response shapes. <!-- @lint-ignore -->
    * Every request needs a GitHub token. Pass the authorization header through stdin to keep the token out of process arguments:
      ```
      gh auth token | sed 's/^/Authorization: Bearer /' | \
        curl -fsSL -H @- 'https://hud.pytorch.org/api/authed/tests?repo=pytorch/pytorch&limit=10' # @lint-ignore: requires authentication
      ```
    * Prefix every path with `/api/authed/`: `/api/tests` becomes `/api/authed/tests`. The unprefixed path answers CLI callers with a 429 bot challenge.
    * Responses carry the HUD page that shows them as `hud_url`: on each test, on each list entry, and on a list or grid as a whole. Give the user these links beside the API results and suggest opening them: the pages are the easier way for a person to see a test's health and history. A null `hud_url` means no page shows that selection; give the per-test links instead. The pages are listed in the OpenAPI document under the `pages` tag with the parameters they take. Never fetch a page from a program, and never prefix a page URL with `/api/authed/`.
  * ClickHouse — if the API is insufficient, query the database directly using [.agents/skills/ci-metrics/gcx-wrapper.sh](gcx-wrapper.sh):
    ```
    .agents/skills/ci-metrics/gcx-wrapper.sh datasources clickhouse list-tables -d ceczcsck1b20wb --database ci
    .agents/skills/ci-metrics/gcx-wrapper.sh datasources clickhouse describe-table runs -d ceczcsck1b20wb --database ci
    .agents/skills/ci-metrics/gcx-wrapper.sh datasources clickhouse query -d ceczcsck1b20wb "SELECT ..."
    ```
    * Important dataset:
      * GitHub webhook data
        * Database: default
          * Note: the default database also contains other non-webhook related tables.
        * Learn more about the event and payload: https://docs.github.com/en/webhooks/webhook-events-and-payloads
      * Tests
        * Database: ci — the tables behind the Tests API. `runs` holds one row per test attempt and billions of rows, so always filter by time. Read `tests`, `environments` and `env_vars` with `FINAL` (`FROM ci.tests FINAL`), including in joins, or you get duplicate rows.
        * Do not use the `tests` database (`all_test_runs` and friends). It is the legacy store; answer every test question from `ci`.

To get additional guidance on common queries, clone https://github.com/pytorch/test-infra into a temporary directory: HUD's saved queries are in `torchci/clickhouse_queries/<name>/query.sql` and the table definitions with column docs are in `clickhouse_db_schema/`.

## Example Queries

Within `pytorch/pytorch` on main, list the top failing workflow jobs in the last two weeks:
```
.agents/skills/ci-metrics/gcx-wrapper.sh datasources clickhouse query -d ceczcsck1b20wb "
  SELECT name, count(DISTINCT id) AS failures
  FROM default.workflow_job
  WHERE conclusion = 'failure'
    AND completed_at >= now() - INTERVAL 2 WEEK
    AND repository_full_name = 'pytorch/pytorch'
    AND head_branch = 'main'
  GROUP BY name ORDER BY failures DESC LIMIT 10"
```

For a test file, count each test's attempts in the last day, including retries. Failures include errors, crashes and timeouts; `other` counts expected failures, unexpected passes and unknown outcomes:
```
.agents/skills/ci-metrics/gcx-wrapper.sh datasources clickhouse query -d ceczcsck1b20wb "
  SELECT t.file, t.suite, t.case_name, count() AS attempts,
    countIf(r.outcome = 'passed') AS passed,
    countIf(r.outcome IN ('failed', 'error', 'crashed', 'timed_out')) AS failed,
    countIf(r.outcome = 'skipped') AS skipped,
    countIf(r.outcome IN ('xfailed', 'xpassed', 'unknown')) AS other
  FROM ci.runs AS r
  JOIN ci.tests AS t FINAL ON r.tests_id = t.id
  WHERE r.started_at >= now() - INTERVAL 1 DAY
    AND t.repo = 'pytorch/pytorch'
    AND t.file = 'test/test_torch.py'
  GROUP BY t.file, t.suite, t.case_name
  ORDER BY attempts DESC LIMIT 20"
```

# CI Infrastructure

CI infrastructure metrics are in the Prometheus datasource (`-d grafanacloud-prom`). To get a better understanding of the data, clone these repositories in a temporary directory:
  * https://github.com/pytorch/ci-infra - In `/osdc` contains the code for OSDC, the infra running PyTorch's CI. Read it to understand how metrics are exported and what metrics are available. Read the docs in /osdc/docs (start with `observability.md`) to understand the scope and project setup.
  * https://github.com/jeanschmidt/actions-runner-controller - To understand how actions-runner-controller exposes data.

## Example Queries

Which runner types have the deepest queue right now (jobs assigned but not yet running)?
```
.agents/skills/ci-metrics/gcx-wrapper.sh datasources prometheus query -d grafanacloud-prom 'topk(10, clamp_min(sum by (name) (gha_assigned_jobs) - sum by (name) (gha_running_jobs), 0))'
```

How many jobs were running per cluster over the last 6 hours, sampled every 30 minutes? Use `--since`/`--step` (or `--from`/`--to`) for a range query:
```
.agents/skills/ci-metrics/gcx-wrapper.sh datasources prometheus query -d grafanacloud-prom 'sum by (cluster) (gha_running_jobs)' --since 6h --step 30m
```
