-- Daily counts per (test, env). See proposal.md section 4.5.

CREATE TABLE tests.health_daily
(
    -- What: calendar day of the attempts.
    -- Derived: toDate(started_at) in the materialized view.
    -- Used: time series; partitioning.
    day              Date,

    -- What: the group: one test variant. Every trigger and workflow is counted
    --   together; per-trigger and per-workflow numbers come from runs joined to
    --   the job.
    -- Derived: copied from runs by the materialized view.
    -- Used: environment-level "where does it run"; per-file timings through
    --   tests.tests.file, the file run_test.py launches.
    test_id          UInt64,
    env_id           UInt64,

    -- What: attempt rows in the group.
    -- Derived: count() per insert block, summed on merge.
    -- Used: denominators.
    attempts         SimpleAggregateFunction(sum, UInt64),

    -- What: attempts that were not the first execution of their test in their
    --   process: in-process reruns, or repeats in rerun-disabled-tests mode.
    -- Derived: countIf(rerun_number > 0).
    -- Used: a cheap flakiness proxy before verdicts are computed.
    reruns           SimpleAggregateFunction(sum, UInt64),

    -- What: attempts by outcome; failed counts failed and error, aborted counts
    --   crashed and timed_out.
    -- Derived: countIf(outcome ...) per insert block, summed on merge.
    -- Used: skip share, failure share, most-skipped and most-failing lists.
    passed           SimpleAggregateFunction(sum, UInt64),
    failed           SimpleAggregateFunction(sum, UInt64),
    skipped          SimpleAggregateFunction(sum, UInt64),
    xfailed          SimpleAggregateFunction(sum, UInt64),
    xpassed          SimpleAggregateFunction(sum, UInt64),
    aborted          SimpleAggregateFunction(sum, UInt64),

    -- What: approximate distinct jobs in the group.
    -- Derived: uniqCombinedState(github_workflow_job_id); read with uniqCombinedMerge.
    -- Used: jobs per day per environment without reading runs.
    jobs             AggregateFunction(uniqCombined, Int64),

    -- What: total and longest attempt time in the group, in seconds.
    -- Derived: dateDiff('millisecond', started_at, ended_at) / 1000, summed and maxed.
    -- Used: cost by file, owner and hardware; slowest tests.
    call_seconds     SimpleAggregateFunction(sum, Float64),
    max_call_seconds SimpleAggregateFunction(max, Float32),

    -- What: newest attempt in the group.
    -- Derived: max(started_at).
    -- Used: "last run" columns; first seen per test variant is min(day) over this table.
    last_started_at  SimpleAggregateFunction(max, DateTime64(3, 'UTC'))
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(day)
ORDER BY (test_id, env_id, day);

CREATE MATERIALIZED VIEW tests.health_daily_mv TO tests.health_daily AS
SELECT
    toDate(started_at) AS day,
    test_id, env_id,
    count() AS attempts,
    countIf(rerun_number > 0) AS reruns,
    countIf(outcome = 'passed') AS passed,
    countIf(outcome IN ('failed', 'error')) AS failed,
    countIf(outcome = 'skipped') AS skipped,
    countIf(outcome = 'xfailed') AS xfailed,
    countIf(outcome = 'xpassed') AS xpassed,
    countIf(outcome IN ('crashed', 'timed_out')) AS aborted,
    uniqCombinedState(github_workflow_job_id) AS jobs,
    sum(dateDiff('millisecond', started_at, ended_at)) / 1000 AS call_seconds,
    max(dateDiff('millisecond', started_at, ended_at)) / 1000 AS max_call_seconds,
    max(started_at) AS last_started_at
FROM tests.runs
GROUP BY day, test_id, env_id;
