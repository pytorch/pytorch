-- Verdict metrics per (test, env, trunk or PR, window), recomputed on a schedule.
-- See proposal.md section 4.6.

CREATE TABLE tests.health
(
    -- What: the test variant, whether trunk or PR runs were counted, and the
    --   window in days (7 or 30).
    -- Derived: the scheduled job's GROUP BY; trunk and PR come from the job's
    --   head_branch in default.workflow_job (proposal.md section 4.4).
    -- Used: the hub's list, search and detail pages read these rows.
    test_id                    UInt64,
    env_id                     UInt64,
    ref_class                  Enum8('trunk' = 1, 'pr' = 2),
    window_days                UInt8,

    -- What: when the job computed the row.
    -- Derived: job clock.
    -- Used: ReplacingMergeTree version; staleness display.
    computed_at                DateTime,

    -- What: jobs in the window with at least one attempt.
    -- Derived: count() of job verdicts.
    -- Used: denominators; "does it run at all".
    jobs                       UInt32,

    -- What: jobs whose last attempt failed, errored, crashed or timed out.
    -- Derived: the verdict is in the failing set.
    -- Used: failure rate.
    failed_jobs                UInt32,

    -- What: jobs with a failing attempt and a passing last attempt.
    -- Derived: verdict passed and bad_attempts > 0.
    -- Used: flake rate; the flaky bot's input.
    flaky_jobs                 UInt32,

    -- What: jobs whose last attempt was skipped.
    -- Derived: verdict skipped.
    -- Used: "where is it skipped".
    skipped_jobs               UInt32,

    -- What: length of the current failing streak, newest job first.
    -- Derived: the array expression in the query below, over verdicts ordered by time.
    -- Used: "failing since when"; trunk-state thresholds.
    consecutive_failing_jobs   UInt16,

    -- What: the earliest failing job in the window, by job time today and by push
    --   time once that join exists (open decision, section 8).
    -- Derived: argMinIf over failing verdicts joined to workflow_job.
    -- Used: the starting point for blame.
    first_failing_sha          FixedString(40),
    first_failing_at           Nullable(DateTime),

    -- What: attempt duration quantiles over passing attempts.
    -- Derived: quantiles(0.5, 0.95) over runs in the window.
    -- Used: slow-test lists; a replacement for slow_tests.json.
    p50_seconds                Float32,
    p95_seconds                Float32,

    -- What: newest attempt, and newest passing attempt, in the window.
    -- Derived: max(started_at) and maxIf(started_at, verdict = 'passed').
    -- Used: "silently stopped running"; "last green".
    last_seen                  DateTime,
    last_passed                Nullable(DateTime)
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (test_id, env_id, ref_class, window_days);

-- The verdict query the scheduled job runs, shown for the 7-day trunk window.
-- A trunk job is one whose head_branch is main or a trunk/<sha> tag.
WITH trunk_jobs AS (
    SELECT id, head_sha
    FROM default.workflow_job
    WHERE started_at >= now() - INTERVAL 8 DAY
      AND (head_branch = 'main' OR head_branch LIKE 'trunk/%')
),
verdicts AS (
    SELECT
        test_id, env_id, github_workflow_job_id AS job,
        argMax(outcome, (started_at, rerun_number)) AS verdict,
        countIf(outcome IN ('failed', 'error', 'crashed', 'timed_out')) AS bad_attempts,
        max(started_at) AS at_time
    FROM tests.runs
    WHERE started_at >= now() - INTERVAL 7 DAY
      AND github_workflow_job_id IN (SELECT id FROM trunk_jobs)
    GROUP BY test_id, env_id, job
)
SELECT
    v.test_id, v.env_id,
    count() AS jobs,
    countIf(verdict IN ('failed', 'error', 'crashed', 'timed_out')) AS failed_jobs,
    countIf(verdict = 'passed' AND bad_attempts > 0) AS flaky_jobs,
    countIf(verdict = 'skipped') AS skipped_jobs,
    -- verdicts newest first; count leading failures, stopping at the first pass or skip
    arrayFirstIndex(x -> NOT x,
        arrayPushBack(
            arrayMap(t -> t.2 IN ('failed', 'error', 'crashed', 'timed_out'),
                arrayReverseSort(t -> t.1, groupArray((at_time, verdict)))),
            0)) - 1 AS consecutive_failing_jobs,
    argMinIf(j.head_sha, at_time, verdict IN ('failed', 'error', 'crashed', 'timed_out')) AS first_failing_sha
FROM verdicts AS v
LEFT JOIN trunk_jobs AS j ON j.id = v.job
GROUP BY v.test_id, v.env_id;
