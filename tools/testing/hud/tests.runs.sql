-- One row per attempt of one test in one environment in one CI job.
-- See proposal.md section 4.4.

CREATE TABLE tests.runs
(
    -- what ran and how ----------------------------------------------------------

    -- What: which test.
    -- Derived: hash of the producer-emitted identity fields, section 4.1, computed
    --   by the ingester.
    -- Used: leading sort key; join to tests.tests.
    test_id                UInt64,

    -- What: in which environment.
    -- Derived: hash of the environment's identity fields, section 4.1, computed
    --   by the ingester.
    -- Used: second sort key; join to tests.environments.
    env_id                 UInt64,

    -- where it ran; everything else about the job is in default.workflow_job -----

    -- What: the CI job.
    -- Derived: the JOB_ID the job exports, written as a testsuite property.
    -- Used: join to default.workflow_job for commit, branch (and so trunk versus
    --   PR, section 4.4), runner, conclusion and URL, and through its run id to
    --   default.workflow_run for the workflow's path and name; the by_job
    --   projection for per-commit reads.
    github_workflow_job_id Int64,

    -- attempt lineage ------------------------------------------------------------

    -- What: 0 for the first execution of this test in this process, then 1, 2, ...
    --   for each rerun after it (or each repeat in rerun-disabled-tests mode).
    -- Derived: document order in the report: the <rerun> elements, then the
    --   <testcase>.
    -- Used: rerun counts; a restart at 0 marks a new process, and a process that
    --   follows a failed one for the same test is a retry (section 4.1); orders
    --   attempts that share started_at; last sort key, so that ReplacingMergeTree
    --   never merges the reruns of a test into one row.
    rerun_number           UInt16,

    -- outcome ---------------------------------------------------------------------

    -- What: what happened, appendix B. crashed and timed_out are always synthetic
    --   rows written by run_test.py, so no separate source flag exists.
    -- Derived: the JUnit children (failure, error, skipped with its type) for
    --   report rows; run_test.py's exit handling for synthetic rows.
    -- Used: every count in the rollups; the verdict of a job.
    outcome                Enum8('passed' = 1, 'failed' = 2, 'error' = 3, 'skipped' = 4,
                                 'xfailed' = 5, 'xpassed' = 6, 'crashed' = 7,
                                 'timed_out' = 8),

    -- time ------------------------------------------------------------------------

    -- What: when the attempt started and ended; the duration is the difference.
    -- Derived: pytest TestReport.start and .stop, written as attributes of
    --   <testcase> and of each <rerun>; gtest timestamp plus time.
    -- Used: ordering of attempts and verdicts, durations, partitioning, TTL.
    started_at             DateTime64(3, 'UTC'),
    ended_at               DateTime64(3, 'UTC'),

    PROJECTION by_job (
        SELECT * ORDER BY (github_workflow_job_id, test_id, started_at, rerun_number)
    )
)
ENGINE = ReplacingMergeTree
PARTITION BY toDate(started_at)
PRIMARY KEY (test_id, env_id, started_at)
ORDER BY (test_id, env_id, started_at, github_workflow_job_id, rerun_number)
TTL toDateTime(started_at) + INTERVAL 180 DAY
SETTINGS index_granularity = 8192;
