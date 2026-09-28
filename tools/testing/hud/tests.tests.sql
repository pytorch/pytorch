-- One row per test case in a launched test file. See proposal.md section 4.2.

CREATE TABLE tests.tests
(
    -- What: identity of one test case in a launched test file.
    -- Derived: sipHash64(concat(repo, '\0', file, '\0', suite, '\0', case_name)),
    --   computed by the ingester inside the INSERT ... SELECT from the values below.
    -- Used: referenced as test_id by runs, health_daily and health; the hub's
    --   stable URL id.
    id            UInt64,

    -- What: repository the test lives in, e.g. pytorch/pytorch.
    -- Derived: GITHUB_REPOSITORY, written by the producer as a testsuite property.
    -- Used: scopes the catalog when other pytorch org repos join; part of the hash;
    --   with file, the join key to tests.owners.
    repo          LowCardinality(String),

    -- What: the file run_test.py launched to run the test: a repo-relative path
    --   such as test/test_torch.py, or test/test_jit.py for the classes it
    --   imports from test/jit/; for C++ tests, the binary as run_test.py names
    --   it (cpp/test_api).
    -- Derived: the file part of the pytest node id, which is the launched file
    --   even when the class is defined in another module.
    -- Used: owner lookup through tests.owners; per-file timings and rollups; the
    --   report's directory in the job's zip.
    file          LowCardinality(String),

    -- What: test class or gtest suite; empty for module-level functions.
    -- Derived: pytest, the last component of the class path in the nodeid;
    --   gtest, the classname attribute.
    -- Used: display and search; the disable-issue join on "test_x (__main__.Suite)".
    suite         String,

    -- What: test name exactly as the runner selects it, parametrization included,
    --   e.g. test_add_cpu_float32 or test_foo[1-2].
    -- Derived: pytest, the name part of the nodeid; gtest, the name attribute.
    -- Used: display and search; LIKE filters on parameters until generators emit
    --   them as separate properties (a future params column).
    case_name     String,

    -- What: when the ingester first saw the test.
    -- Derived: ingester clock at the insert, which happens only for an unseen id.
    -- Used: first seen for the test; version column of ReplacingMergeTree.
    first_seen_at DateTime
)
ENGINE = ReplacingMergeTree(first_seen_at)
ORDER BY id;
